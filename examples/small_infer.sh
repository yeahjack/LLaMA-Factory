#!/usr/bin/env bash
set -euo pipefail

# 可选：从脚本所在目录跳到项目根（原脚本里有 `cd ..`）
cd "$(dirname "$0")/.."

############################################
# 模型选择：按键设置模型与目录
############################################
set_model() {
  local key="$1"
  case "$key" in
    "qwen25_7b")
      BASE_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
      TEMPLATE="qwen"
      MODEL_DIR="results_qwen25_7b"
      ;;
    "qwen25_14b")
      BASE_MODEL_PATH="Qwen/Qwen2.5-14B-Instruct"
      TEMPLATE="qwen"
      MODEL_DIR="results_qwen25_14b"
      ;;
    "llama32_3b")
      BASE_MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"
      TEMPLATE="llama3"
      MODEL_DIR="results_llama32_3b"
      ;;
    "llama3_8b")
      BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
      TEMPLATE="llama3"
      MODEL_DIR="results_llama3_8b"
      ;;
    *)
      echo "❌ 未知的模型键: ${key}"
      exit 1
      ;;
  esac
  RESULTS_BASE_DIR="${MODEL_DIR}"
  MODEL_SHORT="${MODEL_DIR#results_}"
  PRECOMPUTE_RESULTS_DIR="results_${MODEL_SHORT}/base_sys_prompt/"
  export WANDB_PROJECT="TTL_${MODEL_SHORT}"
  echo "==> 选择模型: ${key}"
  echo "    - BASE_MODEL_PATH=${BASE_MODEL_PATH}"
  echo "    - TEMPLATE=${TEMPLATE}"
  echo "    - RESULTS_BASE_DIR=${RESULTS_BASE_DIR}"
  echo "    - PRECOMPUTE_RESULTS_DIR=${PRECOMPUTE_RESULTS_DIR}"
  echo "    - WANDB_PROJECT=${WANDB_PROJECT}"
  echo ""
}

############################################
# 默认参数（可被 env 或命令行覆盖）
############################################
MODEL_KEY="${MODEL_KEY:-qwen25_7b}"      # 模型键：qwen25_7b / qwen25_14b / llama32_3b / llama3_8b
EVAL_SLOTS_PER_GPU="${EVAL_SLOTS_PER_GPU:-10}"  # 评测阶段同一 GPU 的并发 worker 数

TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
TOP_K="${TOP_K:--1}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-250}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# 解析命令行
GPUS_CSV="${GPUS:-}"       # 也可用环境变量 GPUS="0,1,2,4"
DATASETS_CSV="${DATASETS:-}"  # 或环境变量 DATASETS="gsm8k_5k,logiqa_5k,meta_math_5k"

while getopts ":k:g:d:" opt; do
  case $opt in
    k) MODEL_KEY="$OPTARG" ;;
    g) GPUS_CSV="$OPTARG" ;;
    d) DATASETS_CSV="$OPTARG" ;;
    *) ;;
  esac
done
shift $((OPTIND -1))

# 设定模型
set_model "${MODEL_KEY}"

############################################
# 数据集与 GPU 列表
############################################
# 缺省数据集
datasets=(
  "agriculture_5k"
  "geosignal_5k"
  "gen_med_gpt_5k"
  "wealth_5k"
  "alpaca_gpt4_5k"
  "instruction_wild_5k"
  "dolly_5k"
  "gsm8k_5k"
  "logiqa_5k"
  "meta_math_5k"
)

# 缺省 GPU
gpus=(0)

# 若用 -d 或 DATASETS 覆盖
if [[ -n "${DATASETS_CSV}" ]]; then
  IFS=',' read -r -a datasets <<< "${DATASETS_CSV}"
fi
# 若用 -g 或 GPUS 覆盖
if [[ -n "${GPUS_CSV}" ]]; then
  IFS=',' read -r -a gpus <<< "${GPUS_CSV}"
fi

############################################
# 路径与日志
############################################
OUT_DIR="${MODEL_DIR}/base_sys_prompt"
LOG_DIR="logs/${MODEL_SHORT}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

############################################
# FIFO 清理
############################################
infer_fifo=""
eval_fifo=""
cleanup() {
  [[ -n "${infer_fifo}" && -p "${infer_fifo}" ]] && rm -f "${infer_fifo}" || true
  [[ -n "${eval_fifo}" && -p "${eval_fifo}"   ]] && rm -f "${eval_fifo}"   || true
}
trap cleanup EXIT

############################################
# 工具函数：队列与 worker
############################################
feed_queue() {
  local fifo_path="$1"; shift
  (
    for item in "$@"; do
      echo "$item"
    done
  ) > "${fifo_path}" &
}

infer_worker() {
  local gpu="$1"
  local fifo_path="$2"
  while IFS= read -r dataset || [[ -n "${dataset:-}" ]]; do
    [[ -z "${dataset}" ]] && continue
    local out_jsonl="${OUT_DIR}/${dataset}.jsonl"
    local log_file="${LOG_DIR}/infer_${dataset}.log"
    echo "[infer][GPU ${gpu}] ${dataset} -> ${out_jsonl}"
    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/vllm_infer.py \
      --model_name_or_path "${BASE_MODEL_PATH}" \
      --dataset "${dataset}" \
      --template "${TEMPLATE}" \
      --save_name "${out_jsonl}" \
      --temperature "${TEMPERATURE}" \
      --top_p "${TOP_P}" \
      --top_k "${TOP_K}" \
      --seed "${SEED}" \
      --batch_size "${BATCH_SIZE}" \
      --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      > "${log_file}" 2>&1
  done < "${fifo_path}"
}

eval_worker() {
  local gpu="$1"
  local fifo_path="$2"
  while IFS= read -r dataset || [[ -n "${dataset:-}" ]]; do
    [[ -z "${dataset}" ]] && continue
    local input_file="${OUT_DIR}/${dataset}.jsonl"
    local output_file="${OUT_DIR}/${dataset}_metrics.json"
    local log_file="${LOG_DIR}/eval_${dataset}.log"
    echo "[eval ][GPU ${gpu}] ${dataset} -> ${output_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/eval_ttl_aligned.py \
      --filename "${input_file}" \
      --output_filename "${output_file}" \
      --metrics "bertscore,rouge,bleu,em" \
      > "${log_file}" 2>&1
  done < "${fifo_path}"
}

############################################
# 阶段 1：推理（每 GPU 1 个并发，轮番排队）
############################################
infer_fifo="$(mktemp -u)"
mkfifo "${infer_fifo}"
feed_queue "${infer_fifo}" "${datasets[@]}"

for gpu in "${gpus[@]}"; do
  infer_worker "${gpu}" "${infer_fifo}" &
done
wait
rm -f "${infer_fifo}"; infer_fifo=""

############################################
# 阶段 2：评测（每 GPU 最多 EVAL_SLOTS_PER_GPU 个并发）
############################################
eval_fifo="$(mktemp -u)"
mkfifo "${eval_fifo}"
feed_queue "${eval_fifo}" "${datasets[@]}"

for gpu in "${gpus[@]}"; do
  for ((slot=1; slot<=EVAL_SLOTS_PER_GPU; slot++)); do
    eval_worker "${gpu}" "${eval_fifo}" &
  done
done
wait
rm -f "${eval_fifo}"; eval_fifo=""

echo "✅ 全部数据集推理与评测已完成：${OUT_DIR}"