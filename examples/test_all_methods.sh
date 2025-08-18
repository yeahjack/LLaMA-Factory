#!/bin/bash
# 更稳健的设置：捕获管道中任何一步的失败；其余处通过局部处理避免整脚本退出
set -e
set -o pipefail

echo "==> 切换到 LLaMA-Factory 根目录..."
cd "../" || exit 1
echo "==> 当前目录: $(pwd)"
echo ""

export OMP_NUM_THREADS=16
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- 清理与健壮性：中断或错误时尽量清理后台任务 ---
cleanup() {
  echo "==> 捕获到中断/错误，尝试终止后台任务..."
  # 仅终止仍在运行的后台作业；忽略失败
  jobs -pr | xargs -r kill 2>/dev/null || true
}
trap cleanup INT TERM ERR

# --- 依赖自检：尽早发现环境问题（这些缺失属于环境错误，可直接退出） ---
require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "❌ 缺少命令：$1"; exit 1; }; }
require_cmd llamafactory-cli
require_cmd python
require_cmd tee

# =========== 通用配置 ===========
# 如需切换 YAML，在此修改；若不同模型需要不同 YAML，可在 set_model 内按需扩展
BASE_YAML="examples/train_ttl/qwen25_ttl.yaml"

# ===== 模型选择（按需在 models 中开启）=====
# 可用键：
#   qwen25_7b   -> Qwen/Qwen2.5-7B-Instruct, dir: results_qwen25_7b, template: qwen
#   llama32_3b  -> meta-llama/Llama-3.2-3B-Instruct, dir: results_llama32_3b, template: llama3
#   llama3_8b   -> meta-llama/Llama-3.1-8B-Instruct,  dir: results_llama3_8b, template: llama3
models=(
  "qwen25_7b"
  # "llama32_3b"
  # "llama3_8b"
)

# 推理公共选项
NO_DEFAULT_SYSTEM_PROMPT="true"
GPU_MEMORY_UTILIZATION="0.92"

### 流程控制开关 ###
DO_TRAIN="false"
DO_INFER="true"
DO_EVAL="true"
MAX_TRAIN_JOBS_PER_GPU=2
MAX_EVAL_JOBS_PER_GPU=40

# 任务启动节流
ENABLE_LAUNCH_DELAY="true"
LAUNCH_DELAY_SECONDS=3

# =========== TTL 设置（与 YAML 对齐，使用列表进行组合实验） ===========
TTL_SETTING_LIST=("offline_ttl")
TTL_REF_MODE_LIST=("precompute" "simultaneous")
TTL_REF_BATCH_SIZE_LIST=(64)
TTL_ENABLE_INFERENCE_LIST=("false")
TTL_THRESHOLD_LIST=(3)
TTL_SCALER_LIST=(0.1)
TTL_STREAMING_BATCH_SIZE_LIST=(100)

# =========== 实验变量配置 ===========
methods=(
  # "base"   # 仅推理；不使用LoRA；结果保存到 <RESULTS_BASE_DIR>/base
  "ttlu"
  "ttl"
  # "eata"
  # "eata_sdiv"
  # "tent"
  # "ttltent"
  # "ttltent_ppl_nll"
  # "ttltent_nll_nll"
  # "sft"
)

generation_lens=(
  0
  # 80
)

datasets=(
  # "db_geosignal"
  # "db_agriculture"
  # "db_genmedgpt"
  # "db_wealth"
  # "ib_dolly"
  # "ib_alpaca"
  # "ib_instructionwild"
  # "rb_gsm8k"
  # "rb_metamath"
  # "rb_logiqa"
  "agriculture_5k"
  # "geosignal_5k"
  # "gen_med_gpt_5k"
  # "wealth_5k"
  # "alpaca_gpt4_5k"
  # "instruction_wild_5k"
  # "dolly_5k"
  # "gsm8k_5k"
  # "logiqa_5k"
  # "meta_math_5k"
)

gpus=(0)
USE_FULL_ENTROPY_IN_GENERATION="false"

# =========== EATA / EM-FT / TTLTENT 相关变量 ===========
EATA_SELECT_HIGH_ENTROPY="false"
USE_EMFT_LOSS="false"

LOSS_BALANCING_METHOD="moving_average"
ALTERNATING_TRAINING="false"
USE_KL_REGULARIZATION="false"
KL_WEIGHT="0.04"

# =========== 日志根目录按日期分组 ===========
LOG_DATE="$(date +%F)"
LOG_ROOT="logs/${LOG_DATE}"

# =========== 模型选择（由 set_model 设置的当前模型变量） ===========
# 不预设空值，避免无用占位；在 set_model 中赋值
BASE_MODEL_PATH=""      # 例：Qwen/Qwen2.5-7B-Instruct
TEMPLATE=""             # 例：qwen / llama3
MODEL_DIR=""            # 例：results_qwen25_7b
MODEL_SHORT=""          # 例：qwen25_7b (MODEL_DIR 去掉 results_)
RESULTS_BASE_DIR=""     # 与 MODEL_DIR 相同

set_model() {
  local key="$1"
  case "$key" in
    "qwen25_7b")
      BASE_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
      TEMPLATE="qwen"
      MODEL_DIR="results_qwen25_7b"
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
  export WANDB_PROJECT="TTL_${MODEL_SHORT}"
  echo "==> 选择模型: ${key}"
  echo "    - BASE_MODEL_PATH=${BASE_MODEL_PATH}"
  echo "    - TEMPLATE=${TEMPLATE}"
  echo "    - RESULTS_BASE_DIR=${RESULTS_BASE_DIR}"
  echo "    - WANDB_PROJECT=${WANDB_PROJECT}"
  echo ""
}

# =========== 辅助函数 ===========
sanitize_tag() { sed 's/\./p/g' <<< "$1"; }

# 标记当前TTL配置，用于保存路径区分
get_ttl_tag() {
  local sc; sc="$(sanitize_tag "${TTL_SCALER}")"
  echo "ttl-${TTL_SETTING}_ref-${TTL_REF_MODE}_thr${TTL_THRESHOLD}_sc${sc}"
}

get_stage_for_method() {
  local m="$1"
  if [[ "$m" == nll* || "$m" == ppl* ]]; then
    echo "ttlu"
  elif [[ "$m" == "eata_sdiv" ]]; then
    echo "eata"
  elif [[ "$m" == ttltent* ]]; then
    echo "ttltent"
  else
    echo "${m}"
  fi
}

method_uses_generation() {
  local m="$1"
  [[ "$m" == "eata" || "$m" == "eata_sdiv" || "$m" == "tent" || "$m" == ttltent* ]]
}

method_is_pure_ttl() {
  local m="$1"
  [[ "$m" == nll* || "$m" == ppl* || "$m" == "ttlu" ]]
}

get_suffix() {
  local method="$1"
  local gen_len="$2"
  local suffix=""

  if method_uses_generation "${method}"; then
    if [[ "${USE_FULL_ENTROPY_IN_GENERATION}" == "true" && "${gen_len}" -gt 0 ]]; then
      suffix+="_fullent"
    fi
    if [[ ("${method}" == "eata" || "${method}" == "eata_sdiv") && "${EATA_SELECT_HIGH_ENTROPY}" == "true" ]]; then
      suffix+="_highent"
    fi
  fi

  if [[ "${method}" != "sft" && "${USE_EMFT_LOSS}" == "true" ]]; then
    suffix+="_emft"
  fi

  if [[ "$method" == ttltent* ]]; then
    case "${LOSS_BALANCING_METHOD}" in
      "gradient_magnitude") suffix+="_GM" ;;
      "dynamic_weight")     suffix+="_DW" ;;
      "uncertainty")        suffix+="_UC" ;;
      "moving_average")     suffix+="_MA" ;;
      "adaptive_scaling")   suffix+="_AS" ;;
      "static")             suffix+="_ST" ;;
      *)                    suffix+="_${LOSS_BALANCING_METHOD}" ;;
    esac

    if [[ "${ALTERNATING_TRAINING}" == "true" ]]; then
      suffix+="_alt"
    else
      suffix+="_seq"
    fi

    if [[ "${USE_KL_REGULARIZATION}" == "true" ]]; then
      suffix+="_kl${KL_WEIGHT}"
    else
      suffix+="_nokl"
    fi
  fi

  echo "${suffix}"
}

# 逗号拼接（避免依赖 jq）
join_by_comma() {
  local IFS=","
  echo "$*"
}

# ========= 训练 =========
run_train() {
  local method="$1" dataset="$2" gen_len="$3" gpu_id="$4" lr="$5"

  if [[ "$method" == "base" ]]; then
    echo "⏩ [GPU ${gpu_id}] 跳过 base 方法训练（仅推理）。"
    return 0
  fi

  local suffix; suffix="$(get_suffix "$1" "$3")"
  local ttl_tag; ttl_tag="$(get_ttl_tag)"

  local base_save_root="saves/${MODEL_SHORT}/${ttl_tag}"
  local output_dir="${base_save_root}/${method}_search${suffix}/${dataset}_${gen_len}"
  local run_name="${dataset}_${method}_${gen_len}${suffix}"
  mkdir -p "${output_dir}"

  local log_dir="${LOG_ROOT}/${MODEL_SHORT}/${method}/train${suffix}"
  mkdir -p "${log_dir}"
  local log_file="${log_dir}/${dataset}_${gen_len}.log"

  local stage_to_run; stage_to_run="$(get_stage_for_method "${method}")"

  local train_args=(
    "stage=${stage_to_run}"
    "dataset=${dataset}"
    "output_dir=${output_dir}"
    "run_name=${run_name}"
    "learning_rate=${lr}"
    # 显式指定模型与模板，减少对 YAML 的隐式依赖
    "model_name_or_path=${BASE_MODEL_PATH}"
    "template=${TEMPLATE}"
    # 与 YAML 对齐的参数名
    "ttl_setting=${TTL_SETTING}"
    "ttl_ref_mode=${TTL_REF_MODE}"
    "ttl_ref_batch_size=${TTL_REF_BATCH_SIZE}"
    "ttl_direct_inference=${TTL_ENABLE_INFERENCE}"
    "ttl_threshold=${TTL_THRESHOLD}"
    "ttl_sample_efficiency_scaler=${TTL_SCALER}"
    "ttl_streaming_batch_size=${TTL_STREAMING_BATCH_SIZE}"
  )

  # 注意：ttl_loss 已移除，这里不再传递任何 ttl_loss 相关参数

  if method_uses_generation "${method}"; then
    train_args+=("generation_len=${gen_len}")
    if [[ "${USE_FULL_ENTROPY_IN_GENERATION}" == "true" ]]; then
      train_args+=("use_full_entropy_in_generation=true")
    fi
    if [[ "${method}" == "eata_sdiv" ]]; then
      train_args+=("eata_use_sdiv=true")
    fi
    if [[ ("${method}" == "eata" || "${method}" == "eata_sdiv") && "${EATA_SELECT_HIGH_ENTROPY}" == "true" ]]; then
      train_args+=("eata_select_high_entropy=true")
    fi
  fi

  if [[ "${stage_to_run}" == "ttltent" ]]; then
    train_args+=("loss_balancing_method=${LOSS_BALANCING_METHOD}")
    train_args+=("alternating_training=${ALTERNATING_TRAINING}")
    train_args+=("use_kl_regularization=${USE_KL_REGULARIZATION}")
    train_args+=("kl_weight=${KL_WEIGHT}")
  fi

  if [[ "${stage_to_run}" != "sft" && "${USE_EMFT_LOSS}" == "true" ]]; then
    train_args+=("use_emft_loss=true")
  fi

  (
    {
      echo "==> [GPU ${gpu_id}] 启动训练: ${run_name} (model=${MODEL_SHORT})"
      echo "+ CUDA_VISIBLE_DEVICES=\"${gpu_id}\" llamafactory-cli train ${BASE_YAML} ${train_args[*]}"
      CUDA_VISIBLE_DEVICES="${gpu_id}" llamafactory-cli train "${BASE_YAML}" \
        "${train_args[@]}"
      echo "==> [GPU ${gpu_id}] 完成训练: ${run_name}"
    } 2>&1 | tee "${log_file}"
  ) &
}

# ========= 推理（每任务一个进程） =========
run_infer() {
  local method="$1" dataset="$2" gen_len="$3" gpu_id="$4"
  local suffix; suffix="$(get_suffix "$1" "$3")"
  local ttl_tag; ttl_tag="$(get_ttl_tag)"

  local infer_dataset="${dataset}"
  if [[ "$dataset" == rb_* ]]; then
      infer_dataset="${dataset}_test"
  fi

  local result_dir=""
  local adapter_path=""

  if [[ "$method" == "base" ]]; then
    result_dir="${RESULTS_BASE_DIR}/base"
    mkdir -p "${result_dir}"
  else
    local base_save_root="saves/${MODEL_SHORT}/${ttl_tag}"
    adapter_path="${base_save_root}/${method}_search${suffix}/${dataset}_${gen_len}"
    if [[ ! -d "${adapter_path}" ]]; then
      echo "⚠️ [GPU ${gpu_id}] 适配器缺失，跳过 ${adapter_path}"
      return 0   # 不让函数失败导致整脚本退出
    fi
    result_dir="${RESULTS_BASE_DIR}/${ttl_tag}/${method}_${gen_len}${suffix}"
    mkdir -p "${result_dir}"
  fi

  local result_file="${result_dir}/${dataset}.jsonl"

  local log_dir="${LOG_ROOT}/${MODEL_SHORT}/${method}/infer${suffix}"
  mkdir -p "${log_dir}"
  local log_file="${log_dir}/${dataset}_${gen_len}.log"

  local args=(
    --model_name_or_path "${BASE_MODEL_PATH}" --dataset "${infer_dataset}"
    --template "${TEMPLATE}"
    --save_name "${result_file}"
    --temperature 0 --top_p 1 --top_k -1
    --seed 42 --batch_size 5000
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
    --max_new_tokens 512
  )
  [[ "${NO_DEFAULT_SYSTEM_PROMPT}" == "true" ]] && args+=(--default_system '')

  if [[ "$method" != "base" ]]; then
    args+=(--adapter_name_or_path "${adapter_path}")
  fi

  (
    {
      echo "==> [GPU ${gpu_id}] 启动推理: ${dataset}_${method}_${gen_len}${suffix} (model=${MODEL_SHORT})"
      echo "+ CUDA_VISIBLE_DEVICES=\"${gpu_id}\" python scripts/vllm_infer.py ${args[*]}"
      CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/vllm_infer.py "${args[@]}"
      echo "==> [GPU ${gpu_id}] 完成推理: ${dataset}_${method}_${gen_len}${suffix}"
    } 2>&1 | tee "${log_file}"
  ) &
}

# ========= 评估 =========
run_eval() {
  local method="$1" dataset="$2" gen_len="$3" gpu_id="$4"

  local suffix; suffix="$(get_suffix "$1" "$3")"
  local ttl_tag; ttl_tag="$(get_ttl_tag)"

  local input_file=""
  local output_file=""
  if [[ "$method" == "base" ]]; then
    input_file="${RESULTS_BASE_DIR}/base/${dataset}.jsonl"
    output_file="${RESULTS_BASE_DIR}/base/${dataset}_metrics.json"
  else
    input_file="${RESULTS_BASE_DIR}/${ttl_tag}/${method}_${gen_len}${suffix}/${dataset}.jsonl"
    output_file="${RESULTS_BASE_DIR}/${ttl_tag}/${method}_${gen_len}${suffix}/${dataset}_metrics.json"
  fi

  if [[ ! -f "${input_file}" ]]; then
    echo "⚠️ [GPU ${gpu_id}] 推理输出缺失，跳过评估 ${input_file}"
    return 0  # 不让函数失败导致整脚本退出
  fi

  local log_dir="${LOG_ROOT}/${MODEL_SHORT}/${method}/eval${suffix}"
  mkdir -p "${log_dir}"
  local log_file="${log_dir}/${dataset}_${gen_len}.log"

  (
    {
      echo "==> [GPU ${gpu_id}] 启动评估: ${dataset}_${method}_${gen_len}${suffix} (model=${MODEL_SHORT})"
      echo "+ CUDA_VISIBLE_DEVICES=\"${gpu_id}\" python scripts/eval_ttl_aligned.py --filename \"${input_file}\" --output_filename \"${output_file}\" --metrics bertscore,rouge,bleu,em"
      CUDA_VISIBLE_DEVICES="${gpu_id}" python scripts/eval_ttl_aligned.py \
        --filename "${input_file}" --output_filename "${output_file}" \
        --metrics "bertscore,rouge,bleu,em"
      echo "==> [GPU ${gpu_id}] 完成评估: ${dataset}_${method}_${gen_len}${suffix}"
    } 2>&1 | tee "${log_file}"
  ) &
}

# ========= 全局任务调度 =========
execute_stage_globally() {
  local stage_name="$1"

  local tasks_to_run_method=()
  local tasks_to_run_dataset=()
  local tasks_to_run_gen_len=()

  echo ">>> 正在为 [${stage_name^^}] 阶段筛选任务..."
  for i in "${!tasks_method[@]}"; do
    local method="${tasks_method[i]}"
    local dataset="${tasks_dataset[i]}"
    local len="${tasks_gen_len[i]}"
    local suffix="$(get_suffix "$method" "$len")"
    local ttl_tag; ttl_tag="$(get_ttl_tag)"
    local should_run=false

    case "${stage_name}" in
      "train")
        if [[ "$method" == "base" ]]; then
          echo "⏩ [TRAIN] base 方法仅推理，跳过: ${method}/${dataset}/${len}"
          should_run=false
        else
          local base_save_root="saves/${MODEL_SHORT}/${ttl_tag}"
          local adapter_path="${base_save_root}/${method}_search${suffix}/${dataset}_${len}"
          # 训练完成需检测目录内存在 .safetensors 文件
          if [[ -d "${adapter_path}" ]]; then
            if compgen -G "${adapter_path}"/*.safetensors > /dev/null; then
              echo "✅ [TRAIN] 检测到 .safetensors，视为训练已完成，跳过任务: ${method}/${dataset}/${len}"
              should_run=false
            else
              echo "⚠️ [TRAIN] 目录存在但缺少 .safetensors，视为训练中断，将重新训练: ${adapter_path}"
              should_run=true
            fi
          else
            should_run=true
          fi
        fi
        ;;
      "infer")
        local infer_file=""
        if [[ "$method" == "base" ]]; then
          infer_file="${RESULTS_BASE_DIR}/base/${dataset}.jsonl"
        else
          infer_file="${RESULTS_BASE_DIR}/${ttl_tag}/${method}_${len}${suffix}/${dataset}.jsonl"
        fi

        if [[ ! -f "${infer_file}" ]]; then
          should_run=true
        else
          echo "✅ [INFER] 结果已存在，跳过任务: ${method}/${dataset}/${len}"
        fi
        ;;
      "eval")
        local eval_file=""
        if [[ "$method" == "base" ]]; then
          eval_file="${RESULTS_BASE_DIR}/base/${dataset}_metrics.json"
        else
          eval_file="${RESULTS_BASE_DIR}/${ttl_tag}/${method}_${len}${suffix}/${dataset}_metrics.json"
        fi
        if [[ ! -f "${eval_file}" ]]; then
          should_run=true
        else
          echo "✅ [EVAL] 结果已存在，跳过任务: ${method}/${dataset}/${len}"
        fi
        ;;
      *)
        echo "❌ 未知的阶段: ${stage_name}"
        return 1
        ;;
    esac

    if [[ "$should_run" == "true" ]]; then
      tasks_to_run_method+=("$method")
      tasks_to_run_dataset+=("$dataset")
      tasks_to_run_gen_len+=("$len")
    fi
  done

  local num_tasks=${#tasks_to_run_method[@]}
  if (( num_tasks == 0 )); then
    echo "⏩ 在 [${stage_name}] 阶段无任务需要执行，跳过。"
    return
  fi
  echo ">>> [${stage_name^^}] 阶段筛选完毕，需要执行 ${num_tasks} 个任务。"

  local num_gpus=${#gpus[@]}
  local max_jobs_per_gpu=0
  case "${stage_name}" in
    "train") max_jobs_per_gpu=$MAX_TRAIN_JOBS_PER_GPU;;
    "infer") max_jobs_per_gpu=1;;
    "eval")  max_jobs_per_gpu=$MAX_EVAL_JOBS_PER_GPU;;
  esac

  local max_concurrent_jobs=$(( num_gpus * max_jobs_per_gpu ))

  echo "------------------- 启动全局并行 [${stage_name^^}] (${num_tasks}个任务, model=${MODEL_SHORT}) -------------------"
  echo "--> 策略: 每个GPU最多运行 ${max_jobs_per_gpu} 个任务 (总并发上限: ${max_concurrent_jobs})"

  declare -A gpu_load
  for gpu in "${gpus[@]}"; do
    gpu_load[$gpu]=0
  done
  declare -A running_pids
  local task_indices_to_run=()
  for i in $(seq 0 $((num_tasks - 1))); do
    task_indices_to_run+=($i)
  done

  # ====== 标准 per-dataset 路径（不再包含 Multi-LoRA）======
  while [[ ${#task_indices_to_run[@]} -gt 0 || ${#running_pids[@]} -gt 0 ]]; do
    while [[ ${#running_pids[@]} -lt $max_concurrent_jobs && ${#task_indices_to_run[@]} -gt 0 ]]; do
      local target_gpu_id=-1
      for gpu_id in "${gpus[@]}"; do
        if (( ${gpu_load[$gpu_id]:-0} < max_jobs_per_gpu )); then
          target_gpu_id=$gpu_id
          break
        fi
      done
      if [[ $target_gpu_id -eq -1 ]]; then
        break
      fi

      local i=${task_indices_to_run[0]}
      task_indices_to_run=("${task_indices_to_run[@]:1}")
      local method="${tasks_to_run_method[i]}"
      local dataset="${tasks_to_run_dataset[i]}"
      local gen_len="${tasks_to_run_gen_len[i]}"

      gpu_load[$target_gpu_id]=$(( ${gpu_load[$target_gpu_id]} + 1 ))
      echo "--> [${stage_name^^}] 启动任务 [${i+1}/${num_tasks}]: ${method}/${dataset}/${gen_len} on GPU ${target_gpu_id} (负载: ${gpu_load[$target_gpu_id]}/${max_jobs_per_gpu})"

      case "${stage_name}" in
        "train")
          local lr=""
          if [[ "$dataset" == db_* || "$dataset" == ib_* ]]; then lr="5e-5"
          elif [[ "$dataset" == rb_* ]]; then lr="1e-6"
          else lr="5e-5"
          fi
          run_train "${method}" "${dataset}" "${gen_len}" "${target_gpu_id}" "${lr}" || true
          ;;
        "infer")
          run_infer "${method}" "${dataset}" "${gen_len}" "${target_gpu_id}" || true
          ;;
        "eval")
          run_eval "${method}" "${dataset}" "${gen_len}" "${target_gpu_id}" || true
          ;;
      esac

      local pid=$!
      running_pids[$pid]=$target_gpu_id

      if [[ "${ENABLE_LAUNCH_DELAY}" == "true" && -n "${LAUNCH_DELAY_SECONDS}" && "${LAUNCH_DELAY_SECONDS}" -gt 0 && ${#task_indices_to_run[@]} -gt 0 ]]; then
        sleep "${LAUNCH_DELAY_SECONDS}"
      fi
    done

    if [[ ${#running_pids[@]} -gt 0 ]]; then
      # 避免因某个子任务失败导致整脚本退出
      wait -n || true
      for pid in "${!running_pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
          local gpu_returned=${running_pids[$pid]}
          unset running_pids[$pid]
          gpu_load[$gpu_returned]=$(( ${gpu_load[$gpu_returned]} - 1 ))
          echo "--> [${stage_name^^}] (reaped) 任务 ${pid} 完成, GPU ${gpu_returned} 负载降至: ${gpu_load[$gpu_returned]}/${max_jobs_per_gpu}"
        fi
      done
    fi
  done

  echo "--> 所有任务已启动，等待所有剩余的 [${stage_name^^}] 任务完成..."
  wait || true
  echo "✅ 所有 [${stage_name^^}] 任务已完成。"
  echo ""
}

# ========= 主逻辑：按模型 × TTL 组合运行 =========
for MODEL_KEY in "${models[@]}"; do
  set_model "${MODEL_KEY}"

  for TTL_SETTING in "${TTL_SETTING_LIST[@]}"; do
    for TTL_REF_MODE in "${TTL_REF_MODE_LIST[@]}"; do
      for TTL_REF_BATCH_SIZE in "${TTL_REF_BATCH_SIZE_LIST[@]}"; do
        for TTL_ENABLE_INFERENCE in "${TTL_ENABLE_INFERENCE_LIST[@]}"; do
          for TTL_THRESHOLD in "${TTL_THRESHOLD_LIST[@]}"; do
            for TTL_SCALER in "${TTL_SCALER_LIST[@]}"; do
              for TTL_STREAMING_BATCH_SIZE in "${TTL_STREAMING_BATCH_SIZE_LIST[@]}"; do

                echo ">>> 当前 TTL 配置: ttl_setting=${TTL_SETTING}, ref_mode=${TTL_REF_MODE}, ref_bs=${TTL_REF_BATCH_SIZE}, direct_infer=${TTL_ENABLE_INFERENCE}, thr=${TTL_THRESHOLD}, scaler=${TTL_SCALER}, stream_bs=${TTL_STREAMING_BATCH_SIZE}"
                echo ">>> TTL 路径标签: $(get_ttl_tag)"
                echo ""

                echo ">>> 准备数据集列表，交错排列以优化GPU分配..."
                rb_datasets=()
                other_datasets=()
                for d in "${datasets[@]}"; do
                  if [[ "$d" == rb_* ]]; then
                    rb_datasets+=("$d")
                  else
                    other_datasets+=("$d")
                  fi
                done

                interleaved_datasets=()
                len_rb=${#rb_datasets[@]}
                len_other=${#other_datasets[@]}
                max_len=$(( len_rb > len_other ? len_rb : len_other ))

                for (( i=0; i<max_len; i++ )); do
                  if [[ $i -lt $len_rb ]]; then
                    interleaved_datasets+=("${rb_datasets[i]}")
                  fi
                  if [[ $i -lt $len_other ]]; then
                    interleaved_datasets+=("${other_datasets[i]}")
                  fi
                done
                echo ">>> 交错后的数据集列表: ${interleaved_datasets[*]}"
                echo ""

                echo ">>> 构建全局任务列表..."
                echo ">>> TTLTENT配置: 平衡方法=${LOSS_BALANCING_METHOD}, 交替训练=${ALTERNATING_TRAINING}, KL正则化=${USE_KL_REGULARIZATION}"
                tasks_method=()
                tasks_dataset=()
                tasks_gen_len=()

                for method in "${methods[@]}"; do
                  for len in "${generation_lens[@]}"; do
                    if method_is_pure_ttl "${method}" && [[ "${len}" -ne 0 ]]; then
                      echo "INFO: 跳过纯 TTL 方法 '${method}' 因为 generation_len=${len} > 0."
                      continue
                    fi
                    for dataset in "${interleaved_datasets[@]}"; do
                      tasks_method+=("$method")
                      tasks_dataset+=("$dataset")
                      tasks_gen_len+=("$len")
                    done
                  done
                done
                num_total_tasks=${#tasks_method[@]}
                if (( num_total_tasks == 0 )); then
                    echo "⚠️ 未发现任何有效的任务组合，脚本将跳到下一组 TTL 配置。"
                    continue
                fi
                echo ">>> 总共构建了 ${num_total_tasks} 个有效任务组合。"
                echo ""

                if [[ "${DO_TRAIN}" == "true" ]]; then
                  echo "==========================================================="
                  echo "==> 阶段一：执行所有训练任务 (model=${MODEL_SHORT})"
                  echo "==========================================================="
                  execute_stage_globally "train"
                else
                  echo "⏩ 跳过训练阶段。"
                fi
                echo ""

                if [[ "${DO_INFER}" == "true" ]]; then
                  echo "==========================================================="
                  echo "==> 阶段二：执行所有推理任务 (model=${MODEL_SHORT})"
                  echo "==========================================================="
                  execute_stage_globally "infer"
                else
                  echo "⏩ 跳过推理阶段。"
                fi
                echo ""

                if [[ "${DO_EVAL}" == "true" ]]; then
                  echo "==========================================================="
                  echo "==> 阶段三：执行所有评估任务 (model=${MODEL_SHORT})"
                  echo "==========================================================="
                  execute_stage_globally "eval"
                else
                  echo "⏩ 跳过评估阶段。"
                fi
                echo ""

              done
            done
          done
        done
      done
    done
  done
done

echo "==========================================================="
echo "==> 脚本所有阶段执行完毕 ✅"
echo "==========================================================="