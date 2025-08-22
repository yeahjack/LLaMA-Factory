#!/bin/bash

cd ..

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
#   "agriculture_5k"
#   "geosignal_5k"
#   "gen_med_gpt_5k"
#   "wealth_5k"
#   "alpaca_gpt4_5k"
#   "instruction_wild_5k"
#   "dolly_5k"
  "gsm8k_5k"
  "logiqa_5k"
  "meta_math_5k"
)
gpus=(0 1 2 4)

for i in ${!datasets[@]}; do
  dataset=${datasets[$i]}
  gpu=${gpus[$i]}

  CUDA_VISIBLE_DEVICES=$gpu python scripts/vllm_infer.py \
    --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
    --dataset "$dataset" \
    --template "qwen" \
    --save_name "results_qwen25_7b/base_sys_prompt/${dataset}.jsonl" \
    --temperature 0 \
    --top_p 1 \
    --top_k -1 \
    --seed 42 \
    --batch_size 250 \
    --gpu_memory_utilization 0.92 \
    --max_new_tokens 512 \
    > "logs_infer_${dataset}.log" 2>&1 &
done

wait

for i in ${!datasets[@]}; do
  dataset=${datasets[$i]}
  gpu=${gpus[$i]}

  input_file="results_qwen25_7b/base_sys_prompt/${dataset}.jsonl"
  output_file="results_qwen25_7b/base_sys_prompt/${dataset}_metrics.json"

  CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_ttl_aligned.py \
    --filename "$input_file" \
    --output_filename "$output_file" \
    --metrics "bertscore,rouge,bleu,em" \
    > "logs_eval_${dataset}.log" 2>&1 &
done
wait

echo "✅ 四个数据集推理全部完成"