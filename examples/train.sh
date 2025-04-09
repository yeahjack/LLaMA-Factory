export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export OMP_NUM_THREADS=16

cd ../

DS_CONFIG_PATH=examples/deepspeed/ds_z3_config.json
# DISTRIBUTED_ARGS="
#     --nproc_per_node $NPROC_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT
#   "

DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE"

# 模型列表
MODELS=("Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct")

# 数据集组合列表
DATASETS=("simple" "simple complex" "complex")

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do

        # 设置 batch 和 grad acc 参数
        if [[ "$MODEL" == *"7B"* ]]; then
            BATCH_SIZE=4
            GRAD_ACC=1
        else
            BATCH_SIZE=1
            GRAD_ACC=1
        fi

        # 替换 / 为 - 以便用于路径
        SAFE_MODEL_NAME=$(echo $MODEL | sed 's|/|-|g')

        # 替换空格为下划线，用于输出目录
        SAFE_DATASET_NAME=$(echo $DATASET | sed 's| |_|g')

        OUTPUT_DIR="saves/${SAFE_MODEL_NAME}-${SAFE_DATASET_NAME}"

        echo "Running training with MODEL=$MODEL, DATASET=$DATASET"
        echo "Output will be saved to $OUTPUT_DIR"
        echo "Batch size: $BATCH_SIZE, Grad Accumulation: $GRAD_ACC"
        echo "--------------------------------------------------"

        torchrun $DISTRIBUTED_ARGS src/train.py \
            --deepspeed $DS_CONFIG_PATH \
            --stage sft \
            --do_train \
            --use_fast_tokenizer \
            --flash_attn fa2 \
            --model_name_or_path "$MODEL" \
            --dataset $DATASET \
            --template qwen \
            --finetuning_type full \
            --output_dir "$OUTPUT_DIR" \
            --overwrite_cache \
            --overwrite_output_dir \
            --warmup_steps 100 \
            --weight_decay 0.1 \
            --per_device_train_batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $GRAD_ACC \
            --ddp_timeout 9000 \
            --learning_rate 5e-6 \
            --lr_scheduler_type cosine \
            --logging_steps 1 \
            --cutoff_len 4096 \
            --save_steps 1000 \
            --plot_loss \
            --num_train_epochs 3 \
            --bf16

    done
done

cd /hpc2hdd/home/zrao538/proj

python keepbusy.py