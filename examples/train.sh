export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export OMP_NUM_THREADS=16

cd ../

DS_CONFIG_PATH=examples/deepspeed/ds_z3_config.json
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
OUTPUT_PATH=saves/Qwen2.5-7B-Instruct-simple
# DISTRIBUTED_ARGS="
#     --nproc_per_node $NPROC_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT
#   "

DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE"

torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn fa2\
    --model_name_or_path $MODEL_PATH \
    --dataset simple \
    --template qwen \
    --finetuning_type full \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 3 \
    --bf16

cd /hpc2hdd/home/zrao538/proj

python keepbusy.py