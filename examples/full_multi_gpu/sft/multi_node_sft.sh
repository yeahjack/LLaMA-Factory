#!/bin/bash

source ~/micromamba/etc/profile.d/micromamba.sh
micromamba activate llmwatermark

export OMP_NUM_THREADS=16
export WANDB_DISABLED=True

NPROC_PER_NODE=3
NNODES=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29600

cd /home/yijiexu/LLMWatermark/WM-Open-Source-LLM/LLaMA-Factory

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    src/train.py examples/full_multi_gpu/llama2_full_sft_identity.yaml

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    src/train.py examples/full_multi_gpu/llama2_full_sft_identity.yaml
