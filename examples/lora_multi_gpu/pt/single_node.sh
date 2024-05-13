#!/bin/bash

source ~/micromamba/etc/profile.d/micromamba.sh
micromamba activate llmwatermark

export OMP_NUM_THREADS=16
export WANDB_DISABLED=True

NPROC_PER_NODE=3

cd /home/yijiexu/LLMWatermark/WM-Open-Source-LLM/LLaMA-Factory

# CUDA_VISIBLE_DEVICES=3,4,5 python -m torch.distributed.run \
#     --nproc_per_node $NPROC_PER_NODE \
#     --nnodes 1 \
#     --standalone \
#     src/train.py examples/lora_multi_gpu/pt/aar_k2_en.yaml

# CUDA_VISIBLE_DEVICES=3,4,5 python -m torch.distributed.run \
#     --nproc_per_node $NPROC_PER_NODE \
#     --nnodes 1 \
#     --standalone \
#     src/train.py examples/lora_multi_gpu/pt/aar_k3_en.yaml

CUDA_VISIBLE_DEVICES=3,4,5 python -m torch.distributed.run \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes 1 \
    --standalone \
    src/train.py examples/lora_multi_gpu/pt/aar_k2_mu.yaml

CUDA_VISIBLE_DEVICES=3,4,5 python -m torch.distributed.run \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes 1 \
    --standalone \
    src/train.py examples/lora_multi_gpu/pt/aar_k3_mu.yaml

CUDA_VISIBLE_DEVICES=3,4,5 python -m torch.distributed.run \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes 1 \
    --standalone \
    src/train.py examples/lora_multi_gpu/pt/aar_k4_mu.yaml

CUDA_VISIBLE_DEVICES=3,4,5 python -m torch.distributed.run \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes 1 \
    --standalone \
    src/train.py examples/lora_multi_gpu/pt/kgw_gamma0.25_delta2_mu.yaml