#!/bin/bash

#SBATCH -J kgw0.25_2
#SBATCH --mem=80G
#SBATCH --output=/home/yijiexu/LLMWatermark/WM-Open-Source-LLM/LLaMA-Factory/examples/full_multi_gpu/pt/kgw_gamma0.25_delta2_mu_bf16_lr5e-5.out
#SBATCH --error=/home/yijiexu/LLMWatermark/WM-Open-Source-LLM/LLaMA-Factory/examples/full_multi_gpu/pt/kgw_gamma0.25_delta2_mu_bf16_lr5e-5.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

source ~/micromamba/etc/profile.d/micromamba.sh
micromamba activate llmwatermark

export OMP_NUM_THREADS=16
export WANDB_DISABLED=True

NPROC_PER_NODE=3

cd /home/yijiexu/LLMWatermark/WM-Open-Source-LLM/LLaMA-Factory

python -m torch.distributed.run \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes 1 \
    --standalone \
    src/train.py examples/full_multi_gpu/pt/kgw_gamma0.25_delta2_mu_0.00005.yaml

# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run \
#     --nproc_per_node $NPROC_PER_NODE \
#     --nnodes 1 \
#     --standalone \
#     src/train.py examples/full_multi_gpu/pt/aar_k3_en.yaml

# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run \
#     --nproc_per_node $NPROC_PER_NODE \
#     --nnodes 1 \
#     --standalone \
#     src/train.py examples/full_multi_gpu/pt/aar_k2_mu.yaml

# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run \
#     --nproc_per_node $NPROC_PER_NODE \
#     --nnodes 1 \
#     --standalone \
#     src/train.py examples/full_multi_gpu/pt/aar_k3_mu.yaml

# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run \
#     --nproc_per_node $NPROC_PER_NODE \
#     --nnodes 1 \
#     --standalone \
#     src/train.py examples/full_multi_gpu/pt/aar_k4_mu.yaml

# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run \
#     --nproc_per_node $NPROC_PER_NODE \
#     --nnodes 1 \
#     --standalone \
#     src/train.py examples/full_multi_gpu/pt/kgw_gamma0.25_delta2_mu.yaml