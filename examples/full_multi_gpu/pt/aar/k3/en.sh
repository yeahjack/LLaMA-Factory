#!/bin/bash

#SBATCH -J aarK3En
#SBATCH --mem=80G
#SBATCH --output=/home/yijiexu/LLMWatermark/WM-Open-Source-LLM/LLaMA-Factory/examples/full_multi_gpu/pt/aar/k3/en.out
#SBATCH --error=/home/yijiexu/LLMWatermark/WM-Open-Source-LLM/LLaMA-Factory/examples/full_multi_gpu/pt/aar/k3/en.err
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
    src/train.py examples/full_multi_gpu/pt/aar/k3/en.yaml
