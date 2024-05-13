#!/bin/bash

#SBATCH -J sft
#SBATCH --mem=80G
#SBATCH --output=/home/yijiexu/LLMWatermark/WM-Open-Source-LLM/LLaMA-Factory/examples/full_multi_gpu/sft/llama2_full_sft_dolly-15k.output
#SBATCH --error=/home/yijiexu/LLMWatermark/WM-Open-Source-LLM/LLaMA-Factory/examples/full_multi_gpu/sft/llama2_full_sft_dolly-15k.error
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

source ~/micromamba/etc/profile.d/micromamba.sh
micromamba activate llmwatermark

cd /home/yijiexu/LLMWatermark/WM-Open-Source-LLM/LLaMA-Factory

export OMP_NUM_THREADS=16
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_DISABLED=True

NPROC_PER_NODE=3

python -m torch.distributed.run \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes 1 \
    --standalone \
    src/train.py examples/full_multi_gpu/sft/llama2_full_sft_dolly-15k.yaml