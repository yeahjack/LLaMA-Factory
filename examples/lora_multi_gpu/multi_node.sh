#!/bin/bash
# also launch it on slave machine using slave_config.yaml

source ~/micromamba/etc/profile.d/micromamba.sh
micromamba activate llmwatermark

export OMP_NUM_THREADS=16
export WANDB_DISABLED=True

cd /home/yijiexu/LLMWatermark/WM-Open-Source-LLM/LLaMA-Factory

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
    --config_file examples/accelerate/master_config.yaml \
    src/train.py examples/lora_multi_gpu/llama2_lora_pt_ds.yaml
