#!/bin/bash

# Simple train.py runner for PushT
set -e

# Environment setup
export PATH="/home/ubuntu/miniconda/bin:$PATH"
eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"
conda activate wm310

# Set environment variables
export WANDB_BASE_URL=https://api.bandw.top
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1

# Create logs directory
mkdir -p logs

echo "🚀 Starting PushT training..."

# Run train.py with PushT settings
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file=default_config.yaml \
    --num_processes=4 \
    train.py --config-name=train env=pusht \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S)_pusht.log

echo "✅ Training completed!"