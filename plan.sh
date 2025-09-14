#!/bin/bash

# DINO World Model Planning Script
# Runs planning evaluation with latest checkpoint

set -e  # Exit on any error

echo "🎯 DINO World Model Planning Evaluation"
echo "======================================"

# Environment setup
echo "🔧 Setting up environment..."
cd /home/ubuntu/minghao/wm
export PATH="/home/ubuntu/miniconda/bin:$PATH"
eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"
conda activate wm310

# MuJoCo and other environment variables
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export WANDB_BASE_URL=https://api.bandw.top
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1

# Configuration
CKPT_PATH=${CKPT_PATH:-"/home/ubuntu/minghao/wm/outputs/2025-09-13/03-37-25"}
DATA_PATH=${DATA_PATH:-"/home/ubuntu/minghao/data/robomimic/can/ph_convert"}
GPU_ID=${GPU_ID:-2}  # Default to GPU 2

echo "🎯 Planning Configuration:"
echo "Checkpoint: $CKPT_PATH"
echo "Data path: $DATA_PATH" 
echo "GPU: $GPU_ID"

# Verify checkpoint exists
if [ ! -d "$CKPT_PATH" ]; then
    echo "❌ Checkpoint directory not found: $CKPT_PATH"
    exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
    echo "❌ Data directory not found: $DATA_PATH"
    exit 1
fi

echo "🚀 Starting planning evaluation..."

# Run planning
CUDA_VISIBLE_DEVICES=$GPU_ID python plan_projected_latent.py \
    --config-name=plan_projected_latent \
    ckpt_base_path=$CKPT_PATH

echo "✅ Planning evaluation completed!"