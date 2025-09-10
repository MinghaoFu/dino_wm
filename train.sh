#!/bin/bash

# DINO World Model Training Script
# Supports both single and multi-GPU training with auto GPU selection

set -e  # Exit on any error

# Environment setup
echo "🔧 Setting up environment..."
cd /home/ubuntu/minghao/dino_wm
export PATH="/home/ubuntu/miniconda/bin:$PATH"
eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"
conda activate wm310

# Verify environment
echo "🔍 Verifying environment..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Set environment variables
export WANDB_BASE_URL=https://api.bandw.top
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1

# Training configuration
CONFIG_NAME="train_robomimic_compress"
DEBUG_MODE=${DEBUG:-false}
EPOCHS=${EPOCHS:-100}
NUM_GPUS=${NUM_GPUS:-1}  # Default to 1 GPU, can be overridden
GPU_IDS=${GPU_IDS:-"auto"}  # Can specify specific GPUs like "0,1" or "auto"

echo "🚀 Starting DINO World Model training..."
echo "Config: $CONFIG_NAME"
echo "Debug mode: $DEBUG_MODE"
echo "Epochs: $EPOCHS"
echo "Number of GPUs: $NUM_GPUS"

# GPU selection - use simple approach that works with PyTorch
if [ "$GPU_IDS" = "auto" ]; then
    echo "🎯 Auto-selecting best GPUs..."
    if [ "$NUM_GPUS" -eq 1 ]; then
        # Simple selection: find GPU with lowest memory usage  
        BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -1 | cut -d',' -f1)
        export CUDA_VISIBLE_DEVICES=$BEST_GPU
        echo "Selected GPU: $BEST_GPU"
    else
        # For multi-GPU, use first N GPUs with lowest memory
        BEST_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -$NUM_GPUS | cut -d',' -f1 | tr '\n' ',' | sed 's/,$//')
        export CUDA_VISIBLE_DEVICES=$BEST_GPUS
        echo "Selected GPUs: $BEST_GPUS"
    fi
else
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "Using specified GPUs: $GPU_IDS"
fi

# Debug run first (optional)
if [ "$DEBUG_MODE" = "true" ]; then
    echo "🔍 Running debug training (1 epoch)..."
    if [ "$NUM_GPUS" -eq 1 ]; then
        python train_robomimic_compress.py \
            --config-name=$CONFIG_NAME \
            training.epochs=1 \
            debug=true
    else
        accelerate launch --num_processes=$NUM_GPUS \
            train_robomimic_compress.py \
            --config-name=$CONFIG_NAME \
            training.epochs=1 \
            debug=true
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ Debug run successful, proceeding with full training..."
    else
        echo "❌ Debug run failed, exiting..."
        exit 1
    fi
fi

# Full training
echo "🎯 Starting full training..."
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running single GPU training..."
    python train_robomimic_compress.py \
        --config-name=$CONFIG_NAME \
        training.epochs=$EPOCHS
else
    echo "Running multi-GPU training with $NUM_GPUS GPUs..."
    # Multi-GPU with accelerate
    accelerate launch \
        --num_processes=$NUM_GPUS \
        --mixed_precision=no \
        train_robomimic_compress.py \
        --config-name=$CONFIG_NAME \
        training.epochs=$EPOCHS \
        training.batch_size=16  # Reduce batch size per GPU for multi-GPU
fi

# Check training status
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📊 Check WandB: https://wandb.ai/causality_/dino_wm_align_recon"
    echo "📁 Output directory: outputs/$(date +%Y-%m-%d)"
else
    echo "❌ Training failed!"
    exit 1
fi

echo "🎉 Training script completed!"