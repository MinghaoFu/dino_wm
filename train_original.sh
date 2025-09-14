#!/bin/bash

# Original DINO-WM Training Script for Robomimic Dataset
# This script trains the original DINO-WM architecture (without projected latent)

set -e

echo "🚀 Original DINO-WM Training for Robomimic Dataset"
echo "=================================================="

# Environment setup
export PATH="/home/ubuntu/miniconda/bin:$PATH"
eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"
conda activate wm310

# Configuration
CONFIG_NAME="train_robomimic_original"
EPOCHS=${EPOCHS:-50}
NUM_GPUS=${NUM_GPUS:-1}
DEBUG=${DEBUG:-false}

echo "🔧 Configuration:"
echo "  Config: $CONFIG_NAME"
echo "  Epochs: $EPOCHS"
echo "  GPUs: $NUM_GPUS"
echo "  Debug: $DEBUG"

# Auto-select GPUs
if command -v python &> /dev/null; then
    echo "🎯 Auto-selecting best GPUs..."

    if [ -f "select_best_gpus.py" ]; then
        if [ "$NUM_GPUS" -eq 1 ]; then
            SELECTED_GPU=$(python select_best_gpus.py single)
            export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
            echo "Selected GPU: $SELECTED_GPU"
        else
            SELECTED_GPUS=$(python select_best_gpus.py --num_gpus "$NUM_GPUS")
            export CUDA_VISIBLE_DEVICES="$SELECTED_GPUS"
            echo "Selected GPUs: $SELECTED_GPUS"
        fi
    else
        echo "⚠️  GPU auto-selection script not found, using default GPU assignment"
    fi
fi

# Debug mode adjustments
if [ "$DEBUG" = "true" ]; then
    echo "🔍 Running debug training (1 epoch)..."
    EPOCHS=1
    EXTRA_ARGS="debug=true"
else
    echo "🚀 Running full training ($EPOCHS epochs)..."
    EXTRA_ARGS=""
fi

# Training command based on GPU count
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "📱 Single GPU training..."
    python train_robomimic_original.py \
        --config-name=$CONFIG_NAME \
        training.epochs=$EPOCHS \
        $EXTRA_ARGS
else
    echo "🖥️  Multi-GPU training ($NUM_GPUS GPUs)..."
    accelerate launch --num_processes=$NUM_GPUS \
        train_robomimic_original.py \
        --config-name=$CONFIG_NAME \
        training.epochs=$EPOCHS \
        training.batch_size=16 \
        $EXTRA_ARGS
fi

echo "✅ Training completed!"