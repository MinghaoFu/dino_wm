#!/bin/bash

# DINO World Model Planning Script  
# Uses same GPU selection strategy as train.sh

set -e  # Exit on any error

# Environment setup
echo "🔧 Setting up planning environment..."
cd /home/minghao/workspace/dino_wm
source ~/.bashrc  # Load environment variables including LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/minghao/.mujoco/mujoco210/bin:/usr/lib/nvidia
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wm310

# Planning configuration
CONFIG_NAME="plan_projected_latent"
GOAL_H=${GOAL_H:-5}  # Default horizon 5, can be overridden
N_EVALS=${N_EVALS:-10}  # Default number of evaluation seeds per horizon

echo "🚀 Starting DINO World Model planning..."
echo "Config: $CONFIG_NAME"
echo "Goal Horizon: $GOAL_H"

# Enhanced GPU selection using nvitop-based script
echo "🎯 Auto-selecting best GPU..."
if command -v ./select_best_gpus.py &> /dev/null; then
    BEST_GPU=$(./select_best_gpus.py single --quiet)
    echo "Selected GPU: $BEST_GPU"
    export CUDA_VISIBLE_DEVICES=$BEST_GPU
else
    echo "⚠️  Enhanced GPU selection script not found, using basic selection"
    BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -1 | cut -d',' -f1)
    export CUDA_VISIBLE_DEVICES=$BEST_GPU
    echo "Selected GPU: $BEST_GPU"
fi

# Run planning
echo "🎯 Starting planning evaluation..."
python plan_projected_latent.py --config-name=$CONFIG_NAME goal_H=$GOAL_H n_evals=$N_EVALS

# Check planning status
if [ $? -eq 0 ]; then
    echo "✅ Planning completed successfully!"
else
    echo "❌ Planning failed!"
    exit 1
fi

echo "🎉 Planning script completed!"