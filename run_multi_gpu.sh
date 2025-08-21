#!/bin/bash

# Set CUDA devices to use GPU 0 and 1
export CUDA_VISIBLE_DEVICES=0,1

# Initialize conda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wm310

# Run the training script with torchrun for multi-GPU in background
cd /home/minghao/workspace/dino_wm
nohup torchrun --nproc_per_node=2 --master_port=12356 our_train.py > training.log 2>&1 &

echo "Training started in background. Check training.log for output."
echo "Process ID: $!" 