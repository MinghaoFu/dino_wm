#!/bin/bash

# Multi-GPU training script for alignment models
source /home/minghao/miniconda3/bin/activate wm310

echo "Starting alignment training on GPUs 0,1..."
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --master_port=12355 \
    train_robomimic_align.py \
    --config-name=train_robomimic_align \
    > training_align_multigpu.log 2>&1 &

echo "Alignment training started, PID: $!"

# Wait a few seconds before starting the next training
sleep 10

echo "Starting alignment+reconstruction training on GPUs 5,6..."
CUDA_VISIBLE_DEVICES=5,6 torchrun \
    --nproc_per_node=2 \
    --master_port=12356 \
    train_robomimic_align.py \
    --config-name=train_robomimic_align_with_recon \
    > training_align_recon_multigpu.log 2>&1 &

echo "Alignment+reconstruction training started, PID: $!"

echo "Both multi-GPU trainings are now running in background"
echo "Check logs with:"
echo "  tail -f training_align_multigpu.log"
echo "  tail -f training_align_recon_multigpu.log"