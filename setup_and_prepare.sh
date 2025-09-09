#!/bin/bash

# DINO World Model Complete Environment Setup Script
# Updated for 80D Projected Latent Architecture (Sep 9, 2025)

set -e  # Exit on any error

echo "🚀 DINO World Model Environment Setup"
echo "========================================="

# Set the name of the conda environment
env_name="wm310"

# Define the dataset directory and save directory (update these paths as needed)
dataset_dir="/home/minghao/data/robomimic"
robosuite_dir="/home/minghao/workspace/robosuite"
robomimic_dir="/home/minghao/workspace/robomimic"

# Define the dataset types to download (default to PH)
dataset_types=("ph")

# Check for user-specified dataset types
if [ "$1" ]; then
    IFS=',' read -r -a dataset_types <<< "$1"
fi

# Define the tasks to download (default to 'can')
tasks=("can")

# Check for user-specified tasks
if [ "$2" ]; then
    IFS=',' read -r -a tasks <<< "$2"
fi

echo "📦 Creating conda environment: $env_name"
# Create a new conda environment with Python 3.10
conda create -n $env_name python=3.10 -y

# Activate the conda environment
echo "🔧 Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $env_name

echo "📚 Installing core packages..."
# Install PyTorch with CUDA support (updated to use CUDA index)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install core ML packages
pip install transformers==4.28.0 huggingface_hub==0.23.4
pip install scipy numpy Pillow opencv-python termcolor tqdm
pip install diffusers==0.11.1 egl_probe>=1.0.1 h5py imageio imageio-ffmpeg 
pip install matplotlib psutil tensorboard tensorboardX

# Install distributed training and experiment management
pip install accelerate hydra-core wandb einops

# Install enhanced GPU monitoring for auto-selection (NEW)
echo "🎯 Installing enhanced GPU selection tools..."
pip install nvitop pynvml

# Install planning dependencies
echo "🎯 Installing planning dependencies..."
pip install d4rl  # Required for planning evaluation

# Install robosuite
echo "🤖 Installing robosuite..."
if [ -d "$robosuite_dir" ]; then
    cd $robosuite_dir
    pip install .
    cd -
else
    echo "⚠️  Robosuite directory not found: $robosuite_dir"
    echo "Please update the path or install robosuite manually"
fi

# Install robomimic
echo "🤖 Installing robomimic..."
if [ -d "$robomimic_dir" ]; then
    cd $robomimic_dir
    pip install .
    cd -
else
    echo "⚠️  Robomimic directory not found: $robomimic_dir"
    echo "Please update the path or install robomimic manually"
fi

# Install fast download utility
echo "⚡ Installing fast download utilities..."
pip install hf_transfer

# Setup environment variables
echo "🔧 Setting up environment variables..."
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/minghao/.mujoco/mujoco210/bin:/usr/lib/nvidia' >> ~/.bashrc
echo 'export WANDB_BASE_URL=https://api.bandw.top' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface' >> ~/.bashrc
echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> ~/.bashrc

# Load environment variables for current session
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/minghao/.mujoco/mujoco210/bin:/usr/lib/nvidia
export WANDB_BASE_URL=https://api.bandw.top
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1

# Verify PyTorch CUDA installation
echo "🔍 Verifying PyTorch CUDA installation..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Verify enhanced GPU selection
echo "🎯 Testing enhanced GPU selection..."
if command -v ./select_best_gpus.py &> /dev/null; then
    echo "Enhanced GPU selection script found ✅"
    ./select_best_gpus.py single --quiet > /dev/null && echo "GPU selection working ✅"
else
    echo "⚠️  Enhanced GPU selection script not found - please ensure select_best_gpus.py is executable"
fi

# Dataset download and conversion section
echo ""
echo "📥 Dataset Download & Conversion"
echo "================================="

# Create dataset directory if it doesn't exist
mkdir -p $dataset_dir

# Download specified tasks and dataset types
for task in "${tasks[@]}"; do
    for dataset_type in "${dataset_types[@]}"; do
        echo "📥 Downloading task: $task, dataset type: $dataset_type"
        
        if [ -d "$robomimic_dir" ]; then
            python $robomimic_dir/robomimic/scripts/download_datasets.py \
                --tasks $task --dataset_types $dataset_type --hdf5_types all --download_dir $dataset_dir

            # Verify dataset path and ensure it exists
            if [ ! -f "$dataset_dir/$task/$dataset_type/demo_v15.hdf5" ]; then
                echo "❌ Dataset file for $task, $dataset_type not found. Please check the download process."
                exit 1
            fi

            echo "🖼️  Converting states to images for $task ($dataset_type)..."
            # Convert states to images for the specified task and dataset type
            python $robomimic_dir/robomimic/scripts/dataset_states_to_obs.py \
                --dataset $dataset_dir/$task/$dataset_type/demo_v15.hdf5 \
                --output_name $dataset_dir/$task/$dataset_type/image_384_v15.hdf5 \
                --done_mode 2 \
                --camera_names agentview robot0_eye_in_hand \
                --camera_height 384 \
                --camera_width 384
            
            echo "✅ Dataset $task ($dataset_type) downloaded and converted successfully"
        else
            echo "⚠️  Skipping dataset download - robomimic not found"
            echo "Please install robomimic first or update the path: $robomimic_dir"
        fi
    done
done

# Dataset conversion for DINO WM format
echo ""
echo "🔄 DINO WM Dataset Conversion"
echo "============================="
echo "Converting robomimic dataset to DINO WM format..."

# Check if conversion script exists
if [ -f "convert_robomimic_to_dino_wm.py" ]; then
    for task in "${tasks[@]}"; do
        for dataset_type in "${dataset_types[@]}"; do
            input_path="$dataset_dir/$task/$dataset_type/image_384_v15.hdf5"
            output_path="$dataset_dir/$task/${dataset_type}_converted_final"
            
            if [ -f "$input_path" ]; then
                echo "🔄 Converting $input_path to DINO WM format..."
                python convert_robomimic_to_dino_wm.py --input "$input_path" --output "$output_path"
                echo "✅ Converted to: $output_path"
            else
                echo "⚠️  Input file not found: $input_path"
            fi
        done
    done
else
    echo "⚠️  DINO WM conversion script not found: convert_robomimic_to_dino_wm.py"
    echo "Please run the conversion manually after setup"
fi

# Final verification and setup summary
echo ""
echo "✅ DINO World Model Setup Complete!"
echo "=================================="
echo "🎯 Environment: $env_name"
echo "📊 GPU Support: $(python -c "import torch; print('✅ CUDA' if torch.cuda.is_available() else '❌ CPU Only')")"
echo "🗂️  Dataset Directory: $dataset_dir"
echo ""
echo "🚀 Next Steps:"
echo "1. Update dataset path in conf/env/robomimic_can.yaml if needed"
echo "2. Run debug training: python train_robomimic_compress.py --config-name=train_robomimic_compress training.epochs=1 debug=true"
echo "3. Run full training: python train_robomimic_compress.py --config-name=train_robomimic_compress"
echo "4. Run planning: ./plan.sh"
echo ""
echo "📖 See CLAUDE.md for detailed usage instructions"
