#!/bin/bash

# DINO World Model Complete Environment Setup Script
# Updated for complete server setup (Sep 10, 2025)
# Includes: Python 3.10, robomimic/robosuite from source, video conversion fixes

set -e  # Exit on any error

echo "🚀 DINO World Model Complete Server Setup"
echo "=========================================="
echo "This script will:"
echo "✅ Install miniconda if needed"
echo "✅ Create Python 3.10 environment (wm310)"
echo "✅ Install all packages including robosuite/robomimic from source"
echo "✅ Download and convert robomimic datasets"
echo "✅ Fix video conversion issues"
echo "✅ Setup training scripts"
echo ""

# Install miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "📦 Installing miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
fi

# Ensure conda is available
export PATH="$HOME/miniconda/bin:$PATH"
eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"

# Add conda to .bashrc permanently if not already there
if ! grep -q "export PATH.*miniconda" ~/.bashrc; then
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"' >> ~/.bashrc
fi

# Set the name of the conda environment
env_name="wm310"

# Define the dataset directory and save directory (update paths as needed)
dataset_dir="/home/ubuntu/minghao/data/robomimic"
robosuite_dir="/home/ubuntu/minghao/robosuite"
robomimic_dir="/home/ubuntu/minghao/robomimic"

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

# Configure conda and accept Terms of Service
echo "🔧 Configuring conda..."

# Initialize conda for shell integration
conda init bash

conda config --set always_yes true
conda config --set changeps1 false

# Remove existing environment if it exists
if conda info --envs | grep -q "^$env_name "; then
    echo "🗑️  Removing existing environment: $env_name"
    conda remove -n $env_name --all -y
fi

# Create a new conda environment with Python 3.10 (FIXED: was Python 3.9)
echo "📦 Creating conda environment: $env_name (Python 3.10)"
conda create -n $env_name python=3.10 -y

# Activate the conda environment
echo "🔧 Activating conda environment..."
conda activate $env_name

echo "📚 Installing core packages..."

# Upgrade pip first
pip install --upgrade pip

# Install Rust for tokenizers compilation
echo "🦀 Installing Rust for tokenizers..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
export PATH="$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc

# Install PyTorch with CUDA support - auto-detect CUDA version
echo "🔍 Detecting CUDA version..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1-2)
    echo "Detected CUDA Version: $CUDA_VERSION"
    
    # Map CUDA version to PyTorch index
    if [[ "$CUDA_VERSION" == "12.4" ]] || [[ "$CUDA_VERSION" == "12."* ]]; then
        echo "Installing PyTorch for CUDA 12.x..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == "11."* ]]; then
        echo "Installing PyTorch for CUDA 11.x..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Installing PyTorch with default CUDA support..."
        pip install torch torchvision
    fi
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision
fi

# Install core ML packages (FIXED: correct transformers version)
pip install transformers==4.28.0 huggingface_hub==0.23.4
pip install scipy numpy Pillow opencv-python termcolor tqdm
pip install diffusers==0.11.1 egl_probe>=1.0.1 h5py imageio imageio-ffmpeg 
pip install matplotlib psutil tensorboard tensorboardX

# Install distributed training and experiment management
pip install accelerate hydra-core wandb einops

# Install video processing package (CRITICAL: was missing)
pip install decord

# Install fast download utility
echo "⚡ Installing fast download utilities..."
pip install hf_transfer

# Install enhanced GPU monitoring for auto-selection
echo "🎯 Installing enhanced GPU selection tools..."
pip install nvitop pynvml || echo "⚠️ nvitop/pynvml installation failed, using fallback GPU selection"

# Install planning dependencies
echo "🎯 Installing planning dependencies..."
pip install d4rl  # Required for planning evaluation

# Install mujoco_py for robomimic compatibility
echo "🤖 Installing mujoco_py..."
pip install mujoco_py==2.1.2.14 Cython

# Install robosuite from source (CRITICAL: -e for editable install)
echo "🤖 Installing robosuite from source..."
if [ -d "$robosuite_dir" ]; then
    cd $robosuite_dir
    pip install -e .
    cd -
    echo "✅ Robosuite installed from source"
else
    echo "⚠️  Robosuite directory not found: $robosuite_dir"
    echo "Please clone robosuite repository or update the path"
fi

# Install robomimic from source (CRITICAL: -e for editable install)
echo "🤖 Installing robomimic from source..."
if [ -d "$robomimic_dir" ]; then
    cd $robomimic_dir
    pip install -e .
    cd -
    echo "✅ Robomimic installed from source"
else
    echo "⚠️  Robomimic directory not found: $robomimic_dir"
    echo "Please clone robomimic repository or update the path"
fi

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

# Dataset conversion for DINO WM format (FIXED: correct script and paths)
echo ""
echo "🔄 DINO WM Dataset Conversion"
echo "============================="
echo "Converting robomimic dataset to DINO WM format..."

# Check if conversion script exists
if [ -f "convert_robomimic_to_dino_wm_final.py" ]; then
    for task in "${tasks[@]}"; do
        for dataset_type in "${dataset_types[@]}"; do
            source_path="$dataset_dir/$task/$dataset_type"
            output_path="$dataset_dir/$task/${dataset_type}_converted_final"
            
            if [ -f "$source_path/demo_v15.hdf5" ] && [ -f "$source_path/low_dim_v15.hdf5" ] && [ -f "$source_path/image_384_v15.hdf5" ]; then
                echo "🔄 Converting $source_path to DINO WM format..."
                echo "Source: $source_path"
                echo "Output: $output_path"
                
                # Use the robomimic_convert environment for conversion
                python convert_robomimic_to_dino_wm_final.py \
                    --source_dir "$source_path" \
                    --save_data_dir "$output_path"
                
                echo "✅ Converted to: $output_path"
                
                # Verify conversion output
                if [ -f "$output_path/states.pth" ] && [ -d "$output_path/obses" ]; then
                    video_count=$(ls "$output_path/obses/"*.mp4 2>/dev/null | wc -l)
                    echo "✅ Conversion verified: $video_count video files created"
                else
                    echo "⚠️  Conversion may have failed - missing expected output files"
                fi
            else
                echo "⚠️  Required input files not found in: $source_path"
                echo "Expected: demo_v15.hdf5, low_dim_v15.hdf5, image_384_v15.hdf5"
            fi
        done
    done
else
    echo "⚠️  DINO WM conversion script not found: convert_robomimic_to_dino_wm_final.py"
    echo "Please run the conversion manually after setup"
fi

# Update configuration files (FIXED: correct dataset paths)
echo ""
echo "🔧 Updating configuration files..."

# Update dataset path in configuration
config_file="conf/env/robomimic_can.yaml"
if [ -f "$config_file" ]; then
    # Update the data path to point to our converted dataset
    updated_path="$dataset_dir/can/ph_converted_final"
    sed -i "s|data_path:.*|data_path: $updated_path|g" "$config_file"
    echo "✅ Updated dataset path in $config_file to: $updated_path"
else
    echo "⚠️  Configuration file not found: $config_file"
fi

# Make training script executable
if [ -f "train.sh" ]; then
    chmod +x train.sh
    echo "✅ Made train.sh executable"
fi

# Final verification and setup summary
echo ""
echo "🎉 DINO World Model Setup Complete!"
echo "===================================="
echo "🎯 Environment: $env_name (Python 3.10)"
echo "📊 GPU Support: $(python -c "import torch; print('✅ CUDA' if torch.cuda.is_available() else '❌ CPU Only')" 2>/dev/null || echo '❓ Please activate conda environment first')"
echo "🗂️  Dataset Directory: $dataset_dir"
echo "🤖 Robosuite: $([ -d "$robosuite_dir" ] && echo '✅ Installed from source' || echo '⚠️ Not found')"
echo "🤖 Robomimic: $([ -d "$robomimic_dir" ] && echo '✅ Installed from source' || echo '⚠️ Not found')"
echo ""
echo "🚀 Quick Start Commands:"
echo "# Activate environment:"
echo "conda activate $env_name"
echo ""
echo "# Debug training (1 epoch):"
echo "DEBUG=true ./train.sh"
echo ""
echo "# Full training (single GPU):"
echo "./train.sh"
echo ""
echo "# Multi-GPU training (2 GPUs, 50 epochs):"
echo "NUM_GPUS=2 EPOCHS=50 ./train.sh"
echo ""
echo "# Planning evaluation:"
echo "./plan.sh"
echo ""
echo "📖 See CLAUDE.md for detailed instructions and troubleshooting"
echo "🐛 If you encounter issues, check that robosuite/robomimic directories exist"
