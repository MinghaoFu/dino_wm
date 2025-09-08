#!/bin/bash

# Set the name of the conda environment
env_name="wm310"

# Define the dataset directory and save directory (update these paths as needed)
dataset_dir="/home/fuminghao/data/robomimic"
robosuite_dir="/home/fuminghao/workspace/robosuite"
robomimic_dir="/home/fuminghao/workspace/robomimic"

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

# Create a new conda environment with Python 3.10
conda create -n $env_name python=3.10 -y

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $env_name

# Install necessary packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.28.0 huggingface_hub==0.23.4
pip install scipy numpy Pillow opencv-python termcolor tqdm
pip install diffusers==0.11.1 egl_probe>=1.0.1 h5py imageio imageio-ffmpeg matplotlib psutil tensorboard tensorboardX
pip install accelerate hydra-core wandb einops

# Install robosuite
cd $robosuite_dir
pip install .

cd $robomimic_dir
pip install .

# Return to the original directory
cd -

# Install hf_transfer for fast downloads
pip install hf_transfer

# Download specified tasks and dataset types
for task in "${tasks[@]}"; do
    for dataset_type in "${dataset_types[@]}"; do
        echo "Downloading task: $task, dataset type: $dataset_type"
        python $robomimic_dir/robomimic/scripts/download_datasets.py \
            --tasks $task --dataset_types $dataset_type --hdf5_types all --download_dir $dataset_dir

        # Verify dataset path and ensure it exists
        if [ ! -f "$dataset_dir/$task/$dataset_type/demo_v15.hdf5" ]; then
            echo "Dataset file for $task, $dataset_type not found. Please check the download process."
            exit 1
        fi

        # Convert states to images for the specified task and dataset type
        python $robomimic_dir/robomimic/scripts/dataset_states_to_obs.py \
            --dataset $dataset_dir/$task/$dataset_type/demo_v15.hdf5 \
            --output_name $dataset_dir/$task/$dataset_type/image_384_v15.hdf5 \
            --done_mode 2 \
            --camera_names agentview robot0_eye_in_hand \
            --camera_height 384 \
            --camera_width 384
    done
done

# Print success message
echo "Setup and preparation complete."
