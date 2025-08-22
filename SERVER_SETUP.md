# New Server Setup Instructions

## Environment Setup

### 1. Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 2. Create Environment
```bash
# From environment.yml (recommended)
conda env create -f environment.yml
conda activate wm310

# OR from requirements.txt
conda create -n wm310 python=3.10
conda activate wm310
pip install -r requirements.txt
```

### 3. Additional Dependencies
```bash
# MuJoCo (if needed)
conda install -c conda-forge mujoco

# Weights & Biases
wandb login  # Enter your API key
```

## Data & Checkpoints Setup

### 1. Create Directory Structure
```bash
mkdir -p /mnt/data1/minghao/robomimic/checkpoints
mkdir -p /mnt/data1/minghao/robomimic/can
```

### 2. Download Dataset
```bash
# Robomimic CAN dataset - update path as needed
# Copy ph_converted_final dataset to /mnt/data1/minghao/robomimic/can/
```

### 3. Transfer Checkpoints (Optional)
```bash
# Copy trained models from old server:
# /mnt/data1/minghao/robomimic/checkpoints/outputs/robomimic_can_align_recon/outputs/2025-08-22/07-32-53/
```

## Path Configuration

### Update config files if paths differ:
- `conf/train_robomimic_align_with_dynamics.yaml`
- `conf/plan_robomimic_align_7d.yaml`
- Update `ckpt_base_path` and dataset paths

## Environment Variables

### Set working directory:
```bash
cd /path/to/dino_wm
export CUDA_VISIBLE_DEVICES=0  # Adjust GPU selection
```

## Quick Test

### 1. Test Training (1 epoch)
```bash
conda activate wm310
python train_robomimic_align_recon.py --config-path conf --config-name train_robomimic_align_with_dynamics max_epochs=1
```

### 2. Test Planning (if checkpoints available)
```bash
python plan.py --config-path conf --config-name plan_robomimic_align_7d
```

## Key Files Preserved

- ✅ Environment: `environment.yml`, `requirements.txt`
- ✅ Code: All Python scripts and model implementations
- ✅ Configs: All YAML configuration files in `conf/`
- ✅ Documentation: `CLAUDE.md`, `SERVER_SETUP.md`
- ✅ Results Summary: Latest progress in `CLAUDE.md`

## Missing (Need Manual Transfer)

- 🔄 Model checkpoints (~GB): Transfer separately
- 🔄 Dataset: Download robomimic CAN dataset
- 🔄 Training outputs: `outputs/`, `plan_outputs/` (optional)

## GPU Requirements

- Minimum: 1x A100 80GB for training
- Recommended: 4x A100 80GB for parallel experiments
- Planning: Works on 1x GPU with sufficient memory