
### New Server Setup Instructions

#### Environment Setup
- **Create and Activate Conda Environment**:
  ```bash
  conda create -n wm310 python=3.10 -y
  source $(conda info --base)/etc/profile.d/conda.sh
  conda activate wm310
  ```
- **Install Required Packages**:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install transformers==4.28.0 huggingface_hub==0.23.4
  pip install scipy numpy Pillow opencv-python termcolor tqdm
  pip install diffusers==0.11.1 egl_probe>=1.0.1 h5py imageio imageio-ffmpeg matplotlib psutil tensorboard tensorboardX
  pip install accelerate hydra-core wandb einops
  pip install hf_transfer
  ```

#### Dataset Download and Conversion
- **Download Dataset**:
  ```bash
  # Define dataset types and tasks
  dataset_types=("ph")
  tasks=("can")
  # Download datasets
  for task in "${tasks[@]}"; do
      for dataset_type in "${dataset_types[@]}"; do
          python $robomimic_dir/robomimic/scripts/download_datasets.py \
              --tasks $task --dataset_types $dataset_type --hdf5_types all --download_dir $dataset_dir
      done
  done
  ```
- **Convert Dataset**:
  ```bash
  # Convert states to images
  python $robomimic_dir/robomimic/scripts/dataset_states_to_obs.py \
      --dataset $dataset_dir/$task/$dataset_type/demo_v15.hdf5 \
      --output_name $dataset_dir/$task/$dataset_type/image_384_v15.hdf5 \
      --done_mode 2 \
      --camera_names agentview robot0_eye_in_hand \
      --camera_height 384 \
      --camera_width 384
  ```

#### FFmpeg Configuration
- Ensure FFmpeg is installed and configured correctly to avoid frame writing issues.

#### Training Setup
- **Environment Variables**:
  ```bash
  export WANDB_BASE_URL=https://api.bandw.top
  export HF_ENDPOINT=https://hf-mirror.com
  export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface
  export HF_HUB_ENABLE_HF_TRANSFER=1
  ```

### Dino World Model Training Pipeline

This directory contains the training pipeline for a **world model** using a **pretrained DINOv2 encoder**. The learned latent space is aligned with the ground-truth state variables.

Everytime having progress on code, please update this file.

## 🎯 Latest Progress (Aug 22, 2025)

✅ **Successfully implemented and evaluated 7D temporal dynamics loss system**
- **Configurable 7D Dynamics**: Added MSE loss on 7D aligned features o_{t-1}→o_t prediction
- **Training Complete**: Model trained to epoch 65 with alignment + DINO reconstruction + 7D dynamics
- **Planning Evaluation**: Comprehensive horizon analysis (H=3,5,10,15,20,25) completed

### **7D Dynamics Planning Results**
| Horizon | Success Rate | Avg Distance |
|---------|-------------|--------------|
| H=3,5,10| 10% | 0.59-0.92 |
| H=15    | **60%** | 0.72 |
| H=20    | **70%** | 0.73 |
| H=25    | 30% | 0.70 |

**Key Finding**: 7D temporal dynamics most effective for medium horizons (H=15-20), achieving 60-70% success rates vs 10% for short horizons.

**Checkpoint**: `/mnt/data1/minghao/robomimic/checkpoints/outputs/robomimic_can_align_recon/outputs/2025-08-22/07-32-53`
---

#### 🔹 Background: Dino World Model

The DINO world model employs a **pretrained DINO encoder** as a fixed feature extractor. The model is trained using DINO embeddings for next-step prediction. Once the model converges, it is evaluated on planning tasks using the pretrained checkpoint.

---

#### 🔹 Dataset

- **robomimic**

## 🚀 Server Migration Setup Guide

### Environment Setup (One-Step)
```bash
# 1. Create conda environment from environment.yml
conda env create -f environment.yml
conda activate wm310

# 2. Install additional packages
pip install accelerate hydra-submitit-launcher

# 3. Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Dataset Setup & Conversion Scripts
```bash
# Check GPU usage before training
nvidia-smi

# Convert robomimic dataset (if needed)
python convert_robomimic_to_dino_wm.py --input /path/to/hdf5 --output /path/to/ph_converted_final

# Verify dataset structure
ls /path/to/ph_converted_final/  # Should contain: states.pth, actions.pth, velocities.pth, episode_*.mp4
```

### Training Commands Used

#### Debug Training (Recommended first step)
```bash
# Auto-select lowest GPU and debug train
python train_robomimic_compress.py --config-name=train_robomimic_compress training.epochs=1 debug=true

# Manual GPU selection for debug
CUDA_VISIBLE_DEVICES=5 python train_robomimic_compress.py --config-name=train_robomimic_compress training.epochs=1 debug=true
```

#### Full Training
```bash
# Auto GPU selection (recommended)
python train_robomimic_compress.py --config-name=train_robomimic_compress

# Specific GPU
CUDA_VISIBLE_DEVICES=5 python train_robomimic_compress.py --config-name=train_robomimic_compress

# Multi-GPU distributed (if needed)
CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes=2 --gpu_ids=0,1 train_robomimic_compress.py --config-name=train_robomimic_compress
```

#### Monitoring & Utilities
```bash
# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Check training logs
tail -f outputs/*/logs/train.log

# WandB dashboard
# https://wandb.ai/causality_/dino_wm_debug/
```

#### Environment Setting:
```bash
cd /home/minghao/workspace/dino_wm
conda activate wm310
```

---

### Key Files & Configuration

#### 🔹 Main Training Script
**File**: `train_robomimic_compress.py`
- Includes auto GPU selection via `gpu_utils/gpu_utils.py`
- WandB tracking with experiment naming
- Supports debug mode and distributed training

#### 🔹 Configuration Files
**Main Config**: `conf/train_robomimic_compress.yaml`
```yaml
# Key settings for server migration - update paths as needed
encoder:
  z_dim: 12  # 12D z-space architecture
  recon_dino_loss: true
alignment:
  dynamics_7d_loss_weight: 1.0  # Full 12D dynamics
num_gpus: 1  # Auto GPU selection
min_free_memory_gb: 2.0
```

**Environment Config**: `conf/env/robomimic_can.yaml` 
```yaml
# UPDATE THIS PATH for new server
save_data_dir = /home/fuminghao/data/
dataset_path: "{save_data_dir}/can/ph_converted_final"  
```

#### 🔹 Model Architecture Files
- `models/dino.py`: 12D MLP encoder (384D→12D) + reconstruction decoder
- `models/visual_world_model.py`: InfoNCE alignment + temporal dynamics
- `gpu_utils/gpu_utils.py`: Auto GPU selection utility

#### 🔹 Architecture Summary
**Current Implementation (12D InfoNCE)**:
1. **MLP Encoder**: 384D DINO → 12D z-space  
2. **InfoNCE Alignment**: First 7D of z-space → 7D state alignment
3. **Temporal Dynamics**: Full 12D t-1 → t prediction (dynamic_ratio = 1.0)
4. **Reconstruction**: 12D z → 384D DINO features

---

# Training Instructions
- **Always debug first**: Run 1 epoch before full training
- **Auto GPU selection**: Script automatically chooses lowest occupied GPUs  
- **Path updates**: Update `conf/env/robomimic_can.yaml` dataset path for new server
- **WandB tracking**: Experiments logged with different suffixes
