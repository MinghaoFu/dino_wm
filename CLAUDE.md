
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
  pip install d4rl  # Required for planning evaluation
  ```

- **Environment Variables Setup**:
  ```bash
  # Add to ~/.bashrc for persistent setup
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/minghao/.mujoco/mujoco210/bin:/usr/lib/nvidia' >> ~/.bashrc
  source ~/.bashrc
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

## 🎯 Latest Progress (Sep 9, 2025)

✅ **Projected Latent Planning Implementation Complete**
- **🎯 Architecture Redesign**: Implemented 80D projected latent architecture (64D projected + 16D action)
- **🚀 Action Conditioning**: Following original DINO WM pattern - actions as conditioning, not prediction targets  
- **⚙️ Configurable InfoNCE**: Dynamic alignment dimension matching robomimic state dimensions
- **🔧 DDP Fixes**: Resolved multi-GPU DistributedDataParallel attribute access issues
- **🎯 Enhanced GPU Selection**: Fixed planning issues with proper nvitop-based GPU selection
- **✅ Code Cleanup**: Removed unused 12D z-space and DINO reconstruction components
- **📈 Enhanced Planning**: Added progress bars and auto GPU selection to planning pipeline

**Current Architecture (80D Projected Latent)**:
- **Input**: 384D DINO visual + 32D proprio + 16D action = 432D total
- **Projection**: 416D (visual+proprio) → 64D compressed features
- **Final**: 80D (64D projected + 16D action) for temporal prediction
- **Supervision**: Partial - only 64D projected features supervised, actions used as conditioning
- **InfoNCE**: Configurable alignment_dim (default 7D) for state alignment

**Latest Checkpoint**: `/home/minghao/workspace/dino_wm/outputs/2025-09-09/22-17-40` (80D architecture)

### **Previous Progress (Aug 22, 2025)**

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

### Environment Setup (Complete One-Step)
```bash
# 🚀 Complete environment setup with enhanced setup script
chmod +x setup_and_prepare.sh
./setup_and_prepare.sh

# Or specify custom dataset types and tasks:
./setup_and_prepare.sh "ph,mh" "can,lift,square"

# 🔍 Manual verification after setup
conda activate wm310
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
./select_best_gpus.py single  # Test enhanced GPU selection
```

### **Enhanced Setup Script Features** (`setup_and_prepare.sh`)
- **🎯 Auto Environment Creation**: Creates wm310 conda environment with Python 3.10
- **📦 Complete Package Installation**: All ML packages, GPU tools, and dependencies  
- **🔧 CUDA PyTorch**: Installs CUDA-enabled PyTorch (cu118 index)
- **🎯 Enhanced GPU Tools**: nvitop, pynvml for smart GPU selection
- **🤖 Robomimic Integration**: Auto-installs robosuite and robomimic if directories exist
- **📥 Dataset Download**: Downloads and converts robomimic datasets automatically
- **🔄 DINO WM Conversion**: Converts datasets to DINO WM format if script available
- **⚙️ Environment Variables**: Sets up LD_LIBRARY_PATH, WandB, HuggingFace configs
- **✅ Verification**: Tests PyTorch CUDA and GPU selection functionality

### Dataset Setup & Conversion Scripts
```bash
# Check GPU usage before training
nvidia-smi

# Convert robomimic dataset (if needed)
python convert_robomimic_to_dino_wm.py --input /path/to/hdf5 --output /path/to/ph_converted_final

# Verify dataset structure
ls /path/to/ph_converted_final/  # Should contain: states.pth, actions.pth, velocities.pth, episode_*.mp4
```

### Training Commands

#### 🎯 Smart Auto-GPU Training (Recommended)
```bash
# Auto GPU selection with enhanced nvitop integration - automatically selects best GPU
python train_robomimic_compress.py --config-name=train_robomimic_compress

# Debug with 1 epoch first
python train_robomimic_compress.py --config-name=train_robomimic_compress training.epochs=1 debug=true
```

#### Manual GPU Selection (Optional)
```bash
# Manual specific GPU
CUDA_VISIBLE_DEVICES=5 python train_robomimic_compress.py --config-name=train_robomimic_compress
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
# 80D Projected Latent Architecture - Key settings
projected_dim: 64      # Compressed feature dimension (visual+proprio)
alignment_dim: 7       # InfoNCE alignment with robomimic state (configurable)
num_gpus: 1           # Auto GPU selection
min_free_memory_gb: 2.0

model:
  projected_dim: ${projected_dim}
  alignment_dim: ${alignment_dim}
  post_concat_projection: true  # Enable 416D→64D projection

training:
  batch_size: 32        # Optimized for single GPU
  epochs: 10
  lr: 1e-4
```

**Environment Config**: `conf/env/robomimic_can.yaml` 
```yaml
# UPDATE THIS PATH for new server
save_data_dir = /home/minghao/data/
dataset_path: "{save_data_dir}/can/ph_converted_final"  
```

**Planning Config**: `conf/plan_projected_latent.yaml`
```yaml
# Memory-optimized planning configuration
ckpt_base_path: /home/minghao/workspace/dino_wm/outputs/2025-09-09/22-17-40
model_name: projected_latent_mixed
projected_dim: 64
n_evals: 2            # Reduced from 10 for memory optimization

planner:
  sub_planner:
    horizon: 5
    topk: 10          # Reduced from 30
    num_samples: 100  # Reduced from 300  
    opt_steps: 10     # Reduced from 30
```

#### 🔹 Model Architecture Files
- `models/dino.py`: Clean DINO encoder (removed unused 12D z-space & reconstruction)
- `models/visual_world_model.py`: 80D projected latent with action conditioning
- `models/vit.py`: ViT predictor with attention mechanism (memory bottleneck identified)
- `gpu_utils/gpu_utils.py`: Enhanced nvitop-based GPU selection
- `planning/mpc.py`: MPC planner with progress bars and success tracking

#### 🔹 Architecture Summary
**Current Implementation (80D Projected Latent)**:
1. **Input Processing**: 384D DINO visual + 32D proprio + 16D action = 432D total
2. **Post-Concat Projection**: 416D (visual+proprio) → 64D compressed features
3. **Action Conditioning**: 64D projected + 16D action = 80D final representation
4. **Partial Supervision**: Only 64D projected features supervised, actions as conditioning
5. **InfoNCE Alignment**: Configurable alignment_dim (first N dimensions) → state alignment
6. **Temporal Prediction**: 80D t-1 → 80D t with action replacement during rollout

---

# Training Instructions

## 🚀 Training Commands & Scripts

### **Training Pipeline (80D Projected Latent)**
```bash
# 1. Debug training first (always recommended)
python train_robomimic_compress.py --config-name=train_robomimic_compress training.epochs=1 debug=true

# 2. Full training with auto GPU selection
python train_robomimic_compress.py --config-name=train_robomimic_compress

# 3. Monitor training progress
tail -f outputs/*/logs/train.log
watch -n 1 nvidia-smi
```

### **Key Training Features**
- **Auto GPU Selection**: Enhanced nvitop-based selection built into training script
- **DDP Fixes**: Resolved multi-GPU issues by not preparing main model with accelerator
- **Dynamic Configuration**: Predictor dimension computed as `projected_dim + action_emb_dim`
- **WandB Tracking**: Automatic experiment logging with timestamp naming
- **Path Updates**: Update `conf/env/robomimic_can.yaml` dataset path for new server

### **Training Architecture Implementation**
**File**: `train_robomimic_compress.py`
```python
# Key fixes implemented:
# 1. Dynamic predictor dimension
dim=self.cfg.projected_dim + action_emb_dim,  # 64 + 16 = 80D

# 2. DDP wrapper fix - only prepare individual components
self.encoder, self.predictor, self.decoder = self.accelerator.prepare(
    self.encoder, self.predictor, self.decoder
)
# Model NOT prepared - only moved to device
self.model = self.model.to(self.accelerator.device)
```

---

## 🎯 Planning Evaluation

### **Planning Script Usage**
**CRITICAL**: Always use the planning script with auto GPU selection, never run python directly:

```bash
# Correct way to run planning
./plan.sh  # Uses default config (plan_projected_latent) and horizon (H=5)

# With custom parameters
GOAL_H=10 N_EVALS=5 ./plan.sh

# Quick test with timeout
N_EVALS=2 timeout 180 ./plan.sh

# Check available planning configs
ls conf/plan*.yaml
```

### **Planning Script Features**
**File**: `plan.sh`
```bash
# Enhanced GPU selection using nvitop-based script
if command -v ./select_best_gpus.py &> /dev/null; then
    BEST_GPU=$(./select_best_gpus.py single --quiet)
    echo "Selected GPU: $BEST_GPU"
    export CUDA_VISIBLE_DEVICES=$BEST_GPU
fi

# Configurable parameters with defaults
GOAL_H=${GOAL_H:-5}    # Default horizon 5
N_EVALS=${N_EVALS:-10} # Default number of evaluation seeds
```

### **Planning Configs & Architecture**
**Available Configs:**
- `plan_projected_latent.yaml` - **80D architecture** (64D projected + 16D action)
- `plan_robomimic_align_recon.yaml` - Standard alignment + reconstruction  
- `plan.yaml` - Default planning config

**Current Planning Setup (80D Projected Latent)**:
- **Input**: Load 80D checkpoint from latest training
- **Action Conditioning**: 64D projected features + 16D action during rollout
- **Action Replacement**: Following original DINO WM pattern via `replace_actions_from_z()`
- **InfoNCE Alignment**: Configurable `alignment_dim=7` for state matching
- **Auto GPU Selection**: Enhanced nvitop-based GPU selection prevents resource conflicts
- **Progress Tracking**: Real-time MPC iteration progress with success rate display

### **Planning Implementation Details**
**File**: `models/visual_world_model.py` - Key methods:
```python
# Action replacement during planning rollout
def replace_actions_from_z(self, z, action):
    # Following original DINO WM: replace action portion with CEM planned actions
    z_new = z.clone()
    z_new[:, :, :, self.projected_dim:] = action_repeated
    return z_new

# Rollout with action conditioning  
def rollout(self, obs_0, act):
    # Apply action replacement at each timestep
    z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
```

**File**: `planning/mpc.py` - Enhanced with progress bars:
```python
# Progress bar for MPC iterations
pbar_mpc = tqdm(total=self.max_iter, desc="🎯 MPC Planning", leave=False)
success_rate = np.mean(self.is_success)
pbar_mpc.set_postfix({"Iter": self.iter, "Success": f"{success_rate:.1%}"})
```

### **Planning Results & Output**
```bash
# Results saved to timestamped directories
ls plan_outputs/
# Format: YYYYMMDDHHMMSS_projected_latent_64d_gH{horizon}

# Example output files:
plan_outputs/20250909223045_projected_latent_64d_gH5/
├── plan0_0_success.mp4    # MPC iteration videos
├── plan0_1_failure.mp4   
├── plan0.png             # Trajectory comparison plots
└── logs.json             # Planning metrics and logs
```

---

## ⚠️ Multi-GPU OOM Issue & Solutions

### 🔍 **Problem Analysis** 
Multi-GPU training fails with CUDA OOM despite "free" GPUs due to:

1. **Model Architecture**: ViT attention mechanism (`models/vit.py:71`) creates large attention matrices
2. **Batch Distribution**: Multi-GPU uses batch_size=16 per GPU, but attention memory scales quadratically
3. **Memory Fragmentation**: Previous failed processes leave memory fragments
4. **NCCL Overhead**: Distributed communication adds memory overhead

### ✅ **Solutions & Workarounds**

#### **Recommended: Single GPU Training**
```bash
# 🟢 BEST PRACTICE: Use automatic GPU selection (built-in)
python train_robomimic_compress.py --config-name=train_robomimic_compress
```

**Why it works:**
- Uses full batch_size=32 on one clean GPU
- No distributed overhead
- Better memory management
- Proven stable (verified working)

#### **Multi-GPU Fixes (Advanced)**
If multi-GPU is required, try these approaches:

**Option 1: Reduce Batch Size**
```bash
# Modify config to use smaller batch size per GPU
BEST_GPUS=$(./select_best_gpus.py)
CUDA_VISIBLE_DEVICES=$BEST_GPUS accelerate launch --num_processes=2 --gpu_ids=0,1 train_robomimic_compress.py --config-name=train_robomimic_compress training.batch_size=8
```

**Option 2: Gradient Checkpointing**
```bash
# Enable gradient checkpointing to trade compute for memory
CUDA_VISIBLE_DEVICES=$BEST_GPUS accelerate launch --num_processes=2 --gpu_ids=0,1 train_robomimic_compress.py --config-name=train_robomimic_compress model.gradient_checkpointing=true
```

**Option 3: Mixed Precision**
```bash
# Use automatic mixed precision (FP16)
CUDA_VISIBLE_DEVICES=$BEST_GPUS accelerate launch --mixed_precision=fp16 --num_processes=2 --gpu_ids=0,1 train_robomimic_compress.py --config-name=train_robomimic_compress
```

### 📊 **Enhanced GPU Selection Utility**
The enhanced `select_best_gpus.py` script now uses **nvitop** for superior GPU monitoring:

**Key Features:**
- **🎯 Smart Scoring**: Combines memory (70%) + utilization (30%) for optimal selection
- **📊 Rich Monitoring**: Real-time memory and GPU utilization display
- **🎨 Color Indicators**: 🟢 Excellent (<20), 🟡 Moderate (20-50), 🔴 Heavy (>50)
- **⚙️ Configurable Weights**: Adjust memory vs utilization importance
- **🔄 Fallback Support**: Auto-falls back to nvidia-ml-py if nvitop unavailable

**Usage Examples:**
```bash
# Single GPU selection (recommended)
./select_best_gpus.py single

# Multi-GPU selection
./select_best_gpus.py

# Quiet mode (for scripting)
./select_best_gpus.py single --quiet

# Custom weights (prioritize memory over utilization)
./select_best_gpus.py single --memory-weight 0.8 --util-weight 0.2
```

**Example Output:**
```bash
$ ./select_best_gpus.py single
GPU Utilization & Memory Status:
GPU Mem%   Util%  MemMB    Score 
-----------------------------------
3   0.0    0.0    0        0.0    🟢
5   0.0    0.0    0        0.0    🟢  
2   51.9   17.0   42317    41.4   🟡
4   56.4   0.0    45975    39.5   🟡
Selected GPU: 3
```

**Why This is Better:**
- **GPU 6**: 28% memory, 61% utilization → Score 37.9 (busy with compute)
- **GPU 3**: 0% memory, 0% utilization → Score 0.0 (completely free)
- **Old script**: Would only see 28% vs 0% memory, miss the utilization
- ● The issue is that when CUDA_VISIBLE_DEVICES=4 is set, GPU 4 becomes GPU index 0 in the CUDA
   context. But somewhere in the code is still trying to use the wrong index. Let me run with
   a direct GPU selection to avoid the script's auto-selection: please fix this bug, why still trying
- Enhanced nvitop-based GPU selection prevents resource conflicts