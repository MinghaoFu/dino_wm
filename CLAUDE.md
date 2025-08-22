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

#### Environment Setting:
cd /home/minghao/workspace/dino_wm
conda activate wm310

---

#### 🔹 Training Procedure

1. **Latent Projection**  
   Add a projection layer on top of the frozen DINO embedding to map it to a 128-dimensional estimated latent space.

2. **Alignment with True State Variables**  
   Incorporate an alignment loss to ensure that part of the estimated latent space corresponds to the true state variables.

   **Alignment Strategies:**
   - **Option 1:** Linear alignment using **InfoNCE** loss.
   - **Option 2:** Reconstruct the true state variables directly from the aligned part of the estimated latent space.

3. **DINO Reconstruction (NEW)**
   Optional decoder from 128D projected features back to 384D original DINO features with MSE loss.
   
4. **Downstream Evaluation**  
   Evaluate the model's planning performance using the trained checkpoint to assess how well the latent space captures useful dynamics.

---

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

**Everytime you train model, please first run it one epoch to debug whether it can run well, if successful, then train it in background**
**For each training, please train 2 experiments: one with alignment loss, one with alignment+reconstruction loss. Name them with different suffixes in wandb.**
**Everytime you train model, please automatically select GPUs with lowest occupation using available scripts**