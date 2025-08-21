### Dino World Model Training Pipeline

This directory contains the training pipeline for a **world model** using a **pretrained DINOv2 encoder**. The learned latent space is aligned with the ground-truth state variables.

Everytime having progress on code, please update this file.

## 🎯 Latest Progress (Aug 13, 2025)

✅ **Successfully implemented configurable DINO reconstruction loss system**
- **Three Model Variants**: No alignment, alignment only, alignment + DINO reconstruction  
- **Unified Training**: `train_robomimic_align.py` handles both alignment modes automatically
- **DINO Reconstruction**: 384D→128D→384D decoder with MSE loss for feature preservation
- **Auto Configuration**: Model naming, wandb projects (dino_wm_align/dino_wm_align_recon) handled automatically
- **Memory Optimized**: Batch size 64, working on 4 A100 GPUs with automatic selection
- **Working Configs**: `train_robomimic_align.yaml` and `train_robomimic_align_with_recon.yaml` both verified

**Training Protocol**: Debug 1 epoch first, then parallel experiments (align vs align+recon) in background; are you sure it is in running? please run online to confirm it can work then offline
**Checking training**: using "nvitop" to see how the programmes using gpu well
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