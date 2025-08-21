"""
Simple evaluation of trained robomimic DINO world model
Tests forward prediction on dataset samples without full environment planning
"""
import torch
import hydra
import json
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms

def load_ckpt(filename, device="cpu"):
    ckpt = torch.load(filename, map_location=device, weights_only=False)
    return ckpt

def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    
    # Override forward method to skip alignment logic (same as training)
    def forward_no_alignment(obs, act, state=None):
        loss = 0
        loss_components = {}
        z = model.encode(obs, act)
        z_src = z[:, : model.num_hist, :, :]
        z_tgt = z[:, model.num_pred :, :, :]
        
        if model.predictor is not None:
            z_pred = model.predict(z_src)
            
            # Compute prediction loss 
            if model.concat_dim == 1:
                z_visual_loss = model.emb_criterion(
                    z_pred[:, :, :, :-(model.proprio_dim + model.action_dim)], 
                    z_tgt[:, :, :, :-(model.proprio_dim + model.action_dim)].detach()
                )
                z_proprio_loss = model.emb_criterion(
                    z_pred[:, :, :, -(model.proprio_dim + model.action_dim): -model.action_dim], 
                    z_tgt[:, :, :, -(model.proprio_dim + model.action_dim): -model.action_dim].detach()
                )
                z_loss = model.emb_criterion(
                    z_pred[:, :, :, :-model.action_dim], 
                    z_tgt[:, :, :, :-model.action_dim].detach()
                )
            
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss  
            loss_components["z_proprio_loss"] = z_proprio_loss
            
            return z_pred, z_tgt, loss_components
        else:
            return None, None, {}
            
    # Bind the new forward method to the model
    model.forward = forward_no_alignment
    model.to(device)
    return model

def evaluate_model():
    print("=== ROBOMIMIC DINO WORLD MODEL EVALUATION ===")
    
    # Load model config
    model_path = Path('/mnt/data1/minghao/robomimic/checkpoints/outputs/robomimic_can_no_align/outputs/2025-08-06/21-27-17')
    model_cfg = OmegaConf.load(model_path / 'hydra.yaml')
    
    # Load checkpoint
    model_ckpt = model_path / 'checkpoints' / 'model_latest.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from: {model_ckpt}")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {model_cfg.env.dataset.data_path}")
    datasets, traj_dsets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    
    # Load model
    num_action_repeat = model_cfg.get("num_action_repeat", 1)
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device)
    model.eval()
    
    print(f"\n=== MODEL ARCHITECTURE ===")
    print(f"Encoder embedding dim: {model.encoder.emb_dim}")
    print(f"Action embedding dim: {model.action_dim}")
    print(f"Proprio embedding dim: {model.proprio_dim}")
    print(f"Total embedding dim: {model.encoder.emb_dim + model.proprio_dim + model.action_dim}")
    
    # Evaluate on validation set
    print(f"\n=== EVALUATION RESULTS ===")
    val_dataset = datasets["valid"]
    dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    total_losses = {"z_loss": [], "z_visual_loss": [], "z_proprio_loss": []}
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            obs, act, state = batch
            
            # Move to device
            obs = {k: v.to(device) for k, v in obs.items()}
            act = act.to(device)
            if state is not None:
                state = state.to(device)
                
            # Forward pass
            z_pred, z_tgt, loss_components = model(obs, act, state)
            
            # Collect losses
            for key in total_losses.keys():
                if key in loss_components:
                    total_losses[key].append(loss_components[key].item())
            
            if i >= 20:  # Evaluate on subset for speed
                break
    
    # Calculate average losses
    results = {}
    for key, values in total_losses.items():
        if values:
            results[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "count": len(values)
            }
    
    print(f"Visual prediction loss: {results['z_visual_loss']['mean']:.6f} ± {results['z_visual_loss']['std']:.6f}")
    print(f"Proprio prediction loss: {results['z_proprio_loss']['mean']:.6f} ± {results['z_proprio_loss']['std']:.6f}") 
    print(f"Total prediction loss: {results['z_loss']['mean']:.6f} ± {results['z_loss']['std']:.6f}")
    
    # Save results
    with open("robomimic_model_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation completed! Results saved to robomimic_model_evaluation.json")
    print(f"\n=== SUMMARY ===")
    print(f"Model: DINO World Model (No Alignment)")
    print(f"Dataset: RoboMimic Can Task")
    print(f"Dimensions: 432D (384D DINO + 16D action + 32D proprio)")
    print(f"Training: 100 epochs, 4×A100 GPUs")
    print(f"Validation loss: {results['z_loss']['mean']:.6f}")
    
    return results

if __name__ == "__main__":
    results = evaluate_model()