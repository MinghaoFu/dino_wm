#!/usr/bin/env python3

"""
Debug script to understand action dimension mismatch between dataset and model
"""
import torch
import hydra
from pathlib import Path
from omegaconf import OmegaConf

def main():
    print("=== ACTION DIMENSION DEBUG ===")
    
    # Load model config
    model_path = Path('/mnt/data1/minghao/robomimic/checkpoints/outputs/robomimic_can_no_align/outputs/2025-08-06/21-27-17')
    model_cfg = OmegaConf.load(model_path / 'hydra.yaml')
    
    print("Model Config:")
    print(f"  action_emb_dim: {model_cfg.action_emb_dim}")
    print(f"  num_action_repeat: {model_cfg.num_action_repeat}")
    print(f"  Environment: {model_cfg.env.name}")
    print(f"  with_velocity: {model_cfg.env.kwargs.with_velocity}")
    print(f"  with_target: {model_cfg.env.kwargs.with_target}")
    
    # Load dataset
    print("\nLoading dataset...")
    datasets, traj_dsets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    
    # Check dataset properties
    val_dataset = datasets["valid"]
    val_traj_dset = traj_dsets["valid"]
    
    print("\nDataset Properties:")
    print(f"  Dataset len: {len(val_dataset)}")
    print(f"  Trajectory len: {len(val_traj_dset)}")
    
    # Sample from dataset
    sample = val_dataset[0]
    obs, act, state = sample
    
    print(f"\nDataset Sample:")
    print(f"  obs['visual'] shape: {obs['visual'].shape}")
    print(f"  obs['proprio'] shape: {obs['proprio'].shape}")  
    print(f"  act shape: {act.shape}")
    print(f"  state shape: {state.shape}")
    
    # Sample from trajectory dataset
    traj_sample = val_traj_dset[0]
    t_obs, t_act, t_state, t_info = traj_sample
    
    print(f"\nTrajectory Sample:")
    print(f"  obs['visual'] shape: {t_obs['visual'].shape}")
    print(f"  obs['proprio'] shape: {t_obs['proprio'].shape}")
    print(f"  act shape: {t_act.shape}")
    print(f"  state shape: {t_state.shape}")
    print(f"  info: {t_info}")
    
    # Check action statistics
    if hasattr(val_dataset, 'action_dim'):
        print(f"\nDataset action_dim: {val_dataset.action_dim}")
    
    # Load model checkpoint to see what action encoder expects
    model_ckpt = model_path / 'checkpoints' / 'model_latest.pth'
    ckpt = torch.load(model_ckpt, map_location='cpu', weights_only=False)
    
    if 'action_encoder' in ckpt:
        action_encoder = ckpt['action_encoder']
        print(f"\nAction Encoder in checkpoint:")
        print(f"  Type: {type(action_encoder)}")
        if hasattr(action_encoder, 'patch_embed'):
            conv = action_encoder.patch_embed
            print(f"  Input channels: {conv.in_channels}")
            print(f"  Output channels: {conv.out_channels}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Dataset provides: {t_act.shape[-1]}D actions")
    print(f"Model expects: {ckpt['action_encoder'].patch_embed.in_channels}D actions")
    print(f"Action embedding dim: {model_cfg.action_emb_dim}")

if __name__ == "__main__":
    main()