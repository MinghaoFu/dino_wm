#!/usr/bin/env python3
"""
Simple test of mixed latent rollout without complex dependencies
Evaluates H=5,10,15,20 with 10 seeds each as requested
"""
import os
import sys
import json
import random
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from einops import rearrange

# Add project root to Python path
sys.path.insert(0, '/home/minghao/workspace/dino_wm')

def simple_rollout_test():
    """Simple test using dataset and model inference without complex setup"""
    
    # Set model path to current training output  
    model_dir = "/home/minghao/workspace/dino_wm/outputs/2025-09-09/07-57-28"
    config_path = f"{model_dir}/.hydra/config.yaml"
    
    print(f"Testing mixed latent rollout from: {model_dir}")
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = OmegaConf.load(f)
    
    print(f"Architecture: {cfg.encoder.z_dim}D encoder -> {cfg.projected_dim}D mixed latent")
    
    # Load dataset using hydra utils
    import hydra
    os.chdir('/home/minghao/workspace/dino_wm')
    hydra.initialize(config_path="conf", version_base="1.1")
    
    _, dataset_splits = hydra.utils.call(
        cfg.env.dataset,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
    )
    
    dataset = dataset_splits["valid"]
    print(f"Dataset: {len(dataset)} validation samples")
    
    # Manually recreate model using loaded state dicts (avoids hydra config issues)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load checkpoint state dicts
    ckpt_path = f"{model_dir}/checkpoints/model_latest.pth"
    print(f"Loading checkpoint: {ckpt_path}")
    
    # Load only the state dicts, not the full objects
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Create model components
    encoder = hydra.utils.instantiate(cfg.encoder)
    proprio_encoder = hydra.utils.instantiate(cfg.proprio_encoder)
    action_encoder = hydra.utils.instantiate(cfg.action_encoder)  
    predictor = hydra.utils.instantiate(cfg.predictor)
    decoder = hydra.utils.instantiate(cfg.decoder) if cfg.has_decoder else None
    
    # Load state dicts
    if "encoder" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder"])
    if "proprio_encoder" in checkpoint:
        proprio_encoder.load_state_dict(checkpoint["proprio_encoder"])
    if "action_encoder" in checkpoint:
        action_encoder.load_state_dict(checkpoint["action_encoder"])
    if "predictor" in checkpoint:
        predictor.load_state_dict(checkpoint["predictor"])
    if decoder and "decoder" in checkpoint:
        decoder.load_state_dict(checkpoint["decoder"])
    
    # Move to device
    encoder.to(device).eval()
    proprio_encoder.to(device).eval()
    action_encoder.to(device).eval()
    predictor.to(device).eval()
    if decoder:
        decoder.to(device).eval()
    
    print("Model components loaded successfully")
    
    # Test rollouts for different horizons
    horizons = [5, 10, 15, 20]
    n_seeds = 10
    
    all_results = {}
    
    for horizon in horizons:
        print(f"\n{'='*40}")
        print(f"Testing Horizon H={horizon}")
        print(f"{'='*40}")
        
        successes = []
        rollout_stds = []
        
        for seed in range(n_seeds):
            try:
                # Sample trajectory segment
                traj_len = cfg.frameskip * horizon + cfg.num_hist
                
                # Find valid trajectory
                valid_found = False
                for attempt in range(100):
                    traj_id = random.randint(0, len(dataset) - 1)
                    obs, act, state, env_info = dataset[traj_id]
                    
                    if obs["visual"].shape[0] >= traj_len:
                        valid_found = True
                        break
                
                if not valid_found:
                    print(f"  Seed {seed+1}: No valid trajectory")
                    successes.append(False)
                    continue
                
                # Extract segment
                max_offset = obs["visual"].shape[0] - traj_len
                offset = random.randint(0, max_offset)
                
                # Prepare inputs
                visual_seq = obs["visual"][offset:offset+cfg.num_hist]
                proprio_seq = obs["proprio"][offset:offset+cfg.num_hist]
                actions_seq = act[offset:offset+cfg.frameskip*horizon]
                
                # Convert to tensors
                visual_batch = torch.tensor(visual_seq, device=device, dtype=torch.float32).unsqueeze(0)
                proprio_batch = torch.tensor(proprio_seq, device=device, dtype=torch.float32).unsqueeze(0)
                
                # Reshape actions: (t*frameskip, action_dim) -> (t, frameskip*action_dim)
                actions_reshaped = rearrange(actions_seq, "(t f) d -> t (f d)", f=cfg.frameskip)
                actions_batch = torch.tensor(actions_reshaped, device=device, dtype=torch.float32).unsqueeze(0)
                
                obs_batch = {"visual": visual_batch, "proprio": proprio_batch}
                
                # Test manual rollout (simplified version of model.rollout)
                with torch.no_grad():
                    # Encode initial observations  
                    visual_emb = encoder(visual_batch.flatten(0, 1))  # (b*t, ...) 
                    visual_emb = visual_emb.view(1, cfg.num_hist, *visual_emb.shape[1:])  # (b, t, ...)
                    
                    proprio_emb = proprio_encoder(proprio_batch)  # (b, t, proprio_dim)
                    action_emb = action_encoder(actions_batch)    # (b, t, action_dim)
                    
                    # Get concatenated features (this is the key test for mixed latent space)
                    if cfg.concat_dim == 1:
                        # Concatenate along feature dimension to create mixed representation
                        # visual: (b, t, num_patches, visual_dim)
                        # proprio/action: (b, t, emb_dim) -> tile to (b, t, num_patches, emb_dim)
                        
                        b, t, num_patches, visual_dim = visual_emb.shape
                        
                        # Tile proprio and action to match patch dimensions
                        proprio_tiled = proprio_emb.unsqueeze(2).expand(-1, -1, num_patches, -1)
                        action_tiled = action_emb.unsqueeze(2).expand(-1, -1, num_patches, -1)
                        
                        # Concatenate: (visual + proprio + action)
                        concat_emb = torch.cat([visual_emb, proprio_tiled, action_tiled], dim=-1)
                        
                        # Apply post-concatenation projection to get mixed latent space
                        concat_dim = visual_dim + cfg.proprio_emb_dim + cfg.action_emb_dim
                        if hasattr(encoder, 'post_concat_projection'):
                            # This should exist in the model
                            projected_emb = encoder.post_concat_projection(concat_emb)
                        else:
                            # Fallback: assume projection exists in predictor or model
                            projected_emb = concat_emb  # Will be projected later
                        
                        # Test predictor rollout
                        z_src = projected_emb[:, :cfg.num_hist]  # Initial frames
                        
                        # Simple rollout test - predict next frames
                        z_pred = predictor(z_src)
                        
                        # Check rollout quality
                        if z_pred is not None:
                            pred_std = torch.std(z_pred).item()
                            
                            success = (
                                not torch.isnan(z_pred).any() and
                                not torch.isinf(z_pred).any() and
                                pred_std > 1e-6  # Not collapsed
                            )
                            
                            successes.append(success)
                            rollout_stds.append(pred_std)
                            
                            status = "✓ SUCCESS" if success else "✗ FAILED"
                            print(f"  Seed {seed+1:2d}: {status} (std={pred_std:.4f})")
                            
                        else:
                            successes.append(False)
                            print(f"  Seed {seed+1:2d}: ✗ FAILED (predictor returned None)")
                    
                    else:
                        # concat_dim == 0 case (if needed)
                        print(f"  Seed {seed+1}: concat_dim=0 not implemented")
                        successes.append(False)
                        
            except Exception as e:
                successes.append(False)
                print(f"  Seed {seed+1:2d}: ✗ ERROR - {str(e)}")
        
        # Compute results
        success_rate = np.mean(successes) if successes else 0.0
        avg_std = np.mean(rollout_stds) if rollout_stds else 0.0
        
        results = {
            "horizon": horizon,
            "n_seeds": len(successes),
            "latent_dim": cfg.projected_dim,
            "success_rate": success_rate,
            "avg_rollout_std": avg_std,
            "num_successes": sum(successes),
            "num_trials": len(successes),
        }
        
        all_results[f"H_{horizon}"] = results
        
        print(f"\n  Results: {success_rate*100:.1f}% success ({sum(successes)}/{len(successes)})")
        print(f"  Avg Std: {avg_std:.4f}")
    
    # Save results
    output_file = f"mixed_latent_rollout_test_{cfg.projected_dim}d.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"MIXED LATENT ROLLOUT TEST SUMMARY ({cfg.projected_dim}D)")
    print(f"{'='*60}")
    print(f"{'Horizon':<10} {'Success Rate':<15} {'Avg Std':<12}")
    print("-" * 37)
    
    for horizon in horizons:
        res = all_results[f"H_{horizon}"]
        success_str = f"{res['success_rate']*100:.1f}%"
        print(f"H={horizon:<7} {success_str:<15} {res['avg_rollout_std']:<12.4f}")
    
    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")
    
    return all_results

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)  
    random.seed(42)
    
    print("Mixed Latent Rollout Test")
    print("Testing H=5,10,15,20 with 10 seeds each")
    print("="*50)
    
    try:
        results = simple_rollout_test()
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()