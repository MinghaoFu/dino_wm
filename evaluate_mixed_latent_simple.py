#!/usr/bin/env python3
"""
Simple evaluation of mixed latent representation rollout capability
Tests the model's ability to rollout in the mixed latent space
"""
import os
import sys
import json
import torch
import random
import numpy as np
from pathlib import Path
import hydra
from omegaconf import OmegaConf

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def load_trained_model():
    """Load the trained model and config"""
    
    # Model path
    model_dir = "/home/minghao/workspace/dino_wm/outputs/2025-09-09/07-57-28"
    config_path = f"{model_dir}/.hydra/config.yaml"
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = OmegaConf.load(f)
    
    print(f"Loaded config: {cfg.encoder.z_dim}D encoder -> {cfg.projected_dim}D mixed latent")
    
    # Load dataset first to get the data setup
    _, dataset = hydra.utils.call(
        cfg.env.dataset,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
    )
    
    dataset_val = dataset["valid"]
    print(f"Dataset loaded: {len(dataset_val)} validation samples")
    
    # Load model checkpoint manually using training script approach
    sys.path.insert(0, model_dir)
    
    # Import the training module to get model initialization
    from train_robomimic_compress import Trainer
    
    # Create trainer with loaded config
    trainer = Trainer(cfg)
    
    # Load latest checkpoint
    ckpt_path = f"{model_dir}/checkpoints/model_latest.pth"
    trainer.accelerator.load_state(ckpt_path)
    
    print(f"Model loaded from {ckpt_path}")
    
    return trainer.model, cfg, dataset_val

def evaluate_rollout_quality(model, cfg, dataset, horizons=[5, 10, 15, 20], n_seeds=10):
    """Evaluate rollout quality for different horizons"""
    
    model.eval()
    device = next(model.parameters()).device
    
    all_results = {}
    
    for horizon in horizons:
        print(f"\n{'='*50}")
        print(f"Evaluating Horizon H={horizon}")
        print(f"{'='*50}")
        
        successes = []
        rollout_errors = []
        
        for seed in range(n_seeds):
            try:
                # Sample a trajectory segment
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
                    print(f"  Seed {seed+1}: No valid trajectory found")
                    continue
                
                # Sample segment
                max_offset = obs["visual"].shape[0] - traj_len
                offset = random.randint(0, max_offset)
                
                # Prepare input
                obs_seq = {
                    key: arr[offset:offset+cfg.num_hist] 
                    for key, arr in obs.items()
                }
                actions_seq = act[offset:offset+cfg.frameskip*horizon]
                
                # Reshape actions for model (t*frameskip, action_dim) -> (t, frameskip*action_dim)
                from einops import rearrange
                actions_reshaped = rearrange(
                    actions_seq, "(t f) d -> t (f d)", f=cfg.frameskip
                )
                
                # Convert to tensors
                obs_batch = {}
                for key, val in obs_seq.items():
                    obs_batch[key] = torch.tensor(val, device=device, dtype=torch.float32).unsqueeze(0)
                
                actions_batch = torch.tensor(
                    actions_reshaped, device=device, dtype=torch.float32
                ).unsqueeze(0)
                
                # Test rollout
                with torch.no_grad():
                    try:
                        # Use the model's rollout method
                        z_pred, z_final = model.rollout(obs_batch, actions_batch)
                        
                        if z_final is not None:
                            # Check rollout statistics
                            final_mean = torch.mean(z_final).item()
                            final_std = torch.std(z_final).item()
                            
                            # Simple success criteria
                            success = (
                                not torch.isnan(z_final).any() and 
                                not torch.isinf(z_final).any() and
                                final_std > 1e-6  # Not collapsed
                            )
                            
                            successes.append(success)
                            rollout_errors.append(final_std)
                            
                            print(f"  Seed {seed+1}: Success={success}, Final std={final_std:.4f}")
                        else:
                            successes.append(False)
                            print(f"  Seed {seed+1}: Rollout returned None")
                            
                    except Exception as e:
                        successes.append(False)
                        print(f"  Seed {seed+1}: Rollout failed - {str(e)}")
                        
            except Exception as e:
                print(f"  Seed {seed+1}: Failed - {str(e)}")
                successes.append(False)
        
        # Compute results
        success_rate = np.mean(successes) if successes else 0.0
        avg_error = np.mean(rollout_errors) if rollout_errors else float('inf')
        
        results = {
            "horizon": horizon,
            "n_seeds": len(successes),
            "success_rate": success_rate,
            "avg_rollout_std": avg_error,
            "num_successes": sum(successes),
            "num_trials": len(successes),
        }
        
        all_results[f"H_{horizon}"] = results
        
        print(f"\nResults for H={horizon}:")
        print(f"  Success Rate: {success_rate*100:.1f}% ({sum(successes)}/{len(successes)})")
        print(f"  Avg Rollout Std: {avg_error:.4f}")
    
    return all_results

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("Loading trained mixed latent model...")
    
    try:
        model, cfg, dataset = load_trained_model()
        
        print(f"\nModel loaded successfully!")
        print(f"Architecture: {cfg.encoder.z_dim}D -> {cfg.projected_dim}D mixed latent")
        print(f"Dataset: {len(dataset)} validation samples")
        
        # Run evaluation
        horizons = [5, 10, 15, 20]
        n_seeds = 10
        
        results = evaluate_rollout_quality(
            model, cfg, dataset, horizons=horizons, n_seeds=n_seeds
        )
        
        # Save results
        output_file = f"rollout_evaluation_{cfg.projected_dim}d_mixed_latent.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"ROLLOUT EVALUATION SUMMARY ({cfg.projected_dim}D Mixed Latent)")
        print(f"{'='*60}")
        print(f"{'Horizon':<10} {'Success Rate':<15} {'Avg Rollout Std':<15}")
        print("-" * 40)
        
        for horizon in horizons:
            res = results[f"H_{horizon}"]
            print(f"H={horizon:<7} {res['success_rate']*100:>6.1f}%        {res['avg_rollout_std']:>8.4f}")
        
        print(f"{'='*60}")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()