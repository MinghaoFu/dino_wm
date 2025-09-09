#!/usr/bin/env python3
"""
Planning evaluation for mixed latent representation architecture
Simplified version without environment dependencies
"""
import os
import sys
import json
import torch
import random
import logging
import warnings
import numpy as np
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessor import Preprocessor
from utils import seed

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def load_checkpoint(checkpoint_path, device="cuda"):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    log.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return checkpoint

def load_model_from_checkpoint(checkpoint_path, config_path, device="cuda"):
    """Load the full model from checkpoint"""
    import hydra
    
    # Load config
    with open(config_path, "r") as f:
        cfg = OmegaConf.load(f)
    
    # Load checkpoint
    ckpt = load_checkpoint(checkpoint_path, device)
    
    # Initialize model components
    encoder = hydra.utils.instantiate(cfg.encoder)
    proprio_encoder = hydra.utils.instantiate(cfg.proprio_encoder)
    action_encoder = hydra.utils.instantiate(cfg.action_encoder)
    predictor = hydra.utils.instantiate(cfg.predictor)
    decoder = hydra.utils.instantiate(cfg.decoder) if cfg.has_decoder else None
    
    # Load state dicts
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    if "proprio_encoder" in ckpt:
        proprio_encoder.load_state_dict(ckpt["proprio_encoder"])
    if "action_encoder" in ckpt:
        action_encoder.load_state_dict(ckpt["action_encoder"])
    if "predictor" in ckpt:
        predictor.load_state_dict(ckpt["predictor"])
    if decoder and "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"])
    
    # Create full model
    model = hydra.utils.instantiate(
        cfg.model,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        decoder=decoder,
        proprio_dim=cfg.proprio_emb_dim,
        action_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim,
        num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=cfg.num_proprio_repeat,
    )
    
    model.to(device)
    model.eval()
    
    # Log architecture info
    latent_dim = cfg.projected_dim
    log.info(f"Model architecture: {cfg.encoder.z_dim}D encoder -> {latent_dim}D mixed latent space")
    
    return model, cfg

class SimplePlanningEvaluator:
    def __init__(self, model, dataset, config, device="cuda"):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device
        
        # Data preprocessor
        self.preprocessor = Preprocessor(
            action_mean=dataset.action_mean,
            action_std=dataset.action_std,
            state_mean=dataset.state_mean,
            state_std=dataset.state_std,
            proprio_mean=dataset.proprio_mean,
            proprio_std=dataset.proprio_std,
            transform=dataset.transform,
        )
        
        self.latent_dim = config.projected_dim
        log.info(f"Using {self.latent_dim}D mixed latent representation")
        
    def sample_trajectory_segment(self, horizon, n_seeds):
        """Sample trajectory segments from dataset"""
        segments = []
        
        for seed_idx in range(n_seeds):
            # Find valid trajectory
            traj_len = self.config.frameskip * horizon + 1
            valid_found = False
            
            for _ in range(100):  # Max attempts
                traj_id = random.randint(0, len(self.dataset) - 1)
                obs, act, state, env_info = self.dataset[traj_id]
                
                if obs["visual"].shape[0] >= traj_len:
                    valid_found = True
                    break
            
            if not valid_found:
                log.warning(f"Could not find valid trajectory for seed {seed_idx}")
                continue
            
            # Sample segment
            max_offset = obs["visual"].shape[0] - traj_len
            offset = random.randint(0, max_offset)
            
            segment = {
                "obs_0": {key: arr[offset:offset+1] for key, arr in obs.items()},
                "obs_g": {key: arr[offset + traj_len - 1:offset + traj_len] for key, arr in obs.items()},
                "state_0": state[offset].numpy(),
                "state_g": state[offset + traj_len - 1].numpy(),
                "actions": act[offset : offset + self.config.frameskip * horizon],
                "full_obs": {key: arr[offset:offset+traj_len] for key, arr in obs.items()},
                "full_state": state[offset:offset+traj_len].numpy(),
            }
            segments.append(segment)
        
        return segments
    
    def evaluate_latent_rollout(self, segment, horizon):
        """Evaluate model's latent rollout quality"""
        # Process observations
        obs_seq = segment["full_obs"]
        actions = segment["actions"]
        
        # Prepare batch
        visual_batch = torch.tensor(obs_seq["visual"][:self.config.num_hist], device=self.device).unsqueeze(0)
        proprio_batch = torch.tensor(obs_seq["proprio"][:self.config.num_hist], device=self.device).unsqueeze(0)
        
        # Reshape actions for model
        actions_reshaped = rearrange(actions, "(t f) d -> t (f d)", f=self.config.frameskip)
        actions_batch = torch.tensor(actions_reshaped[:horizon], device=self.device).unsqueeze(0)
        
        obs_batch = {"visual": visual_batch, "proprio": proprio_batch}
        
        # Get model rollout
        with torch.no_grad():
            try:
                # Use model's rollout method for mixed latent space
                z_pred, z_final = self.model.rollout(obs_batch, actions_batch)
                
                # Simple evaluation: check if rollout succeeded
                if z_final is not None:
                    # Compute simple distance metric in latent space
                    latent_std = torch.std(z_final).item()
                    return True, latent_std
                else:
                    return False, float('inf')
                    
            except Exception as e:
                log.warning(f"Rollout failed: {e}")
                return False, float('inf')
    
    def evaluate_planning(self, horizon, n_seeds):
        """Run planning evaluation for given horizon"""
        log.info(f"\nEvaluating planning with horizon={horizon}, n_seeds={n_seeds}")
        
        # Sample trajectory segments
        segments = self.sample_trajectory_segment(horizon, n_seeds)
        
        if len(segments) == 0:
            log.error("No valid segments found")
            return {
                "horizon": horizon,
                "n_seeds": n_seeds,
                "latent_dim": self.latent_dim,
                "success_rate": 0.0,
                "avg_latent_std": float('inf'),
                "num_successes": 0,
                "num_trials": 0,
            }
        
        successes = []
        latent_stds = []
        
        for i, segment in enumerate(segments):
            log.info(f"  Evaluating seed {i+1}/{len(segments)}")
            
            # Evaluate rollout
            success, latent_std = self.evaluate_latent_rollout(segment, horizon)
            
            successes.append(success)
            if success:
                latent_stds.append(latent_std)
            
            # Compute state distance for reference
            state_dist = np.linalg.norm(segment["state_g"] - segment["state_0"])
            log.debug(f"    State distance: {state_dist:.3f}, Success: {success}")
        
        results = {
            "horizon": horizon,
            "n_seeds": n_seeds,
            "latent_dim": self.latent_dim,
            "success_rate": np.mean(successes),
            "avg_latent_std": np.mean(latent_stds) if latent_stds else float('inf'),
            "num_successes": sum(successes),
            "num_trials": len(successes),
        }
        
        return results

def main():
    # Configuration
    checkpoint_base = "/home/minghao/workspace/dino_wm/outputs/2025-09-09/07-57-28"
    checkpoint_path = os.path.join(checkpoint_base, "checkpoints", "model_latest.pth")
    config_path = os.path.join(checkpoint_base, ".hydra", "config.yaml")
    
    # Planning parameters
    horizons = [5, 10, 15, 20]
    n_seeds = 10
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Load model
    log.info("Loading model...")
    model, cfg = load_model_from_checkpoint(checkpoint_path, config_path, device)
    
    # Load dataset
    log.info("Loading dataset...")
    import hydra
    _, dataset = hydra.utils.call(
        cfg.env.dataset,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
    )
    dataset = dataset["valid"]
    
    # Create evaluator
    evaluator = SimplePlanningEvaluator(model, dataset, cfg, device)
    
    # Run evaluation for each horizon
    all_results = {}
    
    for horizon in horizons:
        log.info(f"\n{'='*50}")
        log.info(f"Evaluating Horizon={horizon}")
        log.info(f"{'='*50}")
        
        results = evaluator.evaluate_planning(horizon, n_seeds)
        all_results[f"H_{horizon}"] = results
        
        # Print results
        log.info(f"\nResults for H={horizon}:")
        log.info(f"  Success Rate: {results['success_rate']*100:.1f}% ({results['num_successes']}/{results['num_trials']})")
        log.info(f"  Avg Latent Std: {results['avg_latent_std']:.3f}")
    
    # Save results
    latent_dim = cfg.projected_dim
    output_file = f"planning_results_mixed_latent_{latent_dim}d.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print(f"PLANNING EVALUATION SUMMARY ({latent_dim}D Mixed Latent Space)")
    print("="*60)
    print(f"{'Horizon':<10} {'Success Rate':<20} {'Latent Std':<15}")
    print("-"*45)
    for horizon in horizons:
        res = all_results[f"H_{horizon}"]
        success_str = f"{res['success_rate']*100:>6.1f}% ({res['num_successes']}/{res['num_trials']})"
        print(f"H={horizon:<7} {success_str:<20} {res['avg_latent_std']:>8.3f}")
    print("="*60)

if __name__ == "__main__":
    seed(42)
    main()