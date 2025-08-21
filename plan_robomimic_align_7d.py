import os
import gym
import json
import hydra
import random
import torch
import pickle
import wandb
import logging
import warnings
from pathlib import Path
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor", 
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

def load_ckpt(filename, device="cpu"):
    ckpt = torch.load(filename, map_location=device, weights_only=False)
    return ckpt

def load_model_7d_aligned(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")
    
    # Keep proprio encoder for 7D aligned + proprio
    if "proprio_encoder" not in result:
        result["proprio_encoder"] = hydra.utils.instantiate(
            train_cfg.proprio_encoder,
            in_chans=7,
            emb_dim=train_cfg.proprio_emb_dim,
        )

    if "action_encoder" not in result:
        result["action_encoder"] = hydra.utils.instantiate(
            train_cfg.action_encoder,
            in_chans=10,
            emb_dim=train_cfg.action_emb_dim,
        )

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path, weights_only=False)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path, weights_only=False)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    # Modified model instantiation for 7D aligned + proprio
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
        num_proprio_repeat=1,
    )

    return model

class Model:
    """Model wrapper for 7D aligned visual + proprio planning"""
    def __init__(self, cfg, preprocessor):
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.load_model()
        
        # Set planning parameters
        self.frameskip = cfg.get('frameskip', 5)
        self.img_size = cfg.get('img_size', 224)
        
        log.info("Model loaded for 7D aligned visual + proprio planning")
        log.info(f"Alignment matrix available: {hasattr(self.model, 'alignment_W') and self.model.alignment_W is not None}")

    def load_model(self):
        """Load the trained model"""
        model_path = Path(self.cfg.model_path)
        
        # Load training config
        config_path = model_path / ".hydra" / "config.yaml"
        train_cfg = OmegaConf.load(config_path)
        
        # Load model checkpoint
        model_ckpt = model_path / "checkpoints" / "model_latest.pth"
        
        # Load model with 7D aligned + proprio configuration
        self.model = load_model_7d_aligned(model_ckpt, train_cfg, 1, self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        log.info(f"Loaded model from {model_ckpt}")

    def extract_7d_aligned_features(self, obs, act):
        """Extract 7D aligned visual + proprio features"""
        with torch.no_grad():
            # Forward pass through model
            obs_torch = {k: torch.from_numpy(v).float().to(self.device) for k, v in obs.items()}
            act_torch = torch.from_numpy(act).float().to(self.device)
            
            # Get model predictions
            z_pred = self.model.encode_and_predict(obs_torch, act_torch)
            
            # Separate embeddings
            z_obs, z_act = self.model.separate_emb(z_pred)
            z_visual = z_obs["visual"]  # (b, num_hist, num_patches, 128)
            z_proprio = z_obs["proprio"]  # (b, num_hist, 32)
            
            # Extract first half of visual features for alignment (64D)
            half_dim = z_visual.shape[-1] // 2
            z_hat = z_visual[:, :, :, :half_dim]  # (b, num_hist, num_patches, 64)
            
            # Average over patches
            z_hat_avg = torch.mean(z_hat, dim=2)  # (b, num_hist, 64)
            
            # Apply alignment matrix to get 7D aligned representation
            if hasattr(self.model, 'alignment_W') and self.model.alignment_W is not None:
                # Center features as done during training
                z_hat_centered = z_hat_avg - torch.mean(z_hat_avg, dim=(0,1), keepdim=True)
                # Linear projection: 64D -> 7D
                z_aligned_7d = torch.matmul(z_hat_centered, self.model.alignment_W)
            else:
                # Fallback: use simple linear projection
                if not hasattr(self, 'alignment_proj'):
                    self.alignment_proj = torch.nn.Linear(64, 7).to(self.device)
                z_aligned_7d = self.alignment_proj(z_hat_avg)
            
            # Combine 7D aligned visual + 32D proprio = 39D total
            combined_features = torch.cat([z_aligned_7d, z_proprio], dim=-1)  # (b, num_hist, 39)
            
            return combined_features

    def __call__(self, obs, act):
        """Forward pass for planning"""
        # Extract 7D aligned + proprio features
        features = self.extract_7d_aligned_features(obs, act)
        
        # Forward through model for next observation prediction
        with torch.no_grad():
            obs_torch = {k: torch.from_numpy(v).float().to(self.device) for k, v in obs.items()}
            act_torch = torch.from_numpy(act).float().to(self.device)
            
            # Get next observation prediction
            next_obs, _ = self.model(obs_torch, act_torch)
            
            # Convert back to numpy
            next_obs_np = {}
            for k, v in next_obs.items():
                if isinstance(v, torch.Tensor):
                    next_obs_np[k] = v.cpu().numpy()
                else:
                    next_obs_np[k] = v
                    
            return next_obs_np

def main(cfg):
    log.info("Starting 7D aligned visual + proprio planning evaluation")
    log.info(f"Model path: {cfg.model_path}")
    log.info(f"Goal horizon: {cfg.get('goal_H', 3)}")
    
    # Set random seeds
    seed(cfg.seed)
    
    # Initialize model (no preprocessor needed for this approach)
    model = Model(cfg, None)
    
    # Initialize planning evaluator
    evaluator = PlanEvaluator(
        model=model,
        env_cfg=cfg.env,
        goal_H=cfg.get('goal_H', 3),
        planning_cfg=cfg.planning,
        device=model.device,
        num_envs=cfg.get('num_envs', 1),
        max_episode_length=cfg.get('max_episode_length', 400),
        frameskip=cfg.get('frameskip', 5),
        normalize_action=cfg.get('normalize_action', True)
    )
    
    # Run evaluation
    results = evaluator.evaluate(num_episodes=cfg.get('num_episodes', 50))
    
    log.info("=" * 60)
    log.info("PLANNING RESULTS (7D Aligned Visual + 32D Proprio)")
    log.info("=" * 60)
    log.info(f"Success Rate: {results['success_rate']:.1%}")
    log.info(f"Average Episode Length: {results['avg_episode_length']:.1f}")
    log.info(f"Average Episode Reward: {results['avg_episode_reward']:.3f}")
    log.info(f"Goal Horizon: {cfg.get('goal_H', 3)}")
    log.info(f"Feature Dimension: 7D aligned + 32D proprio = 39D total")
    log.info("=" * 60)
    
    return results['success_rate']

if __name__ == "__main__":
    import sys
    
    # Parse goal_H from command line
    goal_H = 3
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith('goal_H='):
                goal_H = int(arg.split('=')[1])
    
    # Load config
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'model_path': '/mnt/data1/minghao/robomimic/checkpoints/outputs/robomimic_can_align/outputs/2025-08-13/16-32-15',
        'goal_H': goal_H,
        'num_episodes': 50,
        'num_envs': 1, 
        'max_episode_length': 400,
        'seed': 42,
        'frameskip': 5,
        'normalize_action': True,
        'img_size': 224,
        'env': OmegaConf.create({
            '_target_': 'env.robomimic.robomimic_wrapper.RobomimicCanEnv',
            'env_name': 'PickPlaceCan',
            'camera_names': ['agentview'],
            'camera_height': 224,
            'camera_width': 224,
            'render': False
        }),
        'planning': OmegaConf.create({
            '_target_': 'planning.cem.CEM',
            'horizon': goal_H,
            'num_iterations': 5,
            'num_samples': 400,
            'num_elites': 40,
            'alpha': 0.1,
            'device': 'cuda',
            'action_dim': 10,
            'action_bounds': [-1.0, 1.0]
        })
    })
    
    main(cfg)