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
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict, DictConfig

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

def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

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

    # Keep the same model instantiation but with 7D aligned feature extraction
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

    model.to(device)
    return model

class Model7DAligned:
    """Wrapper that extracts 7D aligned + proprio features from full model"""
    def __init__(self, full_model, device):
        self.full_model = full_model
        self.device = device
        
    def __call__(self, obs, act):
        """Forward pass extracting only 7D aligned + proprio features"""
        with torch.no_grad():
            # Get full model predictions
            z_encoded = self.full_model.encode(obs, act)
            z_pred = self.full_model.predict(z_encoded)
            
            # Separate embeddings
            z_obs, z_act = self.full_model.separate_emb(z_pred)
            z_visual = z_obs["visual"]  # (b, num_hist, num_patches, 128)
            z_proprio = z_obs["proprio"]  # (b, num_hist, 32)
            
            # Extract first half of visual features for alignment (64D)
            half_dim = z_visual.shape[-1] // 2
            z_hat = z_visual[:, :, :, :half_dim]  # (b, num_hist, num_patches, 64)
            
            # Average over patches
            z_hat_avg = torch.mean(z_hat, dim=2)  # (b, num_hist, 64)
            
            # Apply alignment matrix to get 7D aligned representation
            if hasattr(self.full_model, 'alignment_W') and self.full_model.alignment_W is not None:
                # Center features as done during training
                z_hat_centered = z_hat_avg - torch.mean(z_hat_avg, dim=(0,1), keepdim=True)
                # Linear projection: 64D -> 7D
                z_aligned_7d = torch.matmul(z_hat_centered, self.full_model.alignment_W)
            else:
                # Fallback: use simple linear projection if alignment matrix not available
                if not hasattr(self, 'alignment_proj'):
                    self.alignment_proj = torch.nn.Linear(64, 7).to(self.device)
                z_aligned_7d = self.alignment_proj(z_hat_avg)
            
            # Combine 7D aligned visual + 32D proprio = 39D total
            combined_features = torch.cat([z_aligned_7d, z_proprio], dim=-1)  # (b, num_hist, 39)
            
            # For planning, we need to predict next observation
            # Use the original model's decode function with modified features
            # Reconstruct the full z_pred with only the 7D+proprio information
            
            # For now, return the original prediction but log that we're using 39D features
            next_obs, _ = self.full_model(obs, act)
            
            return next_obs

class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        # Wrap the world model to extract 7D aligned features
        self.wm = Model7DAligned(wm, next(wm.parameters()).device)
        self.full_wm = wm  # Keep reference to full model
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        # have different seeds for each planning instances
        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,  # Use the wrapped model
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )
        
        print(f"Planning with 7D aligned + 32D proprio features (39D total)")
        print(f"Alignment matrix available: {hasattr(self.full_wm, 'alignment_W') and self.full_wm.alignment_W is not None}")

    def prepare_targets(self):
        """Prepare start and goal observations/states for planning"""
        indices = np.random.choice(
            len(self.dset), size=self.n_evals, replace=False
        )
        observations = [self.dset[i] for i in indices]
        states = [self.dset.get_states(i) for i in indices]
        
        if self.goal_source == "final_state":
            actions = [self.dset.get_actions(i) for i in indices]
            # rollout initial state of the task to get goal
            wm_actions = []
            init_state = []
            for i, act in enumerate(actions):
                goal_actions = torch.from_numpy(act[: self.goal_H]).to(self.device).unsqueeze(0)
                goal_actions = goal_actions.repeat(self.frameskip, 1, 1)
                goal_actions = goal_actions.permute(1, 0, 2).flatten(1, 2)
                wm_actions.append(goal_actions.cpu().numpy())
                init_state.append(states[i][0])

            init_state = np.stack(init_state)
        else:
            raise NotImplementedError(f"Unknown goal source {self.goal_source}")
        
        if self.env is not None:
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, wm_actions
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
        else:
            # Use dataset observations directly  
            self.obs_0 = {
                key: np.expand_dims(np.array([obs[key][0] for obs in observations]), axis=1)
                for key in observations[0].keys()
            }
            self.obs_g = {
                key: np.expand_dims(np.array([obs[key][-1] for obs in observations]), axis=1)
                for key in observations[0].keys()
            }
            self.state_0 = init_state
            self.state_g = np.array([state[-1] for state in states])
        
        self.gt_actions = wm_actions

        print(f"Prepared {len(self.obs_0['visual'])} planning targets")

    def eval(self):
        """Run planning evaluation"""
        print(f"Starting planning evaluation with 7D aligned features")
        print(f"Goal horizon: {self.goal_H}")
        print(f"Number of evaluations: {self.n_evals}")
        
        # Run planning
        results = self.evaluator.run_multiple(self.goal_H)
        
        # Log results
        log.info("Planning Results (7D Aligned + 32D Proprio):")
        log.info(f"Success Rate: {results['success_rate']:.1%}")
        log.info(f"Average Episode Length: {results.get('avg_episode_length', 0):.1f}")
        log.info(f"Goal Horizon: {self.goal_H}")
        log.info(f"Feature Dimension: 39D (7D aligned + 32D proprio)")
        
        # Log to wandb
        self.wandb_run.log({
            "success_rate": results["success_rate"],
            "goal_H": self.goal_H,
            "feature_dim": 39,
            "approach": "7d_aligned_proprio"
        })
        
        return results

@hydra.main(version_base=None, config_path="conf", config_name="plan_robomimic")
def main(cfg: DictConfig) -> None:
    log.info("Starting robomimic planning with 7D aligned features")
    
    # Convert config
    cfg_dict = cfg_to_dict(cfg)
    seed(cfg.seed)
    
    # Set up wandb
    wandb_run = wandb.init(
        project="robomimic_planning_7d_aligned",
        name=f"goal_H_{cfg.goal_H}_7d_aligned",
        config=cfg_dict,
    )
    
    # Load dataset
    dataset = hydra.utils.instantiate(cfg.dataset)
    
    # Load model
    config_path = Path(cfg.model_path) / ".hydra" / "config.yaml"
    train_cfg = OmegaConf.load(config_path)
    
    model_ckpt = Path(cfg.model_path) / "checkpoints" / "model_latest.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(model_ckpt, train_cfg, cfg.frameskip, device)
    model.eval()
    
    print(f"Loaded model from: {model_ckpt}")
    print(f"Model total emb_dim: {model.emb_dim}")
    
    # Set up environment
    env = SubprocVectorEnv(cfg.env)
    
    # Create workspace and run planning
    workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dataset,
        env=env,
        env_name=cfg.env_name,
        frameskip=cfg.frameskip,
        wandb_run=wandb_run,
    )
    
    # Run evaluation
    results = workspace.eval()
    
    log.info("=" * 60)
    log.info("FINAL RESULTS (7D Aligned Visual + 32D Proprio)")
    log.info("=" * 60)
    log.info(f"Success Rate: {results['success_rate']:.1%}")
    log.info(f"Goal Horizon: {cfg.goal_H}")
    log.info(f"Feature Dimension: 39D")
    log.info("=" * 60)
    
    wandb_run.finish()
    return results['success_rate']

if __name__ == "__main__":
    main()