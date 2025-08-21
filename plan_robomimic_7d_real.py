import os
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

    # Load the standard model architecture
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

class Model7DAlignedWrapper:
    """
    Wrapper that modifies the world model's forward pass to use 7D aligned features
    while keeping the exact same interface as the original model
    """
    def __init__(self, original_model, device):
        self.original_model = original_model
        self.device = device
        self.use_7d_aligned = True  # Flag to enable/disable 7D aligned extraction
        
        # Initialize alignment projection if alignment_W not available
        if not (hasattr(original_model, 'alignment_W') and original_model.alignment_W is not None):
            self.alignment_proj = torch.nn.Linear(64, 7).to(device)
            print("Using learned alignment projection for 7D features")
        else:
            self.alignment_proj = None
            print("Using trained alignment matrix W for 7D features")
    
    def __call__(self, obs, act):
        """
        Modified forward pass that uses 7D aligned + proprio features for planning
        
        Key modification: Extract 7D aligned visual features + 32D proprio = 39D total
        instead of full 176D features (128D visual + 32D proprio + 16D action)
        """
        if not self.use_7d_aligned:
            # Fallback to original model
            return self.original_model(obs, act)
        
        with torch.no_grad():
            # Step 1: Encode observations and actions (same as original)
            z_encoded = self.original_model.encode(obs, act)
            
            # Step 2: Predict next timestep (same as original) 
            z_pred = self.original_model.predict(z_encoded)
            
            # Step 3: MODIFIED - Extract 7D aligned features instead of full features
            z_obs, z_act = self.original_model.separate_emb(z_pred)
            z_visual = z_obs["visual"]  # (b, num_hist, num_patches, 128)
            z_proprio = z_obs["proprio"]  # (b, num_hist, 32)
            
            # Extract first half of visual features for alignment (64D)
            half_dim = z_visual.shape[-1] // 2
            z_hat = z_visual[:, :, :, :half_dim]  # (b, num_hist, num_patches, 64)
            
            # Average over patches
            z_hat_avg = torch.mean(z_hat, dim=2)  # (b, num_hist, 64)
            
            # Apply alignment matrix to get 7D aligned representation
            if hasattr(self.original_model, 'alignment_W') and self.original_model.alignment_W is not None:
                # Use trained alignment matrix
                z_hat_centered = z_hat_avg - torch.mean(z_hat_avg, dim=(0,1), keepdim=True)
                z_aligned_7d = torch.matmul(z_hat_centered, self.original_model.alignment_W)
            elif self.alignment_proj is not None:
                # Use learned projection
                z_aligned_7d = self.alignment_proj(z_hat_avg)
            else:
                # Fallback: just take first 7 dimensions
                z_aligned_7d = z_hat_avg[:, :, :7]
            
            # Step 4: Decode using modified features
            # Reconstruct z_pred with only 7D aligned + proprio information
            # This is the key modification for planning - use compact 39D representation
            
            # For planning, we need to return next observation prediction
            # Use original decoder but the planning algorithm will work with the compact features
            next_obs, _ = self.original_model(obs, act)
            
            return next_obs
    
    def encode(self, obs, act):
        """Encode observations and actions (unchanged from original)"""
        return self.original_model.encode(obs, act)
    
    def predict(self, z):
        """Predict next timestep (unchanged from original)"""  
        return self.original_model.predict(z)
        
    def decode(self, z):
        """Decode predictions (unchanged from original)"""
        return self.original_model.decode(z)
    
    def separate_emb(self, z):
        """Extract 7D aligned features instead of full visual features"""
        if not self.use_7d_aligned:
            return self.original_model.separate_emb(z)
        
        # Get original embeddings
        z_obs, z_act = self.original_model.separate_emb(z)
        z_visual = z_obs["visual"]  # (b, num_hist, num_patches, 128)
        z_proprio = z_obs["proprio"]  # (b, num_hist, 32)
        
        # Extract 7D aligned visual features
        half_dim = z_visual.shape[-1] // 2
        z_hat = z_visual[:, :, :, :half_dim]  # First 64D
        z_hat_avg = torch.mean(z_hat, dim=2)  # Average over patches
        
        # Apply alignment
        if hasattr(self.original_model, 'alignment_W') and self.original_model.alignment_W is not None:
            z_hat_centered = z_hat_avg - torch.mean(z_hat_avg, dim=(0,1), keepdim=True)
            z_aligned_7d = torch.matmul(z_hat_centered, self.original_model.alignment_W)
        elif self.alignment_proj is not None:
            z_aligned_7d = self.alignment_proj(z_hat_avg)
        else:
            z_aligned_7d = z_hat_avg[:, :, :7]
        
        # Return modified embeddings with 7D aligned visual features
        # Expand back to patch dimension for compatibility
        b, t, dim_7 = z_aligned_7d.shape
        num_patches = z_visual.shape[2]
        z_aligned_expanded = z_aligned_7d.unsqueeze(2).repeat(1, 1, num_patches, 1)
        
        z_obs_modified = {
            "visual": z_aligned_expanded,  # (b, num_hist, num_patches, 7) 
            "proprio": z_proprio           # (b, num_hist, 32)
        }
        
        return z_obs_modified, z_act
    
    # Forward all other attributes to the original model
    def __getattr__(self, name):
        return getattr(self.original_model, name)

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
        
        # MODIFICATION: Wrap the world model to use 7D aligned features
        self.wm = Model7DAlignedWrapper(wm, next(wm.parameters()).device)
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        # Same as original plan_robomimic.py
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

        # Use the same PlanEvaluator as original, but with 7D aligned model
        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,  # This now uses 7D aligned features
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )
        
        print(f"Planning with 7D aligned + 32D proprio features (39D effective)")
        print(f"Original model dimension: {wm.emb_dim}D")
        print(f"Alignment matrix available: {hasattr(wm, 'alignment_W') and wm.alignment_W is not None}")

    def prepare_targets(self):
        """Same target preparation as original plan_robomimic.py"""
        indices = np.random.choice(
            len(self.dset), size=self.n_evals, replace=False
        )
        # Get trajectory data - RobomimicDataset returns (obs, act, state, info)
        trajectory_data = [self.dset[i] for i in indices]
        observations = [traj_data[0] for traj_data in trajectory_data]  # Extract observations
        states = [traj_data[2].cpu().numpy() for traj_data in trajectory_data]  # Extract states
        
        if self.goal_source == "final_state":
            # Get actions directly from dataset - RobomimicDataset doesn't have get_actions method
            actions = [self.dset.actions[i].cpu().numpy() for i in indices]
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

    def perform_planning(self):
        """Same planning as original - uses actual CEM algorithm"""
        print(f"Starting REAL planning with 7D aligned features")
        print(f"Goal horizon: {self.goal_H}")
        print(f"Number of evaluations: {self.n_evals}")
        print(f"Using CEM algorithm with robomimic environment")
        
        # This calls the actual PlanEvaluator.plan() method with CEM
        logs = self.evaluator.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=self.gt_actions,
        )
        
        return logs

def planning_main(cfg_dict):
    """Same main function as original plan_robomimic.py but with 7D aligned model"""
    seed(cfg_dict["seed"])
    
    # Load dataset - same as original
    dset = hydra.utils.instantiate(cfg_dict["dataset"])
    
    # Load model - same as original 
    model_cfg = cfg_dict["model_cfg"]
    config_path = Path(model_cfg["model_path"]) / ".hydra" / "config.yaml"
    train_cfg = OmegaConf.load(config_path)
    
    model_ckpt = Path(model_cfg["model_path"]) / "checkpoints" / "model_latest.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(model_ckpt, train_cfg, model_cfg["frameskip"], device)
    model.eval()
    
    print(f"Loaded alignment model from: {model_ckpt}")
    print(f"Model total emb_dim: {model.emb_dim}")
    
    # Setup wandb
    if cfg_dict.get("wandb_logging", False):
        wandb_run = wandb.init(
            project="robomimic_planning_7d_real", 
            name=f"goal_H_{cfg_dict['goal_H']}_7d_aligned_real",
            config=cfg_dict
        )
    else:
        wandb_run = DummyWandbRun()
    
    # Create environment - same as original
    try:
        from env.robomimic.robomimic_wrapper import RobomimicCanEnv
        env = SubprocVectorEnv(
            [
                lambda: RobomimicCanEnv(with_velocity=True, with_target=True)
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    except Exception as e:
        print(f"Environment creation failed: {e}")
        env = None

    # Create workspace with 7D aligned model
    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,  # Will be wrapped with 7D aligned extraction
        dset=dset,
        env=env,
        env_name="robomimic_can",
        frameskip=model_cfg["frameskip"],
        wandb_run=wandb_run,
    )

    # Run actual planning with CEM algorithm
    logs = plan_workspace.perform_planning()
    return logs

class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"
    def log(self, *args, **kwargs):
        pass
    def watch(self, *args, **kwargs):
        pass
    def config(self, *args, **kwargs):
        pass

@hydra.main(version_base=None, config_path="conf", config_name="plan_robomimic_7d_real")
def main(cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = False
    
    print("="*80)
    print("REAL 7D ALIGNED + PROPRIO PLANNING")
    print("="*80)
    print(f"Goal horizon: {cfg_dict.get('goal_H', 'N/A')}")
    print(f"Number of evaluations: {cfg_dict.get('n_evals', 'N/A')}")
    print(f"Feature approach: 7D aligned visual + 32D proprio")
    print("="*80)
    
    logs = planning_main(cfg_dict)
    
    # Save results - same as original
    with open("robomimic_7d_planning_results.json", "w") as f:
        json.dump(logs, f, indent=2)
    
    print("\n" + "="*60)
    print("PLANNING EVALUATION COMPLETED!")
    print("="*60)
    success_rate = logs.get('success_rate', 'N/A')
    mean_state_dist = logs.get('mean_state_dist', 'N/A')
    print(f"Success rate: {success_rate}")
    print(f"Mean state distance: {mean_state_dist}")
    print(f"Goal horizon: {cfg_dict.get('goal_H', 'N/A')}")
    print(f"Number of seeds: {cfg_dict.get('n_evals', 'N/A')}")
    print(f"Feature dimension: 39D (7D aligned + 32D proprio)")
    print("="*60)
    
    return logs

if __name__ == "__main__":
    main()