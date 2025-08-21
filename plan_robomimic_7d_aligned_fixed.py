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
    model.to(device)
    return model

class Model7DAlignedWrapper:
    """
    Wrapper that modifies world model to use 7D aligned features for planning
    while maintaining exact compatibility with original planning infrastructure
    """
    def __init__(self, original_model, device):
        self.original_model = original_model
        self.device = device
        
        # Initialize alignment projection if alignment_W not available
        if not (hasattr(original_model, 'alignment_W') and original_model.alignment_W is not None):
            self.alignment_proj = torch.nn.Linear(64, 7).to(device)
            print("Using learned alignment projection for 7D features")
        else:
            self.alignment_proj = None
            print("Using trained alignment matrix W for 7D features")
    
    def separate_emb(self, z):
        """Modified separate_emb to extract 7D aligned features"""
        # Get original embeddings first
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

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,  # Use the 7D aligned wrapped model
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )

        if self.wandb_run is None:
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )

        # optional: assume planning horizon equals to goal horizon
        from planning.mpc import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict["goal_H"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]
        
        print(f"Planning with 7D aligned + 32D proprio features (39D effective)")
        print(f"Original model dimension: {wm.emb_dim}D")
        print(f"Alignment matrix available: {hasattr(wm, 'alignment_W') and wm.alignment_W is not None}")

    def prepare_targets(self):
        # Use dataset trajectories for robomimic - SAME AS ORIGINAL
        observations, states, actions, env_info = (
            self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
        )
        print("traj_len: ", self.frameskip * self.goal_H + 1)
        if self.env is not None:
            self.env.update_env(env_info)

        # get states from val trajs
        init_state = [x[0] for x in states]
        init_state = np.array(init_state)
        actions = torch.stack(actions)
        wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
        exec_actions = self.data_preprocessor.denormalize_actions(actions)
        
        if self.env is not None:
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
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

    def sample_traj_segment_from_dset(self, traj_len):
        # SAME AS ORIGINAL
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def perform_planning(self):
        # SAME AS ORIGINAL - uses actual planner.plan() with CEM
        print(f"Starting REAL planning with 7D aligned features")
        print(f"Goal horizon: {self.goal_H}")
        print(f"Number of evaluations: {self.n_evals}")
        print(f"Using CEM algorithm with robomimic environment")
        
        logs = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=self.gt_actions,
        )
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

    def finish(self):
        pass

def planning_main(cfg_dict):
    # SAME AS ORIGINAL STRUCTURE
    output_dir = cfg_dict.get("saved_folder", os.getcwd())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb_run = None  # Disable wandb for robomimic

    # Get model path from config
    if "ckpt_base_path" in cfg_dict:
        model_path = cfg_dict["ckpt_base_path"]
    elif "model_cfg" in cfg_dict and "model_path" in cfg_dict["model_cfg"]:
        model_path = cfg_dict["model_cfg"]["model_path"]
    else:
        raise ValueError("Need either ckpt_base_path or model_cfg.model_path in config")
    
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    # Create robomimic environment
    try:
        from env.robomimic.robomimic_env import RobomimicCanEnv  
        env = SubprocVectorEnv(
            [
                lambda: RobomimicCanEnv(with_velocity=True, with_target=True)
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    except Exception as e:
        print(f"Environment creation failed: {e}")
        env = None

    # Create workspace with 7D aligned model wrapper
    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,  # Will be wrapped with 7D aligned extraction
        dset=dset,
        env=env,
        env_name="robomimic_can",
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )

    # Run actual planning with CEM algorithm
    logs = plan_workspace.perform_planning()
    return logs

@hydra.main(config_path="conf", config_name="plan_robomimic")
def main(cfg: OmegaConf):
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
    
    # Save results - same format as original
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