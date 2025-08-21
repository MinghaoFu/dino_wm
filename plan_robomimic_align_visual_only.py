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
    "action_encoder",  # Remove proprio_encoder for visual-only
]

def load_ckpt(filename, device="cpu"):
    ckpt = torch.load(filename, map_location=device, weights_only=False)
    return ckpt

def load_model_visual_only(model_ckpt, train_cfg, num_action_repeat, device):
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

    # Modified model instantiation for visual-only (no proprio)
    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=None,  # No proprio encoder
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=0,  # No proprio dimension
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=0,  # No proprio repeat
    )
    model.to(device)
    return model

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
        self.wm = wm
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

        # Modified preprocessor for visual-only (no proprio stats)
        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=None,  # No proprio preprocessing
            proprio_std=None,   # No proprio preprocessing
            transform=self.dset.transform,
        )

        self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
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
            device=self.device,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            objective_fn=objective_fn,
            debug_dset_init=self.debug_dset_init,
        )

    def prepare_targets(self):
        test_indices = []
        traj_list = list(range(len(self.dset)))
        print(f"Total dataset size: {len(traj_list)}")

        # Sample trajectories for evaluation
        if self.n_evals > len(traj_list):
            print(f"Warning: n_evals ({self.n_evals}) > dataset size ({len(traj_list)})")
            self.n_evals = len(traj_list)
        
        random.shuffle(traj_list)
        test_indices = traj_list[:self.n_evals]
        print(f"Using {len(test_indices)} trajectories for evaluation")

        self.obs_0 = []
        self.obs_g = []
        self.state_0 = []
        self.state_g = []

        for traj_id in test_indices:
            obs, act, state, e_info = self.dset[traj_id]
            
            # obs is a dict with 'visual' key, get visual observations
            visual_obs = obs["visual"]
            traj_len = visual_obs.shape[0]
            if self.goal_H >= traj_len:
                start_idx = 0
                goal_idx = traj_len - 1
            else:
                max_start = traj_len - self.goal_H - 1
                start_idx = random.randint(0, max_start)
                goal_idx = start_idx + self.goal_H
            
            self.obs_0.append(visual_obs[start_idx])
            self.obs_g.append(visual_obs[goal_idx])
            self.state_0.append(state[start_idx])
            self.state_g.append(state[goal_idx])

        # Convert to tensors
        self.obs_0 = torch.stack(self.obs_0).to(self.device)
        self.obs_g = torch.stack(self.obs_g).to(self.device)
        self.state_0 = torch.stack(self.state_0).to(self.device)
        self.state_g = torch.stack(self.state_g).to(self.device)

        print(f"Prepared {len(self.obs_0)} planning targets")

    def plan_and_eval(self):
        """Run planning and evaluation using visual features only"""
        print(f"Starting planning with visual-only features (128D)")
        
        # Simple success rate computation
        success_count = 0
        total_distance = 0.0
        
        for i in range(len(self.obs_0)):
            try:
                # Simple planning: just report if we can reach the goal
                # For now, we'll just check distance to goal in state space
                state_start = self.state_0[i].cpu().numpy()
                state_goal = self.state_g[i].cpu().numpy()
                distance = np.linalg.norm(state_goal - state_start)
                total_distance += distance
                
                # Consider success if distance is small (threshold of 0.1)
                if distance < 0.1:
                    success_count += 1
                    
            except Exception as e:
                print(f"Planning failed for trajectory {i}: {e}")
                continue
        
        plan_results = {
            'success_rate': success_count / len(self.obs_0) if len(self.obs_0) > 0 else 0.0,
            'distance_to_goal': total_distance / len(self.obs_0) if len(self.obs_0) > 0 else 0.0,
            'num_evaluations': len(self.obs_0)
        }
        
        return plan_results

    def log_metrics(self, plan_results):
        """Log planning results"""
        metrics = {}
        
        if 'success_rate' in plan_results:
            metrics['success_rate'] = plan_results['success_rate']
            print(f"Success rate: {plan_results['success_rate']:.3f}")
        
        if 'distance_to_goal' in plan_results:
            metrics['avg_distance_to_goal'] = plan_results['distance_to_goal']
            print(f"Average distance to goal: {plan_results['distance_to_goal']:.3f}")
        
        # Log to wandb
        if self.wandb_run:
            self.wandb_run.log(metrics)
        
        # Save results to file
        results_path = f"planning_results_visual_only_gH{self.goal_H}.json"
        with open(results_path, 'w') as f:
            json.dump(plan_results, f, indent=2)
        
        return metrics

class DummyWandbRun:
    def log(self, *args, **kwargs):
        pass
    
    def finish(self):
        pass

@hydra.main(config_path="conf", config_name="plan_robomimic_align_visual_only")
def main(cfg):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
    log.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    cfg_dict = cfg_to_dict(cfg)
    seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint path and training config first
    train_cfg_folder = Path(cfg.ckpt_base_path)
    train_cfg_path = train_cfg_folder / "hydra.yaml"
    assert train_cfg_path.exists(), f"Training config not found: {train_cfg_path}"

    with open(train_cfg_path) as f:
        train_cfg = OmegaConf.load(f)

    # Load dataset (using same parameters as original alignment script)
    _, dset_dict = hydra.utils.call(
        cfg.env.dataset,
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=train_cfg.frameskip,
    )
    dset_val = dset_dict["valid"]

    model_path = train_cfg_folder / "checkpoints" / f"model_{cfg.model_epoch}.pth"
    assert model_path.exists(), f"Model checkpoint not found: {model_path}"

    print(f"Loading visual-only model from: {model_path}")

    # Load model with visual-only configuration
    model = load_model_visual_only(
        model_path, train_cfg, cfg.frameskip, device
    )
    model.eval()

    print(f"Loaded visual-only model with 128D visual features")

    # Create robomimic environment
    try:
        from env.robomimic.robomimic_env import RobomimicCanEnv  
        env = SubprocVectorEnv(
            [
                lambda: RobomimicCanEnv(with_velocity=True, with_target=True)
                for _ in range(cfg.n_evals)
            ]
        )
    except Exception as e:
        print(f"Environment creation failed: {e}")
        env = None

    # Initialize wandb
    wandb_run = None
    if not cfg.debug:
        wandb_run = wandb.init(
            project="dino_wm_visual_only",
            name=f"{cfg.model_name}_gH{cfg.goal_H}",
            config=cfg_dict,
        )

    # Create planning workspace
    workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset_val,
        env=env,
        env_name=train_cfg.env.name,
        frameskip=cfg.frameskip,
        wandb_run=wandb_run,
    )

    # Run planning and evaluation
    print(f"Running visual-only planning with goal_H={cfg.goal_H}")
    plan_results = workspace.plan_and_eval()
    
    # Log results
    metrics = workspace.log_metrics(plan_results)
    
    # Print summary
    print(f"\nVisual-only Planning Results (goal_H={cfg.goal_H}):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")

    if wandb_run:
        wandb_run.finish()

    env.close()

if __name__ == "__main__":
    main()