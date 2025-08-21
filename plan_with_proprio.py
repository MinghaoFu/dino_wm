"""
Proprio-only planning for DINO World Model
This version uses only proprioceptive information instead of visual features
"""

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
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from utils import cfg_to_dict, seed

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "proprio_encoder", 
    "action_encoder",
    "predictor",
]

class ProprioOnlyPlanWorkspace:
    def __init__(
        self,
        cfg_dict,
        encoder,
        proprio_encoder,
        action_encoder, 
        predictor,
        dset,
        wandb_run=None,
    ):
        self.cfg_dict = cfg_dict
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.dset = dset
        self.wandb_run = wandb_run or DummyWandbRun()
        
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.frameskip = 5  # From training config
        
        # Initialize preprocessor with dataset statistics
        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=None,  # No transform needed for planning
        )
        # Add action_dim property
        self.data_preprocessor.action_dim = len(self.dset.action_mean) * self.frameskip
        
        # Setup evaluation seeds
        self.eval_seed = [1 + i * 99 for i in range(self.n_evals)]
        log.info(f"eval_seed:  {self.eval_seed}")
        
        # Prepare planning targets
        self.prepare_targets()
        
        # Initialize MPC planner with CEM sub-planner
        from planning.proprio_cem import ProprioOnlyCEMPlanner
        from planning.proprio_mpc import ProprioOnlyMPCPlanner, ProprioEnvironmentInterface
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create CEM sub-planner
        cem_planner = ProprioOnlyCEMPlanner(
            horizon=3,  # Shorter horizon for faster testing
            topk=10,    # Fewer top candidates
            num_samples=30,  # Fewer samples for faster execution
            var_scale=1.0,
            opt_steps=3,  # Fewer optimization steps
            eval_every=1,
            preprocessor=self.data_preprocessor,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            device=device
        )
        
        # Create MPC planner
        self.planner = ProprioOnlyMPCPlanner(
            sub_planner=cem_planner,
            max_iter=5,  # Fewer MPC iterations for testing
            n_taken_actions=1,  # Execute 1 action per MPC step
            device=device
        )
        
        # Create environment interface for state transitions
        self.env_interface = ProprioEnvironmentInterface(
            use_world_model=True,
            world_model_components={
                'encoder': self.encoder,
                'proprio_encoder': self.proprio_encoder,
                'action_encoder': self.action_encoder,
                'predictor': self.predictor
            }
        )
        
        # Output files
        output_dir = cfg_dict["saved_folder"]
        self.log_filename = os.path.join(output_dir, "logs.json")
        self.targets_filename = os.path.join(output_dir, "plan_targets.pkl")
        
        # Save planning targets
        with open(self.targets_filename, "wb") as f:
            pickle.dump({
                'proprio_0': self.proprio_0,
                'proprio_g': self.proprio_g,
                'state_0': self.state_0,
                'state_g': self.state_g,
                'gt_actions': self.gt_actions
            }, f)
        print(f"Dumped plan targets to {self.targets_filename}")

    def prepare_targets(self):
        """Prepare planning targets using only proprioceptive information"""
        states = []
        actions = []
        proprios = []
        
        # Sample trajectory segments from dataset
        observations, states, actions, env_info = self.sample_traj_segment_from_dset(
            traj_len=self.frameskip * self.goal_H + 1
        )
        
        # Extract initial and goal states/proprio
        init_state = [x[0] for x in states]
        goal_state = [x[-1] for x in states] 
        init_proprio = [obs['proprio'][0] for obs in observations]
        goal_proprio = [obs['proprio'][-1] for obs in observations]
        
        self.state_0 = np.array(init_state)  # (n_evals, state_dim)
        self.state_g = np.array(goal_state)  # (n_evals, state_dim)
        
        # Convert proprio to numpy arrays with consistent shapes
        self.proprio_0 = np.stack([np.array(p) for p in init_proprio])  # (n_evals, proprio_dim)
        self.proprio_g = np.stack([np.array(p) for p in goal_proprio])  # (n_evals, proprio_dim)
        
        # Process actions for planning
        actions = torch.stack(actions)
        if self.goal_source == "random_action":
            actions = torch.randn_like(actions)
        wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
        
        self.gt_actions = wm_actions

    def sample_traj_segment_from_dset(self, traj_len):
        """Sample trajectory segments from dataset"""
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

        # Sample trajectory segments
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:
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
        """Perform MPC planning using only proprioceptive information"""
        # Plan using MPC with proprio representations
        actions, action_len = self.planner.plan(
            proprio_0=self.proprio_0,
            proprio_g=self.proprio_g,
            actions=self.gt_actions if hasattr(self, 'gt_actions') else None,
            env_interface=self.env_interface
        )
        
        # Evaluate planning results
        logs = self.eval_proprio_planning(actions.detach(), action_len)
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs

    def eval_proprio_planning(self, actions, action_len):
        """Evaluate planning results using proprioceptive information"""
        # Use the planner's evaluation method for consistency
        device = actions.device
        
        proprio_0_tensor = torch.tensor(self.proprio_0, device=device)
        proprio_g_tensor = torch.tensor(self.proprio_g, device=device)
        
        # Evaluate final costs
        final_costs = self.planner.evaluate_actions(proprio_0_tensor, proprio_g_tensor, actions)
        
        # Convert costs to distances and success
        state_dists = final_costs.cpu().numpy()
        successes = state_dists < 0.5  # Success threshold
        
        success_rate = np.mean(successes)
        mean_state_dist = np.mean(state_dists)
        
        print(f"Success rate: {success_rate:.3f}")
        print(f"Mean state distance: {mean_state_dist:.3f}")
        
        return {
            'success_rate': success_rate,
            'mean_state_dist': mean_state_dist,
            'successes': successes.tolist() if hasattr(successes, 'tolist') else successes,
            'state_dists': state_dists.tolist() if hasattr(state_dists, 'tolist') else state_dists
        }


def load_ckpt(snapshot_path, device):
    """Load checkpoint with only required components"""
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_proprio_model(model_ckpt, train_cfg, device):
    """Load model components for proprio-only planning"""
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    # Load encoder
    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
        result["encoder"].to(device)

    if "proprio_encoder" not in result:
        raise ValueError("Proprio encoder not found in model checkpoint")
    if "action_encoder" not in result:
        raise ValueError("Action encoder not found in model checkpoint")
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    return result["encoder"], result["proprio_encoder"], result["action_encoder"], result["predictor"]


def planning_main(cfg_dict):
    """Main planning function for proprio-only planning"""
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_proprio_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"
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

    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    encoder, proprio_encoder, action_encoder, predictor = load_proprio_model(model_ckpt, model_cfg, device=device)

    plan_workspace = ProprioOnlyPlanWorkspace(
        cfg_dict=cfg_dict,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        dset=dset,
        wandb_run=wandb_run,
    )

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


@hydra.main(config_path="conf", config_name="plan_robomimic")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = True
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()