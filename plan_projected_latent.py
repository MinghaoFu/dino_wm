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
from tqdm import tqdm
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

def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                timeout_min=500,
                cpus_per_task=2,
                mem_gb=8,
                partition="short",
                slurm_additional_parameters={"requeue": "1"},
            )
            function = submitit.helpers.DelayedSubmission(
                planning_main_in_dir, subdir_path, cfg_dict
            )
            job = executor.submit(function)
            jobs.append(job)
        return jobs

def build_plan_cfg_dicts(
    cfg_dict, goal_sources, goal_Hs, alphas, planners, n_evals
):
    plan_cfg_dicts = []
    for planner_dict in planners:
        for goal_source, goal_H, alpha in product(goal_sources, goal_Hs, alphas):
            plan_cfg_dict = cfg_dict.copy()
            plan_cfg_dict["planner"] = planner_dict
            plan_cfg_dict["goal_source"] = goal_source
            plan_cfg_dict["goal_H"] = goal_H
            plan_cfg_dict["objective"]["alpha"] = alpha
            plan_cfg_dict["n_evals"] = n_evals
            plan_cfg_dicts.append(plan_cfg_dict)
    return plan_cfg_dicts

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

        # Pass projected_dim directly to avoid recursive interpolation
        objective_config = cfg_dict["objective"].copy()
        objective_config["projected_dim"] = cfg_dict.get("projected_dim", 64)
        objective_fn = hydra.utils.call(
            objective_config,
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

    def prepare_targets(self):
        if self.goal_source == "dset":
            # Use dataset trajectories
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
                
        elif self.goal_source == "random_state":
            # sample random states from the dataset distribution
            random_state_indices = random.sample(
                range(len(self.dset)), k=self.n_evals * 2
            )
            random_states = [self.dset[i][2] for i in random_state_indices]
            random_obses = [self.dset[i][0] for i in random_state_indices]

            init_state_indices = random_state_indices[: self.n_evals]
            goal_state_indices = random_state_indices[self.n_evals :]

            self.state_0 = np.array([random_states[i][0] for i in range(self.n_evals)])
            self.state_g = np.array([random_states[i][0] for i in range(self.n_evals, 2 * self.n_evals)])

            if self.env is not None:
                rollout_obses, rollout_states = self.env.rollout(
                    self.eval_seed, self.state_0, np.zeros((self.n_evals, 1, self.action_dim))
                )
                self.obs_0 = {
                    key: np.expand_dims(arr[:, 0], axis=1)
                    for key, arr in rollout_obses.items()
                }

                rollout_obses, rollout_states = self.env.rollout(
                    self.eval_seed, self.state_g, np.zeros((self.n_evals, 1, self.action_dim))
                )
                self.obs_g = {
                    key: np.expand_dims(arr[:, 0], axis=1)
                    for key, arr in rollout_obses.items()
                }
            else:
                # use dataset observations
                self.obs_0 = {
                    key: np.expand_dims(np.array([random_obses[i][key][0] for i in range(self.n_evals)]), axis=1)
                    for key in random_obses[0].keys()
                }
                self.obs_g = {
                    key: np.expand_dims(np.array([random_obses[i][key][0] for i in range(self.n_evals, 2 * self.n_evals)]), axis=1)
                    for key in random_obses[0].keys()
                }
        
        self.gt_actions = None  # not used in random_state goal source

    def sample_traj_segment_from_dset(self, traj_len):
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
        print(f"🔍 Sampling {self.n_evals} evaluation trajectories...")
        for i in tqdm(range(self.n_evals), desc="📊 Sampling trajectories", leave=False):
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
        print(f"🚀 Starting planning with {self.n_evals} evaluation seeds, horizon H={self.goal_H}")
        print(f"📈 Using 80D architecture: 64D projected + 16D action")
        logs = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=self.gt_actions,
        )
        print(f"✅ Planning completed!")
        return logs

def load_ckpt(filename, device="cpu"):
    # IMPORTANT: Pre-load dinov2 module to ensure it's available during checkpoint loading
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    # Pre-load the dinov2 model to make the module available for unpickling
    _ = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", force_reload=False)
    
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

    # MODIFICATION: Use projected_dim for mixed latent representation
    # instead of high-dimensional concatenated features
    projected_dim = getattr(train_cfg, 'projected_dim', 64)  # Default to 64 if not specified
    print(f"Using projected latent dimension: {projected_dim}D (instead of high-dim concatenated features)")

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
        # MODIFICATION: Pass projected_dim to ensure model uses projected representation
        projected_dim=projected_dim,
    )
    model.to(device)
    return model

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
    output_dir = cfg_dict.get("saved_folder", os.getcwd())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=cfg_dict.get("wandb_project", "dino_wm_planning"),
            name=cfg_dict.get("wandb_name", f"planning_{cfg_dict.get('goal_H', 'unknown')}H"),
            config=cfg_dict,
            reinit=True,
        )
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = ckpt_base_path
    
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

    # use dummy vector env for wall and deformable envs
    if model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )

    logs = plan_workspace.perform_planning()
    return logs


@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    # Use wandb_logging from config if specified, default to False
    if "wandb_logging" not in cfg_dict:
        cfg_dict["wandb_logging"] = False
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()