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
        # Use the dino_no_proj encoder for full 384-dim DINO embeddings
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
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
    
    # Override forward method to skip alignment logic (same as training)
    def forward_no_alignment(obs, act, state=None):
        loss = 0
        loss_components = {}
        z = model.encode(obs, act)
        z_src = z[:, : model.num_hist, :, :]
        z_tgt = z[:, model.num_pred :, :, :]
        visual_src = obs['visual'][:, : model.num_hist, ...]
        visual_tgt = obs['visual'][:, model.num_pred :, ...]

        if model.predictor is not None:
            z_pred = model.predict(z_src)
            if model.decoder is not None:
                obs_pred, diff_pred = model.decode(
                    z_pred.detach()
                )
                visual_pred = obs_pred['visual']
                recon_loss_pred = model.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + model.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio dims (exclude action dims)
            if model.concat_dim == 0:
                z_visual_loss = model.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                z_proprio_loss = model.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                z_loss = model.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            elif model.concat_dim == 1:
                z_visual_loss = model.emb_criterion(
                    z_pred[:, :, :, :-(model.proprio_dim + model.action_dim)], 
                    z_tgt[:, :, :, :-(model.proprio_dim + model.action_dim)].detach()
                )
                z_proprio_loss = model.emb_criterion(
                    z_pred[:, :, :, -(model.proprio_dim + model.action_dim): -model.action_dim], 
                    z_tgt[:, :, :, -(model.proprio_dim + model.action_dim): -model.action_dim].detach()
                )
                z_loss = model.emb_criterion(
                    z_pred[:, :, :, :-model.action_dim], 
                    z_tgt[:, :, :, :-model.action_dim].detach()
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
            
            # SKIP alignment logic completely
        else:
            visual_pred = None
            z_pred = None

        if model.decoder is not None:
            obs_reconstructed, diff_reconstructed = model.decode(z)
            visual_reconstructed = obs_reconstructed['visual']
            recon_loss = model.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss = recon_loss + model.decoder_latent_loss_weight * diff_reconstructed
            loss = loss + decoder_loss
            loss_components["decoder_recon_loss"] = recon_loss
            loss_components["decoder_vq_loss"] = diff_reconstructed
            loss_components["decoder_loss"] = decoder_loss
        else:
            visual_reconstructed = None

        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components
        
    # Bind the new forward method to the model
    model.forward = forward_no_alignment
    
    model.to(device)
    return model

@hydra.main(config_path="conf", config_name="plan_robomimic_no_align")
def planning_main(cfg: OmegaConf):
    print(f"Config: {cfg}")
    
    # Load model config from checkpoint directory  
    model_path = Path(cfg.ckpt_base_path)
    hydra_cfg_path = model_path / ".hydra" / "config.yaml"
    
    if not hydra_cfg_path.exists():
        hydra_cfg_path = model_path / "hydra.yaml"
        
    if not hydra_cfg_path.exists():
        raise FileNotFoundError(f"Could not find config at {hydra_cfg_path}")
        
    model_cfg = OmegaConf.load(hydra_cfg_path)
    
    # Determine checkpoint path
    if cfg.model_epoch == "latest":
        model_ckpt = model_path / "checkpoints" / "model_latest.pth"
    else:
        model_ckpt = model_path / "checkpoints" / f"model_{cfg.model_epoch}.pth"
        
    if not model_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_ckpt}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset for goal selection
    print(f"Loading dataset from {model_cfg.env.dataset.data_path}")
    datasets, traj_dsets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    
    # Load model
    num_action_repeat = model_cfg.get("num_action_repeat", 1)
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Encoder embedding dim: {model.encoder.emb_dim}")
    print(f"Total model embedding dim: {model.encoder.emb_dim + model.proprio_dim + model.action_dim}")
    
    # Initialize environment - use robomimic environment directly
    print("Initializing robomimic environment...")
    try:
        # Try different ways to access the environment config
        if hasattr(model_cfg.env, 'env'):
            env_fn = lambda: hydra.utils.call(model_cfg.env.env)
        elif hasattr(model_cfg.env.dataset, 'env_cfg'):
            env_fn = lambda: hydra.utils.call(model_cfg.env.dataset.env_cfg.env)
        else:
            # Fallback: create robomimic environment directly
            from env.robomimic.robomimic_wrapper import RobomimicWrapper
            env_fn = lambda: RobomimicWrapper(
                task_name="CanPH",
                data_path="/mnt/data1/minghao/robomimic/can/ph_converted_final"
            )
        envs = SubprocVectorEnv([env_fn for _ in range(1)])  # Single environment for testing
    except Exception as e:
        print(f"Environment initialization failed: {e}")
        print("Using minimal environment for evaluation...")
        envs = None
    
    # Initialize preprocessor with robomimic settings - use dataset statistics  
    print("Initializing preprocessor...")
    from torchvision import transforms
    from datasets.robomimic_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD
    
    preprocessor = Preprocessor(
        action_mean=ACTION_MEAN[:2],  # 2D actions
        action_std=ACTION_STD[:2],
        state_mean=PROPRIO_MEAN[:7],  # 7D states 
        state_std=PROPRIO_STD[:7],
        proprio_mean=PROPRIO_MEAN[:7],
        proprio_std=PROPRIO_STD[:7],
        transform=transforms.Compose([
            transforms.Resize(model_cfg.img_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    # Initialize evaluator
    evaluator = PlanEvaluator(
        model=model,
        preprocessor=preprocessor,
        envs=envs,
        traj_dset=traj_dsets["train"],
        device=device,
        seed=cfg.seed,
        n_evals=cfg.n_evals,
        goal_source=cfg.goal_source,
        goal_H=cfg.goal_H,
        n_plot_samples=cfg.n_plot_samples,
    )
    
    # Run planning evaluation
    print("Starting planning evaluation...")
    cfg.planner.sub_planner.horizon = cfg["goal_H"]
    cfg.planner.n_taken_actions = cfg["goal_H"]
    
    results = evaluator.evaluate(cfg.planner, cfg.objective)
    # optional: assume planning horizon equals to goal horizon

    # Save results
    results_path = "planning_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Planning evaluation completed! Results saved to {results_path}")
    print(f"Success rate: {results.get('success_rate', 'N/A')}")
    
    return results

if __name__ == "__main__":
    planning_main()