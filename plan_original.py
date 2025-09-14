#!/usr/bin/env python3

import hydra
import logging
import os
import torch
from omegaconf import OmegaConf, open_dict

from planning.workspace import PlanWorkspace


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="plan_original")
def main(cfg):
    """
    Planning script for original DINO-WM architecture (no projection)
    Uses full 416D features (384D visual + 32D proprio) for planning
    """
    # Update config with checkpoint information
    with open_dict(cfg):
        # Find latest checkpoint
        ckpt_dir = os.path.join(cfg.ckpt_base_path, "checkpoints")
        if os.path.exists(ckpt_dir):
            latest_ckpt = os.path.join(ckpt_dir, "model_latest.pth")
            if os.path.exists(latest_ckpt):
                cfg.ckpt_path = latest_ckpt
                log.info(f"Found latest checkpoint: {latest_ckpt}")
            else:
                log.error(f"No checkpoint found in {ckpt_dir}")
                return
        else:
            log.error(f"Checkpoint directory not found: {ckpt_dir}")
            return
            
        # Set output directory for planning results
        cfg.output_dir = os.path.join(cfg.ckpt_base_path, "planning_results")
        os.makedirs(cfg.output_dir, exist_ok=True)
        
        # Original DINO-WM planning uses full features (no projection)
        cfg.use_projection = False
        cfg.feature_dim = 416  # 384D visual + 32D proprio
    
    # Auto-select GPU
    if torch.cuda.is_available():
        # Use GPU 0 by default, or specify with CUDA_VISIBLE_DEVICES
        device = torch.device("cuda:0")
        log.info(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        log.warning("CUDA not available, using CPU")
    
    # Set device in config
    with open_dict(cfg):
        cfg.device = device
    
    log.info("="*50)
    log.info("Original DINO-WM Planning Evaluation")
    log.info("="*50)
    log.info(f"Checkpoint: {cfg.ckpt_path}")
    log.info(f"Output directory: {cfg.output_dir}")
    log.info(f"Feature dimension: {cfg.feature_dim}D")
    log.info(f"Device: {device}")
    log.info("="*50)
    
    # Create planning workspace
    workspace = PlanWorkspace(cfg)
    
    # Run planning evaluation
    log.info("Starting planning evaluation...")
    results = workspace.perform_planning()
    
    # Log results summary
    if results:
        log.info("Planning Results Summary:")
        log.info("-" * 30)
        for key, value in results.items():
            if isinstance(value, (int, float)):
                log.info(f"{key}: {value:.4f}")
            else:
                log.info(f"{key}: {value}")
        log.info("-" * 30)
    else:
        log.warning("Planning evaluation returned empty results")
    
    log.info("Planning evaluation completed!")


if __name__ == "__main__":
    main()