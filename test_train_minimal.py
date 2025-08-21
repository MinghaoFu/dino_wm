#!/usr/bin/env python3
"""Minimal single-GPU training test"""

import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0

import hydra
from omegaconf import OmegaConf
from our_train import Trainer

@hydra.main(config_path="conf", config_name="train_minimal", version_base=None)
def main(cfg: OmegaConf):
    print(f"Testing with batch size: {cfg.training.batch_size}")
    print(f"Using GPU: {torch.cuda.current_device()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    try:
        trainer = Trainer(cfg)
        print("✓ Trainer initialized successfully")
        
        # Just run one training step
        trainer.model.train()
        dataloader = trainer.dataloaders["train"]
        data = next(iter(dataloader))
        obs, act, state = data
        
        print(f"✓ Data loaded: obs={obs['visual'].shape}, act={act.shape}")
        
        # Forward pass
        z_out, visual_out, visual_reconstructed, loss, loss_components = trainer.model(obs, act)
        print(f"✓ Forward pass successful: loss={loss.item():.4f}")
        
        # Backward pass
        trainer.encoder_optimizer.zero_grad()
        trainer.decoder_optimizer.zero_grad() 
        trainer.predictor_optimizer.zero_grad()
        trainer.action_encoder_optimizer.zero_grad()
        
        loss.backward()
        print("✓ Backward pass successful")
        
        print("✓ All tests passed! Training should work.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()