#!/usr/bin/env python3

import hydra
from omegaconf import OmegaConf
import os
import sys

# 添加 dino_wm 到路径
sys.path.append('/home/minghao/workspace/dino_wm')

@hydra.main(config_path="dino_wm/conf", config_name="train_robomimic")
def test_config(cfg: OmegaConf):
    """测试 robomimic 配置"""
    print("=== TESTING ROBOMIMIC CONFIG ===")
    print(f"Environment: {cfg.env.name}")
    print(f"Data path: {cfg.env.dataset.data_path}")
    print(f"Data path exists: {os.path.exists(cfg.env.dataset.data_path)}")
    print(f"Image size: {cfg.img_size}")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Epochs: {cfg.training.epochs}")
    print(f"Model: {cfg.model._target_}")
    
    # 检查数据文件是否存在
    data_path = cfg.env.dataset.data_path
    required_files = ["states.pth", "velocities.pth", "abs_actions.pth", "rel_actions.pth", "seq_lengths.pkl"]
    
    print("\n=== CHECKING DATA FILES ===")
    for file in required_files:
        file_path = os.path.join(data_path, file)
        exists = os.path.exists(file_path)
        print(f"{file}: {'✅' if exists else '❌'}")
    
    # 检查视频文件
    obses_dir = os.path.join(data_path, "obses")
    if os.path.exists(obses_dir):
        video_files = [f for f in os.listdir(obses_dir) if f.endswith('.mp4')]
        print(f"Video files: {len(video_files)} found")
    else:
        print("❌ obses directory not found")
    
    # 测试数据集加载
    print("\n=== TESTING DATASET LOADING ===")
    try:
        from datasets.robomimic_dset import load_robomimic_slice_train_val
        from datasets.img_transforms import default_transform
        
        transform = default_transform(cfg.img_size)
        datasets, traj_dsets = load_robomimic_slice_train_val(
            transform=transform,
            data_path=cfg.env.dataset.data_path,
            normalize_action=cfg.env.dataset.normalize_action,
            with_velocity=cfg.env.dataset.with_velocity,
            num_hist=cfg.num_hist,
            num_pred=cfg.num_pred,
            frameskip=cfg.frameskip,
        )
        
        print("✅ Dataset loaded successfully!")
        print(f"Train dataset size: {len(datasets['train'])}")
        print(f"Val dataset size: {len(datasets['valid'])}")
        
        # 测试一个样本
        if len(datasets['train']) > 0:
            sample = datasets['train'][0]
            obs, act, state = sample
            print(f"Sample obs keys: {list(obs.keys())}")
            print(f"Sample obs['visual'] shape: {obs['visual'].shape}")
            print(f"Sample obs['proprio'] shape: {obs['proprio'].shape}")
            print(f"Sample action shape: {act.shape}")
            print(f"Sample state shape: {state.shape}")
            
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== CONFIG SUMMARY ===")
    print(f"Action embedding dim: {cfg.action_emb_dim}")
    print(f"Proprio embedding dim: {cfg.proprio_emb_dim}")
    print(f"Has predictor: {cfg.has_predictor}")
    print(f"Has decoder: {cfg.has_decoder}")
    print(f"Debug mode: {cfg.debug}")

if __name__ == "__main__":
    test_config() 