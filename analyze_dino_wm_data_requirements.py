#!/usr/bin/env python3

import torch
import pickle
from pathlib import Path

def analyze_dino_wm_data_requirements():
    """分析dino_wm系统实际需要的数据格式"""
    
    print("=== DINO_WM DATA REQUIREMENTS ANALYSIS ===")
    
    # 分析pusht dataset的实际使用情况
    pusht_dir = Path("/mnt/data1/minghao/bmw48/osfstorage/datasets/pusht_noise/train")
    
    print("\n=== PUSHT DATASET ANALYSIS ===")
    
    # 加载pusht数据
    states = torch.load(pusht_dir / "states.pth")
    velocities = torch.load(pusht_dir / "velocities.pth")
    abs_actions = torch.load(pusht_dir / "abs_actions.pth")
    rel_actions = torch.load(pusht_dir / "rel_actions.pth")
    tokens = torch.load(pusht_dir / "tokens.pth")
    
    with open(pusht_dir / "seq_lengths.pkl", 'rb') as f:
        seq_lengths = pickle.load(f)
    
    print(f"States shape: {states.shape}")
    print(f"Velocities shape: {velocities.shape}")
    print(f"Abs actions shape: {abs_actions.shape}")
    print(f"Rel actions shape: {rel_actions.shape}")
    print(f"Tokens type: {type(tokens)}")
    print(f"Tokens length: {len(tokens)}")
    print(f"First token shape: {tokens[0].shape}")
    print(f"Number of sequences: {len(seq_lengths)}")
    
    # 分析视频文件
    video_files = list((pusht_dir / "obses").glob("*.mp4"))
    print(f"Number of video files: {len(video_files)}")
    print(f"First few video files: {[f.name for f in video_files[:5]]}")
    
    print("\n=== DINO_WM DATA LOADER ANALYSIS ===")
    
    # 从代码分析，dino_wm的data loader返回：
    # obs, act, state = data
    # 其中：
    # - obs: dict with keys "visual" and "proprio"
    # - act: action tensor
    # - state: state tensor
    
    print("Data loader returns:")
    print("- obs: dict with keys 'visual' and 'proprio'")
    print("- act: action tensor")
    print("- state: state tensor")
    
    print("\n=== REQUIRED DATA FORMAT ===")
    print("Based on dino_wm analysis, we need:")
    print("1. states.pth: torch tensor [num_episodes, max_seq_len, state_dim]")
    print("2. velocities.pth: torch tensor [num_episodes, max_seq_len, vel_dim]")
    print("3. abs_actions.pth: torch tensor [num_episodes, max_seq_len, action_dim]")
    print("4. rel_actions.pth: torch tensor [num_episodes, max_seq_len, action_dim]")
    print("5. seq_lengths.pkl: list of sequence lengths")
    print("6. obses/: directory with episode_*.mp4 files")
    print("7. tokens.pth: list of torch tensors (VQVAE tokens) - OPTIONAL")
    
    print("\n=== KEY INSIGHTS ===")
    print("- dino_wm使用DINO作为encoder，不需要预计算的tokens")
    print("- 主要需要：states, actions, velocities, videos")
    print("- tokens.pth是可选的，用于VQVAE相关功能")
    print("- 数据通过TrajSlicerDataset进行切片处理")
    print("- 训练时使用obs['visual']和obs['proprio']")

if __name__ == "__main__":
    analyze_dino_wm_data_requirements() 