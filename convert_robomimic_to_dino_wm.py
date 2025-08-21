#!/usr/bin/env python3

import h5py
import torch
import pickle
import numpy as np
import cv2
from pathlib import Path
import os
from tqdm import tqdm

def convert_robomimic_to_dino_wm(
    source_dir="/mnt/data1/minghao/robomimic/can/ph",
    target_dir="/mnt/data1/minghao/robomimic/can/ph_converted"
):
    """
    Convert robomimic dataset format to dino_wm format
    
    Source format:
    - demo_v15.hdf5: Contains demo data with actions, obs, next_obs, etc.
    - low_dim_v15.hdf5: Contains low-dimensional state data
    - image.hdf5: Contains image data
    - image_dino.hdf5: Contains DINO features
    
    Target format:
    - states.pth: torch tensor of shape [num_episodes, max_seq_len, state_dim]
    - velocities.pth: torch tensor of shape [num_episodes, max_seq_len, vel_dim]
    - abs_actions.pth: torch tensor of shape [num_episodes, max_seq_len, action_dim]
    - rel_actions.pth: torch tensor of shape [num_episodes, max_seq_len, action_dim]
    - tokens.pth: torch tensor of shape [num_episodes, max_seq_len, token_dim]
    - seq_lengths.pkl: list of sequence lengths
    - obses/: directory containing episode_*.mp4 files
    """
    
    print("=== CONVERTING ROBOMIMIC TO DINO_WM FORMAT ===")
    
    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    (target_path / "obses").mkdir(exist_ok=True)
    
    # Load source data
    print("Loading source data...")
    demo_file = h5py.File(f"{source_dir}/demo_v15.hdf5", 'r')
    low_dim_file = h5py.File(f"{source_dir}/low_dim_v15.hdf5", 'r')
    image_file = h5py.File(f"{source_dir}/image.hdf5", 'r')
    dino_file = h5py.File(f"{source_dir}/image_dino.hdf5", 'r')
    
    # Get demo keys from data group (excluding 'mask')
    demo_keys = [key for key in demo_file['data'].keys() if key.startswith('demo_')]
    demo_keys.sort(key=lambda x: int(x.split('_')[1]))  # Sort by demo number
    
    print(f"Found {len(demo_keys)} demos")
    
    # Analyze data structure to determine dimensions
    first_demo = demo_keys[0]
    first_demo_data = low_dim_file['data'][first_demo]
    
    # Extract state information from low_dim data
    # We'll use robot0_joint_pos, robot0_gripper_qpos, and object as state
    state_dim = 7 + 2 + 14  # joint_pos + gripper_pos + object = 23
    
    # Action dimension
    action_dim = 7  # actions are 7-dimensional
    
    # DINO feature dimension
    dino_dim = 768
    
    # Find max sequence length
    max_seq_len = 0
    seq_lengths = []
    
    for demo_key in demo_keys:
        demo_data = low_dim_file['data'][demo_key]
        seq_len = demo_data['actions'].shape[0]
        seq_lengths.append(seq_len)
        max_seq_len = max(max_seq_len, seq_len)
    
    print(f"Max sequence length: {max_seq_len}")
    print(f"Total episodes: {len(demo_keys)}")
    
    # Initialize tensors
    states = torch.zeros(len(demo_keys), max_seq_len, state_dim, dtype=torch.float64)
    velocities = torch.zeros(len(demo_keys), max_seq_len, 2, dtype=torch.float64)  # 2D velocities
    abs_actions = torch.zeros(len(demo_keys), max_seq_len, action_dim, dtype=torch.float64)
    rel_actions = torch.zeros(len(demo_keys), max_seq_len, action_dim, dtype=torch.float64)
    tokens = torch.zeros(len(demo_keys), max_seq_len, dino_dim, dtype=torch.float32)
    
    # Process each demo
    print("Processing demos...")
    for i, demo_key in enumerate(tqdm(demo_keys)):
        # Get data from different files
        low_dim_data = low_dim_file['data'][demo_key]
        image_data = image_file['data'][demo_key]
        seq_len = low_dim_data['actions'].shape[0]
        
        # Extract actions
        actions = low_dim_data['actions'][:]  # shape: (seq_len, 7)
        abs_actions[i, :seq_len] = torch.from_numpy(actions)
        
        # For relative actions, we'll use the same as absolute for now
        # In a real implementation, you might want to compute relative actions
        rel_actions[i, :seq_len] = torch.from_numpy(actions)
        
        # Extract state information from low_dim data
        low_dim_obs = low_dim_data['obs']
        
        # Combine joint positions, gripper positions, and object state
        joint_pos = low_dim_obs['robot0_joint_pos'][:]  # (seq_len, 7)
        gripper_pos = low_dim_obs['robot0_gripper_qpos'][:]  # (seq_len, 2)
        object_state = low_dim_obs['object'][:]  # (seq_len, 14)
        
        # Concatenate to form state
        state = np.concatenate([joint_pos, gripper_pos, object_state], axis=1)  # (seq_len, 23)
        states[i, :seq_len] = torch.from_numpy(state)
        
        # For velocities, we'll use joint velocities for now
        # You might want to compute actual velocities from positions
        joint_vel = low_dim_obs['robot0_joint_vel'][:]  # (seq_len, 7)
        # Use first two joint velocities as 2D velocity
        velocities[i, :seq_len, 0] = torch.from_numpy(joint_vel[:, 0])
        velocities[i, :seq_len, 1] = torch.from_numpy(joint_vel[:, 1])
        
        # Extract DINO features
        dino_features = dino_file[demo_key][:]  # (seq_len, 768)
        tokens[i, :seq_len] = torch.from_numpy(dino_features)
        
        # Save images as MP4 files from image data
        image_obs = image_data['obs']
        images = image_obs['agentview_image'][:]  # (seq_len, 224, 224, 3)
        
        # Create video writer
        video_path = target_path / "obses" / f"episode_{i:05d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (224, 224))
        
        for frame_idx in range(seq_len):
            # Convert from RGB to BGR for OpenCV
            frame = images[frame_idx]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
    
    # Save tensors
    print("Saving tensors...")
    torch.save(states, target_path / "states.pth")
    torch.save(velocities, target_path / "velocities.pth")
    torch.save(abs_actions, target_path / "abs_actions.pth")
    torch.save(rel_actions, target_path / "rel_actions.pth")
    torch.save(tokens, target_path / "tokens.pth")
    
    # Save sequence lengths
    with open(target_path / "seq_lengths.pkl", 'wb') as f:
        pickle.dump(seq_lengths, f)
    
    # Close files
    demo_file.close()
    low_dim_file.close()
    image_file.close()
    dino_file.close()
    
    print(f"Conversion complete! Output saved to: {target_path}")
    print(f"States shape: {states.shape}")
    print(f"Velocities shape: {velocities.shape}")
    print(f"Abs actions shape: {abs_actions.shape}")
    print(f"Rel actions shape: {rel_actions.shape}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Number of video files: {len(list((target_path / 'obses').glob('*.mp4')))}")

if __name__ == "__main__":
    convert_robomimic_to_dino_wm() 