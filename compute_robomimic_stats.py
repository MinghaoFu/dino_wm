#!/usr/bin/env python3
"""
Compute dataset statistics for robomimic dataset to fix action normalization
"""
import torch
import pickle
from pathlib import Path

def compute_robomimic_stats():
    data_path = Path("/mnt/data1/minghao/robomimic/can/ph_converted_final")
    
    # Load states, actions, and velocities
    states = torch.load(data_path / "states.pth").float()
    velocities = torch.load(data_path / "velocities.pth").float()
    rel_actions = torch.load(data_path / "rel_actions.pth").float()
    abs_actions = torch.load(data_path / "abs_actions.pth").float()
    
    # Add velocities to states (as done in dataset)
    states_with_vel = torch.cat([states, velocities], dim=-1)
    
    print("Dataset shape info:")
    print(f"States shape: {states.shape}")
    print(f"Velocities shape: {velocities.shape}")
    print(f"States with velocity shape: {states_with_vel.shape}")
    print(f"Relative actions shape: {rel_actions.shape}")
    print(f"Absolute actions shape: {abs_actions.shape}")
    
    # Flatten across all trajectories for statistics
    states_flat = states_with_vel.reshape(-1, states_with_vel.shape[-1])  # Use states with velocity
    rel_actions_flat = rel_actions.reshape(-1, rel_actions.shape[-1])
    abs_actions_flat = abs_actions.reshape(-1, abs_actions.shape[-1])
    
    # Compute statistics
    print("\n=== RELATIVE ACTIONS STATS ===")
    rel_action_mean = rel_actions_flat.mean(dim=0)
    rel_action_std = rel_actions_flat.std(dim=0)
    print(f"REL_ACTION_MEAN = {rel_action_mean.tolist()}")
    print(f"REL_ACTION_STD = {rel_action_std.tolist()}")
    
    print("\n=== ABSOLUTE ACTIONS STATS ===")
    abs_action_mean = abs_actions_flat.mean(dim=0)
    abs_action_std = abs_actions_flat.std(dim=0)
    print(f"ABS_ACTION_MEAN = {abs_action_mean.tolist()}")
    print(f"ABS_ACTION_STD = {abs_action_std.tolist()}")
    
    print("\n=== STATE STATS ===")
    state_mean = states_flat.mean(dim=0)
    state_std = states_flat.std(dim=0)
    print(f"STATE_MEAN = {state_mean.tolist()}")
    print(f"STATE_STD = {state_std.tolist()}")
    
    # For robomimic, proprio is all 7 dimensions of state (agent_x, agent_y, block_x, block_y, angle, vel_x, vel_y)
    print("\n=== PROPRIO STATS (all 7 dims) ===")
    proprio_mean = state_mean  # All 7 dimensions
    proprio_std = state_std    # All 7 dimensions  
    print(f"PROPRIO_MEAN = {proprio_mean.tolist()}")  
    print(f"PROPRIO_STD = {proprio_std.tolist()}")
    
    print("\n=== ACTION RANGES ===")
    print(f"Rel actions min: {rel_actions_flat.min(dim=0)[0].tolist()}")
    print(f"Rel actions max: {rel_actions_flat.max(dim=0)[0].tolist()}")
    print(f"Abs actions min: {abs_actions_flat.min(dim=0)[0].tolist()}")
    print(f"Abs actions max: {abs_actions_flat.max(dim=0)[0].tolist()}")

if __name__ == "__main__":
    compute_robomimic_stats()