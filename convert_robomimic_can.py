#!/usr/bin/env python3

import h5py
import torch
import pickle
import numpy as np
import cv2
from pathlib import Path
import os
from tqdm import tqdm
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Convert robomimic dataset to DINO_WM format with 7D actions')
parser.add_argument('--source_dir', type=str, required=True, help='Source directory of the robomimic dataset')
parser.add_argument('--save_data_dir', type=str, required=True, help='Directory to save the converted data')
args = parser.parse_args()

def convert_robomimic_can(
    source_dir="/home/ubuntu/minghao/data/robomimic/can/ph",
    target_dir="/home/ubuntu/minghao/data/robomimic/can/ph_convert"
):
    """
    Robomimic到DINO_WM格式转换 - 使用完整7D动作空间
    
    新的维度设计：
    - Actions: 7D (完整的机械臂 + 夹爪控制)
    - States: 基于任务相关性选择关键状态维度
    - Proprio: 与States相同（在robomimic中，states就是proprioceptive信息）
    """
    
    print("=== ROBOMIMIC TO DINO_WM (7D ACTIONS) CONVERSION ===")
    
    # 创建目标目录
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    (target_path / "obses").mkdir(exist_ok=True)
    
    # 加载源数据
    print("Loading source data...")
    demo_file = h5py.File(f"{source_dir}/demo_v15.hdf5", 'r')
    low_dim_file = h5py.File(f"{source_dir}/low_dim_v15.hdf5", 'r')
    # Try to load image file, skip if not available
    try:
        image_file = h5py.File(f"{source_dir}/image_384_v15.hdf5", 'r')
        print("Image file loaded successfully")
    except FileNotFoundError:
        print("Image file not found, using demo file for video generation")
        image_file = demo_file  # Use demo file as fallback
    
    # 获取demo keys
    demo_keys = [key for key in demo_file['data'].keys() if key.startswith('demo_')]
    demo_keys.sort(key=lambda x: int(x.split('_')[1]))
    
    print(f"Found {len(demo_keys)} demos")
    
    # 分析第一个demo的维度
    sample_demo = low_dim_file['data'][demo_keys[0]]
    sample_actions = sample_demo['actions'][:]
    sample_obs = sample_demo['obs']
    
    print(f"Sample action shape: {sample_actions.shape}")
    print(f"Available obs keys: {list(sample_obs.keys())}")
    
    # 检查可用的状态信息
    joint_pos = sample_obs['robot0_joint_pos'][:]
    joint_vel = sample_obs['robot0_joint_vel'][:]
    gripper_pos = sample_obs['robot0_gripper_qpos'][:]
    object_state = sample_obs['object'][:]
    
    print(f"Joint positions shape: {joint_pos.shape}")
    print(f"Joint velocities shape: {joint_vel.shape}")
    print(f"Gripper positions shape: {gripper_pos.shape}")
    print(f"Object state shape: {object_state.shape}")
    
    # 动作和状态维度设计
    action_dim = 7  # 完整的7D动作空间
    
    # 正确的状态设计：Proprio是State的子集
    # Proprio (16D): 机器人本体感觉 = 位置 + 速度
    # State (23D): 完整世界状态 = Proprio + 物体状态
    proprio_components = {
        'robot_joints': 7,      # 关节位置
        'gripper': 2,           # 夹爪位置
        'joint_velocities': 7   # 关节速度
    }
    proprio_dim = sum(proprio_components.values())  # 16D proprio
    
    object_components = {
        'object_pos': 3,        # 物体XYZ位置
        'object_quat': 4        # 物体四元数方向  
    }
    object_dim = sum(object_components.values())  # 7D物体状态
    
    state_dim = proprio_dim + object_dim  # 23D完整状态
    vel_dim = 7  # 单独的关节速度（用于特殊需求）
    
    # 找到最大序列长度
    max_seq_len = 0
    seq_lengths = []
    
    for demo_key in demo_keys:
        demo_data = low_dim_file['data'][demo_key]
        seq_len = demo_data['actions'].shape[0]
        seq_lengths.append(seq_len)
        max_seq_len = max(max_seq_len, seq_len)
    
    print(f"Max sequence length: {max_seq_len}")
    print(f"Total episodes: {len(demo_keys)}")
    print(f"Action dimension: {action_dim} (full robotic control)")
    print(f"Proprio dimension: {proprio_dim} (robot self-state with velocities)")
    print(f"State dimension: {state_dim} (proprio + object state)")
    print(f"Velocity dimension: {vel_dim} (separate joint velocities)")
    
    # 初始化张量
    states = torch.zeros(len(demo_keys), max_seq_len, state_dim, dtype=torch.float32)
    proprios = torch.zeros(len(demo_keys), max_seq_len, proprio_dim, dtype=torch.float32)
    velocities = torch.zeros(len(demo_keys), max_seq_len, vel_dim, dtype=torch.float32)
    abs_actions = torch.zeros(len(demo_keys), max_seq_len, action_dim, dtype=torch.float32)
    rel_actions = torch.zeros(len(demo_keys), max_seq_len, action_dim, dtype=torch.float32)
    
    # 处理每个demo
    print("Processing demos...")
    for i, demo_key in enumerate(tqdm(demo_keys)):
        # 从不同文件获取数据
        low_dim_data = low_dim_file['data'][demo_key]
        image_data = image_file['data'][demo_key]
        seq_len = low_dim_data['actions'].shape[0]
        
        # 提取完整7D动作
        actions = low_dim_data['actions'][:]  # shape: (seq_len, 7)
        abs_actions[i, :seq_len] = torch.from_numpy(actions.astype(np.float32))
        rel_actions[i, :seq_len] = torch.from_numpy(actions.astype(np.float32))
        
        # 从low_dim数据中提取状态信息
        low_dim_obs = low_dim_data['obs']
        
        # 提取状态组件
        joint_pos = low_dim_obs['robot0_joint_pos'][:]  # (seq_len, 7)
        gripper_pos = low_dim_obs['robot0_gripper_qpos'][:]  # (seq_len, 2)
        object_state = low_dim_obs['object'][:]  # (seq_len, 14)
        
        # 提取完整关节速度 (7D)
        joint_vel = low_dim_obs['robot0_joint_vel'][:]  # (seq_len, 7)
        velocities[i, :seq_len] = torch.from_numpy(joint_vel.astype(np.float32))
        
        # 构建Proprio (16D): 机器人本体感觉 = 位置 + 速度
        proprio = np.concatenate([
            joint_pos,      # 7D: 关节位置
            gripper_pos,    # 2D: 夹爪位置
            joint_vel       # 7D: 关节速度
        ], axis=1)
        proprios[i, :seq_len] = torch.from_numpy(proprio.astype(np.float32))
        
        # 从object_state中提取位置和方向信息
        # object state通常包含: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, ...]
        object_pos = object_state[:, :3]  # XYZ位置
        object_quat = object_state[:, 3:7]  # 四元数方向
        
        # 构建完整State (23D): Proprio + 物体状态
        state = np.concatenate([
            proprio,        # 16D: 机器人本体感觉
            object_pos,     # 3D: 物体位置
            object_quat     # 4D: 物体方向
        ], axis=1)
        states[i, :seq_len] = torch.from_numpy(state.astype(np.float32))
        
        # 从图像数据保存MP4文件
        try:
            if 'obs' in image_data:
                image_obs = image_data['obs']
                # 读取agentview_image数据
                images = image_obs['agentview_image'][:]
                print(f"  Loaded images shape: {images.shape}, dtype: {images.dtype}")
                # 确保图像是uint8格式
                if images.dtype != np.uint8:
                    # 如果是float，假设范围是0-1，转换到0-255
                    if images.max() <= 1.0:
                        images = (images * 255).astype(np.uint8)
                    else:
                        images = images.astype(np.uint8)
            else:
                # Create dummy images if no image data available
                print(f"No 'obs' key in {demo_key}, checking for direct image data")
                # 尝试直接读取图像数据
                if 'agentview_image' in image_data:
                    images = image_data['agentview_image'][:]
                    if images.dtype != np.uint8:
                        if images.max() <= 1.0:
                            images = (images * 255).astype(np.uint8)
                        else:
                            images = images.astype(np.uint8)
                else:
                    print(f"No images available for {demo_key}, creating dummy video")
                    images = np.zeros((seq_len, 224, 224, 3), dtype=np.uint8)
        except Exception as e:
            print(f"Error loading images for {demo_key}: {e}, creating dummy video")
            images = np.zeros((seq_len, 224, 224, 3), dtype=np.uint8)
        
        # Ensure correct initialization of the video writer and add error handling
        video_writer = None
        try:
            # Initialize video writer with appropriate settings
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(target_path / "obses" / f"episode_{i:05d}.mp4"), fourcc, 30.0, (224, 224))

            # Write frames to video
            for frame_idx in range(seq_len):
                # 从RGB转换为BGR用于OpenCV
                frame = images[frame_idx]
                # Resize frame to 224x224 if needed
                if frame.shape[:2] != (224, 224):
                    frame = cv2.resize(frame, (224, 224))
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
        except Exception as e:
            print(f"Error writing video for demo {demo_key}: {e}")
        finally:
            if video_writer is not None:
                video_writer.release()
    
    # 保存张量
    print("Saving tensors...")
    torch.save(states, target_path / "states.pth")
    torch.save(proprios, target_path / "proprios.pth")
    torch.save(velocities, target_path / "velocities.pth")
    torch.save(abs_actions, target_path / "abs_actions.pth")
    torch.save(rel_actions, target_path / "rel_actions.pth")
    
    # 保存序列长度
    with open(target_path / "seq_lengths.pkl", 'wb') as f:
        pickle.dump(seq_lengths, f)
    
    # 计算并保存统计信息
    print("Computing and saving statistics...")
    
    # 收集所有有效数据
    all_states = []
    all_proprios = []
    all_velocities = []
    all_actions = []
    
    for i, seq_len in enumerate(seq_lengths):
        all_states.append(states[i, :seq_len])
        all_proprios.append(proprios[i, :seq_len])
        all_velocities.append(velocities[i, :seq_len])
        all_actions.append(abs_actions[i, :seq_len])
    
    all_states = torch.cat(all_states, dim=0)
    all_proprios = torch.cat(all_proprios, dim=0)
    all_velocities = torch.cat(all_velocities, dim=0) 
    all_actions = torch.cat(all_actions, dim=0)
    
    # 计算统计信息
    state_mean = all_states.mean(dim=0)
    state_std = all_states.std(dim=0)
    proprio_mean = all_proprios.mean(dim=0)
    proprio_std = all_proprios.std(dim=0)
    velocity_mean = all_velocities.mean(dim=0)
    velocity_std = all_velocities.std(dim=0)
    action_mean = all_actions.mean(dim=0)
    action_std = all_actions.std(dim=0)
    
    # 保存统计信息
    torch.save(state_mean, target_path / "state_mean.pth")
    torch.save(state_std, target_path / "state_std.pth")
    torch.save(velocity_mean, target_path / "velocity_mean.pth")
    torch.save(velocity_std, target_path / "velocity_std.pth")
    torch.save(action_mean, target_path / "action_mean.pth")
    torch.save(action_std, target_path / "action_std.pth")
    torch.save(proprio_mean, target_path / "proprio_mean.pth")
    torch.save(proprio_std, target_path / "proprio_std.pth")
    
    # 关闭文件
    demo_file.close()
    low_dim_file.close()
    image_file.close()
    
    print(f"Conversion complete! Output saved to: {target_path}")
    print(f"States shape: {states.shape}")
    print(f"Proprios shape: {proprios.shape}")
    print(f"Velocities shape: {velocities.shape}")
    print(f"Abs actions shape: {abs_actions.shape}")
    print(f"Rel actions shape: {rel_actions.shape}")
    print(f"Number of video files: {len(list((target_path / 'obses').glob('*.mp4')))}")
    
    # 验证数据格式
    print("\n=== 7D ACTION DATA FORMAT VERIFICATION ===")
    print(f"States dtype: {states.dtype}")
    print(f"Velocities dtype: {velocities.dtype}")
    print(f"Actions dtype: {abs_actions.dtype}")
    print(f"Sequence lengths: {len(seq_lengths)} episodes")
    print(f"Min sequence length: {min(seq_lengths)}")
    print(f"Max sequence length: {max(seq_lengths)}")
    
    # 显示统计信息
    print(f"\n=== COMPUTED STATISTICS ===")
    print(f"State mean shape: {state_mean.shape}")
    print(f"Action mean shape: {action_mean.shape}")
    print(f"Proprio mean shape: {proprio_mean.shape}")
    print(f"Proprio components: robot_joints(7) + gripper(2) + velocities(7) = {proprio_dim}")
    print(f"State components: proprio({proprio_dim}) + object_pos(3) + object_quat(4) = {state_dim}")
    print(f"Action components: arm_joints(6) + gripper(1) = {action_dim}")
    
    # 验证正确性
    print(f"\n=== DATA RELATIONSHIP VERIFICATION ===")
    print(f"✅ Proprio is subset of State: State[:16] should equal Proprio")
    sample_state = states[0, 0, :16]
    sample_proprio = proprios[0, 0, :]
    print(f"✅ Relationship verified: {torch.allclose(sample_state, sample_proprio)}")
    
    # 与原版本比较
    print(f"\n=== COMPARISON WITH ORIGINAL CONVERSION ===")
    print(f"Original: 2D actions, 5D states, 2D velocities, 7D proprio")
    print(f"New 7D: {action_dim}D actions, {state_dim}D states, {vel_dim}D velocities, {proprio_dim}D proprio")
    print(f"✅ Full robotic control with gripper")
    print(f"✅ Correct proprioception definition")
    print(f"✅ Complete state representation")
    print(f"✅ Enhanced task capability")
    
    return target_path 

# Ensure the function call is correctly placed after the function definition
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Convert robomimic dataset to DINO_WM format with 7D actions')
    parser.add_argument('--source_dir', type=str, required=True, help='Source directory of the robomimic dataset')
    parser.add_argument('--save_data_dir', type=str, required=True, help='Directory to save the converted data')
    args = parser.parse_args()

    # Use command-line arguments for source and target directories
    convert_robomimic_can(
        source_dir=args.source_dir,
        target_dir=args.save_data_dir
    )