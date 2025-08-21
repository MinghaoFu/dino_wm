#!/usr/bin/env python3

import h5py
import torch
import pickle
import numpy as np
import cv2
from pathlib import Path
import os
from tqdm import tqdm

def convert_robomimic_to_dino_wm_v2(
    source_dir="/mnt/data1/minghao/robomimic/can/ph",
    target_dir="/mnt/data1/minghao/robomimic/can/ph_converted_v2"
):
    """
    转换robomimic数据集格式到dino_wm格式 (v2)
    
    基于分析，dino_wm系统真正需要的数据：
    1. states.pth: [num_episodes, max_seq_len, state_dim]
    2. velocities.pth: [num_episodes, max_seq_len, vel_dim] 
    3. abs_actions.pth: [num_episodes, max_seq_len, action_dim]
    4. rel_actions.pth: [num_episodes, max_seq_len, action_dim]
    5. seq_lengths.pkl: 序列长度列表
    6. obses/: episode_*.mp4 视频文件
    
    不需要tokens.pth，因为dino_wm使用DINO实时编码图像
    """
    
    print("=== CONVERTING ROBOMIMIC TO DINO_WM FORMAT (V2) ===")
    
    # 创建目标目录
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    (target_path / "obses").mkdir(exist_ok=True)
    
    # 加载源数据
    print("Loading source data...")
    demo_file = h5py.File(f"{source_dir}/demo_v15.hdf5", 'r')
    low_dim_file = h5py.File(f"{source_dir}/low_dim_v15.hdf5", 'r')
    image_file = h5py.File(f"{source_dir}/image.hdf5", 'r')
    
    # 获取demo keys
    demo_keys = [key for key in demo_file['data'].keys() if key.startswith('demo_')]
    demo_keys.sort(key=lambda x: int(x.split('_')[1]))
    
    print(f"Found {len(demo_keys)} demos")
    
    # 分析数据结构和维度
    first_demo = demo_keys[0]
    first_demo_data = low_dim_file['data'][first_demo]
    
    # 从robomimic数据中提取状态信息
    # 使用robot0_joint_pos, robot0_gripper_qpos, object作为状态
    state_dim = 7 + 2 + 14  # joint_pos + gripper_pos + object = 23
    
    # 动作维度
    action_dim = 7  # actions是7维的
    
    # 速度维度 (2D)
    vel_dim = 2
    
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
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Velocity dimension: {vel_dim}")
    
    # 初始化张量
    states = torch.zeros(len(demo_keys), max_seq_len, state_dim, dtype=torch.float32)
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
        
        # 提取动作
        actions = low_dim_data['actions'][:]  # shape: (seq_len, 7)
        abs_actions[i, :seq_len] = torch.from_numpy(actions.astype(np.float32))
        
        # 相对动作，暂时使用与绝对动作相同
        # 在实际应用中，你可能想要计算相对动作
        rel_actions[i, :seq_len] = torch.from_numpy(actions.astype(np.float32))
        
        # 从low_dim数据中提取状态信息
        low_dim_obs = low_dim_data['obs']
        
        # 组合关节位置、夹爪位置和物体状态
        joint_pos = low_dim_obs['robot0_joint_pos'][:]  # (seq_len, 7)
        gripper_pos = low_dim_obs['robot0_gripper_qpos'][:]  # (seq_len, 2)
        object_state = low_dim_obs['object'][:]  # (seq_len, 14)
        
        # 连接形成状态
        state = np.concatenate([joint_pos, gripper_pos, object_state], axis=1)  # (seq_len, 23)
        states[i, :seq_len] = torch.from_numpy(state.astype(np.float32))
        
        # 计算速度 (使用关节速度的前两个维度作为2D速度)
        joint_vel = low_dim_obs['robot0_joint_vel'][:]  # (seq_len, 7)
        velocities[i, :seq_len, 0] = torch.from_numpy(joint_vel[:, 0].astype(np.float32))
        velocities[i, :seq_len, 1] = torch.from_numpy(joint_vel[:, 1].astype(np.float32))
        
        # 从图像数据保存MP4文件
        image_obs = image_data['obs']
        images = image_obs['agentview_image'][:]
        
        # 创建视频写入器
        video_path = target_path / "obses" / f"episode_{i:05d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (224, 224))
        
        for frame_idx in range(seq_len):
            # 从RGB转换为BGR用于OpenCV
            frame = images[frame_idx]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
    
    # 保存张量
    print("Saving tensors...")
    torch.save(states, target_path / "states.pth")
    torch.save(velocities, target_path / "velocities.pth")
    torch.save(abs_actions, target_path / "abs_actions.pth")
    torch.save(rel_actions, target_path / "rel_actions.pth")
    
    # 保存序列长度
    with open(target_path / "seq_lengths.pkl", 'wb') as f:
        pickle.dump(seq_lengths, f)
    
    # 关闭文件
    demo_file.close()
    low_dim_file.close()
    image_file.close()
    
    print(f"Conversion complete! Output saved to: {target_path}")
    print(f"States shape: {states.shape}")
    print(f"Velocities shape: {velocities.shape}")
    print(f"Abs actions shape: {abs_actions.shape}")
    print(f"Rel actions shape: {rel_actions.shape}")
    print(f"Number of video files: {len(list((target_path / 'obses').glob('*.mp4')))}")
    
    # 验证数据格式
    print("\n=== DATA FORMAT VERIFICATION ===")
    print(f"States dtype: {states.dtype}")
    print(f"Velocities dtype: {velocities.dtype}")
    print(f"Actions dtype: {abs_actions.dtype}")
    print(f"Sequence lengths: {len(seq_lengths)} episodes")
    print(f"Min sequence length: {min(seq_lengths)}")
    print(f"Max sequence length: {max(seq_lengths)}")
    
    return target_path

if __name__ == "__main__":
    convert_robomimic_to_dino_wm_v2() 