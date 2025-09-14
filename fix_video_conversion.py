#!/usr/bin/env python3
"""
Fix video conversion - Generate proper videos from robomimic data
This script renders episodes directly using robosuite environment
"""

import h5py
import torch
import pickle
import numpy as np
import cv2
from pathlib import Path
import os
from tqdm import tqdm
import argparse

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def generate_videos_from_states(
    source_dir="/home/ubuntu/minghao/data/robomimic/can/ph",
    target_dir="/home/ubuntu/minghao/data/robomimic/can/ph_convert"
):
    """
    Generate videos by replaying states in robosuite environment
    """
    print("=== FIXING VIDEO CONVERSION ===")

    # Import robosuite and robomimic
    try:
        import robosuite as suite
        from robosuite.wrappers import GymWrapper
        import robomimic
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils
    except ImportError as e:
        print(f"Error importing required packages: {e}")
        print("Please ensure robosuite and robomimic are installed")
        return

    # Create target directory
    target_path = Path(target_dir)
    video_dir = target_path / "obses"
    video_dir.mkdir(exist_ok=True, parents=True)

    # Load demo file
    print("Loading demo data...")
    demo_file = h5py.File(f"{source_dir}/demo_v15.hdf5", 'r')

    # Get environment metadata
    env_meta = FileUtils.get_env_metadata_from_dataset(demo_file)
    print(f"Environment: {env_meta['env_name']}")
    print(f"Environment kwargs: {env_meta.get('env_kwargs', {})}")

    # Create environment for rendering
    print("Creating environment for rendering...")
    env = suite.make(
        env_name=env_meta['env_name'],
        robots="Panda",  # Use Panda robot
        has_renderer=False,  # No on-screen renderer
        has_offscreen_renderer=True,  # Use offscreen rendering
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=384,
        camera_widths=384,
        reward_shaping=True,
        control_freq=20,
        horizon=500,
        **env_meta.get('env_kwargs', {})
    )

    # Get demo keys
    demo_keys = [key for key in demo_file['data'].keys() if key.startswith('demo_')]
    demo_keys.sort(key=lambda x: int(x.split('_')[1]))

    print(f"Found {len(demo_keys)} demos")

    # Process each demo
    print("Generating videos for each demo...")
    for i, demo_key in enumerate(tqdm(demo_keys)):
        demo_data = demo_file['data'][demo_key]
        states = demo_data['states'][:]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = str(video_dir / f"episode_{i:05d}.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (224, 224))

        # Reset environment and replay states
        for state_idx, state in enumerate(states):
            # Set environment to this state
            env.sim.set_state_from_flattened(state)
            env.sim.forward()

            # Render camera view
            img = env.sim.render(
                camera_name="agentview",
                width=384,
                height=384,
                depth=False
            )

            # Convert from RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Resize to 224x224
            img_resized = cv2.resize(img_bgr, (224, 224))

            # Write frame
            video_writer.write(img_resized)

        video_writer.release()

        # Verify video was created and has content
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret and frame is not None:
                if np.mean(frame) > 1:  # Check if not all black
                    print(f"✓ Video {i:05d}: OK (mean pixel value: {np.mean(frame):.1f})")
                else:
                    print(f"✗ Video {i:05d}: Black frames detected!")
            cap.release()

    demo_file.close()
    env.close()

    print(f"\n✅ Videos generated in: {video_dir}")
    print("Run the original conversion script to complete the dataset conversion")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fix video conversion for robomimic dataset')
    parser.add_argument('--source_dir', type=str,
                       default="/home/ubuntu/minghao/data/robomimic/can/ph",
                       help='Source directory of the robomimic dataset')
    parser.add_argument('--target_dir', type=str,
                       default="/home/ubuntu/minghao/data/robomimic/can/ph_convert",
                       help='Target directory for converted data')

    args = parser.parse_args()
    generate_videos_from_states(args.source_dir, args.target_dir)