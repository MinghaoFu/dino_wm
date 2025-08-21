#!/usr/bin/env python3

import os
import sys
import subprocess

def run_robomimic_training():
    """运行robomimic训练"""
    
    print("=== STARTING ROBOMIMIC TRAINING ===")
    
    # 切换到dino_wm目录
    dino_wm_dir = "dino_wm"
    if not os.path.exists(dino_wm_dir):
        print(f"❌ Error: {dino_wm_dir} directory not found")
        return
    
    # 检查数据路径
    data_path = "/mnt/data1/minghao/robomimic/can/ph_converted_final"
    if not os.path.exists(data_path):
        print(f"❌ Error: Data path {data_path} not found")
        return
    
    print(f"✅ Data path exists: {data_path}")
    
    # 检查配置文件
    config_file = os.path.join(dino_wm_dir, "conf", "train_robomimic.yaml")
    if not os.path.exists(config_file):
        print(f"❌ Error: Config file {config_file} not found")
        return
    
    print(f"✅ Config file exists: {config_file}")
    
    # 运行训练
    print("🚀 Starting training...")
    cmd = [
        "python", "train_robomimic.py",
        "--config-path", "conf",
        "--config-name", "train_robomimic"
    ]
    
    try:
        subprocess.run(cmd, cwd=dino_wm_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
    except KeyboardInterrupt:
        print("⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    run_robomimic_training() 