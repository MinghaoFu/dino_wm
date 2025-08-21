import os
import torch
import numpy as np
import hydra
import json
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from einops import rearrange
import warnings

warnings.filterwarnings("ignore")

def load_dinov2_model():
    """Load DINOv2 model"""
    print("Loading DINOv2 model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.eval()
    return model

def extract_features_from_dataset(model, dset, num_samples=100, device='cuda'):
    """Extract DINOv2 features from dataset"""
    print(f"Extracting features from dataset, num_samples: {num_samples}")
    
    features_list = []
    actions_list = []
    states_list = []
    
    model = model.to(device)
    
    for i in range(min(num_samples, len(dset))):
        if i % 10 == 0:
            print(f"Processing sample {i}/{num_samples}")
        
        # Get sample from dataset
        sample = dset[i]  # Returns (obs, actions, states, env_info)
        obs, actions, states, env_info = sample
        
        # Extract visual features
        with torch.no_grad():
            # Process visual data: [T, C, H, W] -> [T, H, W, C] -> [T, H, W, C]
            visual_data = obs['visual'].permute(0, 2, 3, 1)  # [T, H, W, C]
            
            # Process each frame
            frame_features = []
            for t in range(visual_data.shape[0]):
                frame = visual_data[t]  # [H, W, C]
                frame = frame.unsqueeze(0)  # [1, H, W, C]
                frame = frame.permute(0, 3, 1, 2)  # [1, C, H, W]
                frame = frame.to(device)
                
                # Extract features using DINOv2
                features = model.forward_features(frame)
                # Use the last layer features (CLS token)
                cls_features = features['x_norm_clstoken']  # [1, 1024] for ViT-L/14
                frame_features.append(cls_features.cpu().numpy())
            
            # Stack all frame features
            episode_features = np.array(frame_features)  # [T, 1024]
            
            features_list.append(episode_features)
            actions_list.append(actions.numpy())
            states_list.append(states.numpy())
    
    return features_list, actions_list, states_list

def save_features(features_list, actions_list, states_list, output_dir):
    """Save extracted features"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save features
    for i, (features, actions, states) in enumerate(zip(features_list, actions_list, states_list)):
        np.save(output_path / f"sample_{i}_features.npy", features)
        np.save(output_path / f"sample_{i}_actions.npy", actions)
        np.save(output_path / f"sample_{i}_states.npy", states)
    
    # Save summary
    total_samples = len(features_list)
    feature_dim = features_list[0].shape[1] if features_list else 0
    
    summary = {
        "total_samples": total_samples,
        "feature_dim": feature_dim,
        "feature_shape": features_list[0].shape if features_list else None,
        "action_shape": actions_list[0].shape if actions_list else None,
        "state_shape": states_list[0].shape if states_list else None
    }
    
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved {total_samples} samples to {output_path}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Feature shape: {features_list[0].shape if features_list else None}")
    
    return summary

def main():
    # Configuration
    model_name = "point_maze"  # Change this to your model name
    ckpt_base_path = "/mnt/data1/minghao/bmw48/osfstorage/checkpoints/"
    num_samples = 100
    output_dir = "dinov2_features"
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model configuration
    model_path = f"{ckpt_base_path}/outputs/{model_name}/"
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)
    
    # Load dataset
    print("Loading dataset...")
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]
    print(f"Dataset loaded with {len(dset)} samples")
    
    # Load DINOv2 model
    model = load_dinov2_model()
    
    # Extract features
    features_list, actions_list, states_list = extract_features_from_dataset(
        model, dset, num_samples=num_samples, device=device
    )
    
    # Save features
    summary = save_features(features_list, actions_list, states_list, output_dir)
    
    print("Feature extraction completed!")
    print(f"Output directory: {output_dir}")
    print(f"Summary: {summary}")

if __name__ == "__main__":
    main() 