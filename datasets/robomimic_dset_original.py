"""
Dataset loader for original DINO-WM implementation
Separate from the main dataset to avoid conflicts
"""

import torch
import numpy as np
from pathlib import Path
from datasets.traj_dset import TrajDataset, TrajSlicerDataset
from datasets.robomimic_dset import RobomimicDataset

def load_robomimic_slice_train_val_original(
    data_path,
    split_ratio=0.9,
    transform=None,
    with_velocity=True,
    normalize_action=True,
    n_rollout=None,
    num_hist=3,
    num_pred=1,
    frameskip=5,
):
    """
    Load robomimic dataset for original DINO-WM implementation
    This version properly handles the dataset indexing
    """
    print(f"Loading dataset from {data_path}")
    
    # Load the full dataset
    full_dset = RobomimicDataset(
        data_path=data_path,
        transform=transform,
        with_velocity=with_velocity,
        normalize_action=normalize_action,
        n_rollout=n_rollout
    )
    
    # Get the total number of samples
    total_samples = len(full_dset)
    train_size = int(total_samples * split_ratio)
    
    print(f"Total samples: {total_samples}, Train: {train_size}, Val: {total_samples - train_size}")
    
    # Create train/val splits with proper indexing
    class IndexedDataset:
        """Wrapper to handle proper indexing"""
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
            self._len = len(indices)
            
        def __len__(self):
            return self._len
            
        def __getitem__(self, idx):
            if idx >= self._len:
                raise IndexError(f"Index {idx} out of range for dataset with {self._len} samples")
            actual_idx = self.indices[idx]
            return self.dataset[actual_idx]
            
        def get_seq_length(self, idx):
            if idx >= self._len:
                return 0
            actual_idx = self.indices[idx]
            return self.dataset.get_seq_length(actual_idx)
            
        def get_frames(self, idx, frame_indices):
            actual_idx = self.indices[idx]
            return self.dataset.get_frames(actual_idx, frame_indices)
            
        @property
        def action_dim(self):
            return self.dataset.action_dim
            
        @property
        def proprio_dim(self):
            return self.dataset.proprio_dim
            
        @property
        def state_dim(self):
            return getattr(self.dataset, 'state_dim', self.dataset.proprio_dim)
            
        @property
        def action_mean(self):
            return self.dataset.action_mean
            
        @property
        def action_std(self):
            return self.dataset.action_std
            
        @property
        def proprio_mean(self):
            return self.dataset.proprio_mean
            
        @property
        def proprio_std(self):
            return self.dataset.proprio_std
            
        @property
        def transform(self):
            return self.dataset.transform
    
    # Create train and validation indices
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_samples))
    
    # Create indexed datasets
    train_dset = IndexedDataset(full_dset, train_indices)
    val_dset = IndexedDataset(full_dset, val_indices)
    
    # Create sliced datasets for temporal sequences
    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)
    
    print(f"Train slices: {len(train_slices)}, Val slices: {len(val_slices)}")
    
    return train_slices, val_slices