#!/usr/bin/env python3
"""Minimal test to find working batch size with current GPU memory constraints"""

import torch
import os
import hydra
from omegaconf import OmegaConf

# Test with very small batch size first
def test_memory():
    print("Testing minimal memory usage...")
    
    # Try to allocate on different GPUs to find free memory
    for gpu_id in range(8):
        try:
            device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
            
            # Get memory info
            total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            reserved_mem = torch.cuda.memory_reserved(gpu_id) / 1024**3
            allocated_mem = torch.cuda.memory_allocated(gpu_id) / 1024**3
            free_mem = total_mem - reserved_mem
            
            print(f"GPU {gpu_id}: Total: {total_mem:.1f}GB, Reserved: {reserved_mem:.1f}GB, Allocated: {allocated_mem:.1f}GB, Free: {free_mem:.1f}GB")
            
            # Test tiny tensor allocation
            if free_mem > 10:  # At least 10GB free
                try:
                    test_tensor = torch.randn(1, 3, 224, 224, device=device)
                    print(f"GPU {gpu_id}: Can allocate tensors ✓")
                    del test_tensor
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"GPU {gpu_id}: Cannot allocate tensors: {e}")
            else:
                print(f"GPU {gpu_id}: Insufficient free memory")
                
        except Exception as e:
            print(f"GPU {gpu_id}: Error accessing: {e}")

if __name__ == "__main__":
    test_memory()