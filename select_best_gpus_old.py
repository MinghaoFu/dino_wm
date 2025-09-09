#!/usr/bin/env python3
"""
Script to automatically select the 4 GPUs with the lowest memory occupation
"""
import subprocess
import re

def get_gpu_memory_usage():
    """Get memory usage for all GPUs"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True, check=True)
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                gpu_id = int(parts[0])
                memory_used = int(parts[1])
                gpu_info.append((gpu_id, memory_used))
        
        return gpu_info
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return []

def select_best_gpus(num_gpus=4):
    """Select the num_gpus GPUs with lowest memory usage"""
    gpu_info = get_gpu_memory_usage()
    if not gpu_info:
        return []
    
    # Sort by memory usage (ascending)
    sorted_gpus = sorted(gpu_info, key=lambda x: x[1])
    
    # Select the first num_gpus
    best_gpus = [str(gpu[0]) for gpu in sorted_gpus[:num_gpus]]
    
    print(f"GPU memory usage:")
    for gpu_id, memory in sorted_gpus:
        print(f"  GPU {gpu_id}: {memory} MiB")
    
    print(f"Selected GPUs with lowest usage: {best_gpus}")
    return best_gpus

def select_single_best_gpu():
    """Select the single GPU with lowest memory usage"""
    gpu_info = get_gpu_memory_usage()
    if not gpu_info:
        return "0"  # fallback
    
    # Sort by memory usage (ascending)
    sorted_gpus = sorted(gpu_info, key=lambda x: x[1])
    
    best_gpu = str(sorted_gpus[0][0])
    print(f"GPU memory usage:")
    for gpu_id, memory in sorted_gpus:
        print(f"  GPU {gpu_id}: {memory} MiB")
    
    print(f"Selected GPU with lowest usage: {best_gpu}")
    return best_gpu

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        gpu = select_single_best_gpu()
        print(gpu)
    else:
        gpus = select_best_gpus(4)
        if gpus:
            print(",".join(gpus))
        else:
            print("0,1,2,3")  # fallback