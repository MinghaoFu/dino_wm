"""
GPU utilities for automatic selection of lowest occupied GPUs
"""
import os
import subprocess
import logging

log = logging.getLogger(__name__)

def auto_select_gpus(num_gpus=1, min_free_memory_gb=2.0):
    """
    Automatically select GPUs with lowest memory usage and set CUDA_VISIBLE_DEVICES.
    
    Args:
        num_gpus: Number of GPUs to select
        min_free_memory_gb: Minimum free memory required in GB
        
    Returns:
        List of selected GPU IDs
    """
    try:
        # Get GPU memory info
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, check=True)
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                gpu_id = int(parts[0])
                used_mb = int(parts[1])
                total_mb = int(parts[2])
                gpu_info.append((gpu_id, used_mb, total_mb))
        
        if not gpu_info:
            log.warning("No GPU info found, using default GPU selection")
            return list(range(num_gpus))
        
        min_free_memory_mb = min_free_memory_gb * 1024
        
        # Calculate usage and filter by minimum free memory
        available_gpus = []
        for gpu_id, used_mb, total_mb in gpu_info:
            free_mb = total_mb - used_mb
            usage_percent = (used_mb / total_mb) * 100
            
            if free_mb >= min_free_memory_mb:
                available_gpus.append((gpu_id, usage_percent, free_mb, used_mb, total_mb))
        
        if len(available_gpus) < num_gpus:
            log.warning(f"Only {len(available_gpus)} GPUs have >= {min_free_memory_gb}GB free memory, requested {num_gpus}")
            if len(available_gpus) == 0:
                log.error("No GPUs meet memory requirements!")
                return list(range(num_gpus))  # Fallback to default
        
        # Sort by usage percentage (lowest first), then by free memory (highest first)
        available_gpus.sort(key=lambda x: (x[1], -x[2]))
        
        # Select the best GPUs
        selected_gpus = available_gpus[:min(num_gpus, len(available_gpus))]
        
        # Log selection info
        log.info("GPU Memory Status:")
        for gpu_id, used_mb, total_mb in gpu_info:
            free_mb = total_mb - used_mb
            usage_percent = (used_mb / total_mb) * 100
            status = "SELECTED" if gpu_id in [g[0] for g in selected_gpus] else "        "
            log.info(f"  GPU {gpu_id}: {used_mb:5d}MB/{total_mb:5d}MB ({usage_percent:5.1f}%) - {free_mb:5d}MB free {status}")
        
        selected_gpu_ids = [gpu_id for gpu_id, _, _, _, _ in selected_gpus]
        
        # Set CUDA_VISIBLE_DEVICES if not already set by external launcher
        if 'CUDA_VISIBLE_DEVICES' not in os.environ or os.environ.get('ACCELERATE_USE_CUDA', False):
            cuda_visible = ','.join(map(str, selected_gpu_ids))
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible
            log.info(f"Set CUDA_VISIBLE_DEVICES={cuda_visible}")
        else:
            log.info(f"CUDA_VISIBLE_DEVICES already set: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        
        return selected_gpu_ids
        
    except Exception as e:
        log.error(f"Error in auto GPU selection: {e}")
        return list(range(num_gpus))  # Fallback to default