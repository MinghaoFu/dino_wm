#!/usr/bin/env python3
"""
Enhanced GPU selection script using nvitop for both memory and utilization metrics
"""
import sys
import argparse
from typing import List, Tuple

try:
    from nvitop import Device
    NVITOP_AVAILABLE = True
except ImportError:
    NVITOP_AVAILABLE = False
    print("Warning: nvitop not available, falling back to nvidia-ml-py", file=sys.stderr)
    try:
        import pynvml
        PYNVML_AVAILABLE = True
    except ImportError:
        PYNVML_AVAILABLE = False
        print("Error: Neither nvitop nor pynvml available", file=sys.stderr)
        sys.exit(1)

def get_gpu_metrics_nvitop() -> List[Tuple[int, float, float, int]]:
    """Get GPU metrics using nvitop: (gpu_id, memory_percent, gpu_util_percent, memory_used_mb)"""
    devices = Device.all()
    gpu_metrics = []
    
    for device in devices:
        gpu_id = device.cuda_index
        memory_used = device.memory_used_human()
        memory_total = device.memory_total_human()
        memory_percent = device.memory_percent()
        gpu_util = device.gpu_utilization()
        
        # Convert memory to MB for compatibility
        memory_used_mb = int(device.memory_used() / 1024 / 1024) if device.memory_used() else 0
        
        gpu_metrics.append((gpu_id, memory_percent, gpu_util, memory_used_mb))
    
    return gpu_metrics

def get_gpu_metrics_fallback() -> List[Tuple[int, float, float, int]]:
    """Fallback GPU metrics using pynvml"""
    if not PYNVML_AVAILABLE:
        return []
    
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    gpu_metrics = []
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        
        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used_mb = mem_info.used // 1024 // 1024
        memory_percent = (mem_info.used / mem_info.total) * 100
        
        # GPU utilization
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
        except:
            gpu_util = 0  # fallback if utilization not available
        
        gpu_metrics.append((i, memory_percent, gpu_util, memory_used_mb))
    
    return gpu_metrics

def calculate_gpu_score(memory_percent: float, gpu_util: float, memory_weight: float = 0.7, util_weight: float = 0.3) -> float:
    """Calculate GPU suitability score (lower is better)"""
    return (memory_percent * memory_weight) + (gpu_util * util_weight)

def select_best_gpus(num_gpus: int = 4, memory_weight: float = 0.7, util_weight: float = 0.3) -> List[str]:
    """Select the best GPUs based on memory and utilization"""
    
    # Try nvitop first, fallback to pynvml
    if NVITOP_AVAILABLE:
        gpu_metrics = get_gpu_metrics_nvitop()
    else:
        gpu_metrics = get_gpu_metrics_fallback()
    
    if not gpu_metrics:
        print("Error: Could not retrieve GPU metrics", file=sys.stderr)
        return ["0", "1", "2", "3"][:num_gpus]  # fallback
    
    # Calculate scores and sort
    gpu_scores = []
    for gpu_id, memory_percent, gpu_util, memory_used_mb in gpu_metrics:
        score = calculate_gpu_score(memory_percent, gpu_util, memory_weight, util_weight)
        gpu_scores.append((gpu_id, score, memory_percent, gpu_util, memory_used_mb))
    
    # Sort by score (lower is better)
    sorted_gpus = sorted(gpu_scores, key=lambda x: x[1])
    
    # Display information
    print("GPU Utilization & Memory Status:")
    print(f"{'GPU':<3} {'Mem%':<6} {'Util%':<6} {'MemMB':<8} {'Score':<6}")
    print("-" * 35)
    for gpu_id, score, mem_pct, gpu_util, mem_mb in sorted_gpus:
        indicator = "🟢" if score < 20 else "🟡" if score < 50 else "🔴"
        print(f"{gpu_id:<3} {mem_pct:<6.1f} {gpu_util:<6.1f} {mem_mb:<8} {score:<6.1f} {indicator}")
    
    # Select best GPUs
    best_gpus = [str(gpu[0]) for gpu in sorted_gpus[:num_gpus]]
    print(f"Selected {num_gpus} best GPUs: {best_gpus}")
    
    return best_gpus

def select_single_best_gpu(memory_weight: float = 0.7, util_weight: float = 0.3) -> str:
    """Select the single best GPU"""
    best_gpus = select_best_gpus(num_gpus=1, memory_weight=memory_weight, util_weight=util_weight)
    return best_gpus[0] if best_gpus else "0"

def main():
    parser = argparse.ArgumentParser(description="Select best GPUs based on memory and utilization")
    parser.add_argument("mode", nargs="?", choices=["single", "multi"], default="multi",
                       help="Selection mode: single GPU or multiple GPUs (default: multi)")
    parser.add_argument("--num-gpus", type=int, default=4, 
                       help="Number of GPUs to select in multi mode (default: 4)")
    parser.add_argument("--memory-weight", type=float, default=0.7,
                       help="Weight for memory usage in selection (default: 0.7)")
    parser.add_argument("--util-weight", type=float, default=0.3,
                       help="Weight for GPU utilization in selection (default: 0.3)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Only output the selected GPU IDs")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        gpu = select_single_best_gpu(args.memory_weight, args.util_weight)
        if args.quiet:
            print(gpu)
        else:
            print(f"Selected GPU: {gpu}")
    else:
        gpus = select_best_gpus(args.num_gpus, args.memory_weight, args.util_weight) 
        if args.quiet:
            print(",".join(gpus))
        else:
            print(f"Selected GPUs: {','.join(gpus)}")

if __name__ == "__main__":
    main()