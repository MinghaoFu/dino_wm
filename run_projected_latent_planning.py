#!/usr/bin/env python3
"""
Batch evaluation script for projected latent planning
Evaluates H=5,10,15,20 with 10 seeds each as requested
"""

import os
import json
import subprocess
import sys
from pathlib import Path
import argparse

def run_planning_evaluation(ckpt_path, projected_dim=64, horizons=[5, 10, 15, 20], n_seeds=10):
    """
    Run planning evaluation for multiple horizons
    
    Args:
        ckpt_path: Path to model checkpoint directory
        projected_dim: Projected latent dimension (e.g., 64, 128)
        horizons: List of planning horizons to evaluate
        n_seeds: Number of evaluation seeds per horizon
    """
    
    results = {}
    
    print(f"🚀 Starting Projected Latent Planning Evaluation")
    print(f"📊 Model: {projected_dim}D projected latent representation")
    print(f"📍 Checkpoint: {ckpt_path}")
    print(f"🎯 Horizons: {horizons}")
    print(f"🌱 Seeds per horizon: {n_seeds}")
    print("=" * 60)
    
    for horizon in horizons:
        print(f"\n🔄 Evaluating Horizon H={horizon}")
        print("-" * 40)
        
        # Construct command
        cmd = [
            "python", "plan_projected_latent.py",
            "--config-name=plan_projected_latent",
            f"ckpt_base_path={ckpt_path}",
            f"projected_dim={projected_dim}",
            f"goal_H={horizon}",
            f"n_evals={n_seeds}",
            f"seed=42",
        ]
        
        try:
            print(f"Running: {' '.join(cmd)}")
            
            # Run planning
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/home/minghao/workspace/dino_wm"
            )
            
            if result.returncode == 0:
                print(f"✅ H={horizon} completed successfully")
                
                # Try to extract results from output
                success_rate = "N/A"
                for line in result.stdout.split('\n'):
                    if "success_rate" in line.lower():
                        # Try to extract success rate from logs
                        pass
                
                results[f"H_{horizon}"] = {
                    "horizon": horizon,
                    "n_seeds": n_seeds,
                    "projected_dim": projected_dim,
                    "status": "completed",
                    "success_rate": success_rate,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
            else:
                print(f"❌ H={horizon} failed with return code {result.returncode}")
                print(f"Error output: {result.stderr}")
                
                results[f"H_{horizon}"] = {
                    "horizon": horizon,
                    "n_seeds": n_seeds,
                    "projected_dim": projected_dim,
                    "status": "failed",
                    "error": result.stderr,
                    "stdout": result.stdout
                }
                
        except Exception as e:
            print(f"❌ H={horizon} failed with exception: {str(e)}")
            results[f"H_{horizon}"] = {
                "horizon": horizon,
                "status": "error",
                "error": str(e)
            }
    
    # Save results
    results_file = f"projected_latent_planning_results_{projected_dim}d.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 Evaluation Summary")
    print("=" * 60)
    print(f"{'Horizon':<10} {'Status':<12} {'Success Rate':<15}")
    print("-" * 37)
    
    for horizon in horizons:
        res = results.get(f"H_{horizon}", {})
        status = res.get("status", "unknown")
        success_rate = res.get("success_rate", "N/A")
        print(f"H={horizon:<7} {status:<12} {success_rate:<15}")
    
    print("=" * 60)
    print(f"📄 Results saved to: {results_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run projected latent planning evaluation")
    parser.add_argument("--ckpt_path", type=str, 
                       default="/home/minghao/workspace/dino_wm/outputs/2025-09-09/07-57-28",
                       help="Path to model checkpoint directory")
    parser.add_argument("--projected_dim", type=int, default=64,
                       help="Projected latent dimension")
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 15, 20],
                       help="Planning horizons to evaluate")
    parser.add_argument("--n_seeds", type=int, default=10,
                       help="Number of evaluation seeds per horizon")
    
    args = parser.parse_args()
    
    # Verify checkpoint path exists
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        print(f"❌ Error: Checkpoint path does not exist: {ckpt_path}")
        sys.exit(1)
    
    # Run evaluation
    results = run_planning_evaluation(
        ckpt_path=str(ckpt_path),
        projected_dim=args.projected_dim,
        horizons=args.horizons,
        n_seeds=args.n_seeds
    )
    
    print("\n🏁 Planning evaluation completed!")

if __name__ == "__main__":
    main()