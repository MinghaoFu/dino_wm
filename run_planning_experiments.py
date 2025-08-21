#!/usr/bin/env python3
import subprocess
import json
import os
import time
# import pandas as pd

def run_experiment(goal_H):
    """Run planning experiment for a specific goal_H value"""
    cmd = [
        "/home/minghao/miniconda3/envs/wm310/bin/python", 
        "plan_robomimic_align.py", 
        f"goal_H={goal_H}"
    ]
    
    print(f"Running experiment with goal_H={goal_H}...")
    log_file = f"planning_align_gH{goal_H}.log"
    
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    if result.returncode == 0:
        # Read the results JSON file
        results_file = f"planning_results_align_gH{goal_H}.json"
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                data = json.load(f)
            print(f"✓ goal_H={goal_H}: Success rate={data['success_rate']:.3f}, Mean state dist={data['mean_state_dist']:.3f}")
            return data
        else:
            print(f"✗ goal_H={goal_H}: Results file not found")
            return None
    else:
        print(f"✗ goal_H={goal_H}: Experiment failed")
        return None

def main():
    """Run all planning experiments and generate comparison table"""
    goal_H_values = [5, 10, 15, 20, 25, 30]
    results = []
    
    print("=== Running Planning Experiments ===")
    
    for goal_H in goal_H_values:
        result = run_experiment(goal_H)
        if result:
            results.append({
                "goal_H": goal_H,
                "success_rate": result["success_rate"],
                "mean_state_dist": result["mean_state_dist"],
                "n_evaluations": result["n_evaluations"],
                "model_dimensions": result["model_config"]["dimensions"],
                "visual_dim": result["model_config"]["visual_dim"],
                "proprio_dim": result["model_config"]["proprio_dim"],
                "action_dim": result["model_config"]["action_dim"]
            })
        else:
            results.append({
                "goal_H": goal_H,
                "success_rate": None,
                "mean_state_dist": None,
                "n_evaluations": 0,
                "model_dimensions": "N/A",
                "visual_dim": "N/A",
                "proprio_dim": "N/A",
                "action_dim": "N/A"
            })
    
    # Create comparison table
    print("\n=== Planning Results Comparison Table ===")
    print("Model Type: Alignment Model (128D projected DINO)")
    print(f"Model Dimensions: {results[0]['model_dimensions'] if results else 'N/A'}")
    print()
    print(f"{'goal_H':<8} {'success_rate':<12} {'mean_state_dist':<16} {'n_evaluations':<14}")
    print("-" * 60)
    for result in results:
        goal_H = result['goal_H']
        success_rate = f"{result['success_rate']:.3f}" if result['success_rate'] is not None else "N/A"
        mean_dist = f"{result['mean_state_dist']:.3f}" if result['mean_state_dist'] is not None else "N/A"
        n_evals = result['n_evaluations']
        print(f"{goal_H:<8} {success_rate:<12} {mean_dist:<16} {n_evals:<14}")
    
    # Save detailed results
    results_summary = {
        "model_type": "alignment_model",
        "model_dimensions": results[0]["model_dimensions"] if results else "N/A",
        "visual_dim": results[0]["visual_dim"] if results else "N/A", 
        "proprio_dim": results[0]["proprio_dim"] if results else "N/A",
        "action_dim": results[0]["action_dim"] if results else "N/A",
        "experiments": results,
        "summary_table": results
    }
    
    # Save to JSON
    with open("planning_experiments_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    # Save to CSV format
    with open("planning_experiments_summary.csv", "w") as f:
        f.write("goal_H,success_rate,mean_state_dist,n_evaluations,model_dimensions\n")
        for result in results:
            f.write(f"{result['goal_H']},{result['success_rate']},{result['mean_state_dist']},{result['n_evaluations']},{result['model_dimensions']}\n")
    
    print(f"\n✓ Results saved to:")
    print(f"  - planning_experiments_summary.json")
    print(f"  - planning_experiments_summary.csv")
    
    return results

if __name__ == "__main__":
    main()