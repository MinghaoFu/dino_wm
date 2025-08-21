#!/usr/bin/env python3
"""
Script to analyze planning results from nohup.out file.
Extracts state_dist arrays and success rates for different experiments.
"""

import re
import numpy as np
import json
from collections import defaultdict

def parse_nohup_file(filepath):
    """Parse nohup.out file to extract planning results."""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find all planning result directories to identify experiments
    planning_dirs = re.findall(r'Planning result saved dir: (.+)', content)
    
    # Extract goal_H values from directory names
    goal_h_values = []
    for dir_path in planning_dirs:
        # Look for gH pattern in directory name
        gH_match = re.search(r'gH(\d+)', dir_path)
        if gH_match:
            goal_h_values.append(int(gH_match.group(1)))
        else:
            goal_h_values.append(None)  # Unknown goal_H
    
    # Find all MPC iterations
    mpc_iter_positions = [(m.start(), m.group()) for m in re.finditer(r'MPC iter \d+ Eval -------', content)]
    
    # Find all success rate and state_dist patterns
    success_patterns = list(re.finditer(r'Success rate:\s+([\d\.]+)', content))
    state_dist_patterns = list(re.finditer(r"'state_dist': array\((\[[^\]]+\])", content, re.DOTALL))
    
    experiments = []
    
    # Group results by experiment (based on planning directories)
    experiment_boundaries = []
    for i, dir_path in enumerate(planning_dirs):
        start_pos = content.find(dir_path)
        experiment_boundaries.append(start_pos)
    
    # Add end of file as final boundary
    experiment_boundaries.append(len(content))
    
    for exp_idx in range(len(planning_dirs)):
        start_pos = experiment_boundaries[exp_idx]
        end_pos = experiment_boundaries[exp_idx + 1] if exp_idx + 1 < len(experiment_boundaries) else len(content)
        
        exp_content = content[start_pos:end_pos]
        goal_h = goal_h_values[exp_idx]
        
        # Find MPC iterations in this experiment
        exp_mpc_iters = []
        exp_success_rates = []
        exp_state_dists = []
        
        # Extract all success rates and state_dist arrays in this experiment
        success_matches = re.finditer(r'Success rate:\s+([\d\.]+)', exp_content)
        state_dist_matches = re.finditer(r"'state_dist': array\((\[[^\]]+\])", exp_content, re.DOTALL)
        
        success_rates = [float(m.group(1)) for m in success_matches]
        
        # Parse state_dist arrays
        state_dists = []
        for m in re.finditer(r"'state_dist': array\((\[[^\]]+\])", exp_content, re.DOTALL):
            try:
                # Clean up the array string and convert to numpy array
                array_str = m.group(1)
                # Remove extra whitespace and format for parsing
                array_str = re.sub(r'\s+', ' ', array_str.strip())
                array_str = array_str.replace('[', '').replace(']', '')
                values = [float(x.rstrip(',')) for x in array_str.split() if x.strip() and x.strip() != ',']
                state_dists.append(np.array(values))
            except Exception as e:
                print(f"Error parsing state_dist array: {e}")
                continue
        
        # Find MPC iteration markers in this experiment
        mpc_markers = list(re.finditer(r'MPC iter (\d+) Eval -------', exp_content))
        
        experiments.append({
            'experiment_id': exp_idx,
            'goal_H': goal_h,
            'directory': planning_dirs[exp_idx],
            'num_mpc_iterations': len(mpc_markers),
            'success_rates': success_rates,
            'state_dists': state_dists,
            'num_success_entries': len(success_rates),
            'num_state_dist_entries': len(state_dists)
        })
    
    return experiments

def analyze_experiments(experiments):
    """Analyze the extracted experiment data."""
    
    analysis = {}
    
    for exp in experiments:
        goal_h = exp['goal_H'] or 'unknown'
        exp_id = exp['experiment_id']
        
        if len(exp['state_dists']) == 0:
            continue
            
        # Calculate statistics for state distances
        all_state_dists = np.concatenate(exp['state_dists'])
        avg_state_dist = np.mean(all_state_dists)
        std_state_dist = np.std(all_state_dists)
        
        # Calculate final success rate (last entry)
        final_success_rate = exp['success_rates'][-1] if exp['success_rates'] else 0
        
        # Calculate average state distance per iteration
        iter_avg_distances = []
        for state_dist_array in exp['state_dists']:
            iter_avg_distances.append(np.mean(state_dist_array))
        
        analysis[f'experiment_{exp_id}_gH{goal_h}'] = {
            'goal_H': goal_h,
            'experiment_id': exp_id,
            'directory': exp['directory'],
            'final_success_rate': final_success_rate,
            'overall_avg_state_dist': float(avg_state_dist),
            'overall_std_state_dist': float(std_state_dist),
            'num_planning_iterations': len(exp['state_dists']),
            'avg_state_dist_per_iteration': [float(x) for x in iter_avg_distances],
            'success_rate_progression': exp['success_rates'],
            'num_mpc_iterations': exp['num_mpc_iterations'],
            'total_state_samples': len(all_state_dists)
        }
    
    return analysis

def main():
    filepath = '/home/minghao/workspace/dino_wm/nohup.out'
    
    print("Parsing nohup.out file...")
    experiments = parse_nohup_file(filepath)
    
    print(f"Found {len(experiments)} experiments")
    for exp in experiments:
        print(f"  Experiment {exp['experiment_id']}: goal_H={exp['goal_H']}, "
              f"{exp['num_success_entries']} success entries, "
              f"{exp['num_state_dist_entries']} state_dist arrays")
    
    print("\nAnalyzing experiments...")
    analysis = analyze_experiments(experiments)
    
    print("\n" + "="*80)
    print("PLANNING RESULTS ANALYSIS")
    print("="*80)
    
    for key, data in analysis.items():
        print(f"\n{key}:")
        print(f"  Goal Horizon (goal_H): {data['goal_H']}")
        print(f"  Final Success Rate: {data['final_success_rate']:.3f}")
        print(f"  Overall Average State Distance: {data['overall_avg_state_dist']:.3f} ± {data['overall_std_state_dist']:.3f}")
        print(f"  Number of Planning Iterations: {data['num_planning_iterations']}")
        print(f"  Number of MPC Iterations: {data['num_mpc_iterations']}")
        print(f"  Total State Samples: {data['total_state_samples']}")
        
        if len(data['avg_state_dist_per_iteration']) > 0:
            print(f"  Average State Distance per Iteration:")
            for i, avg_dist in enumerate(data['avg_state_dist_per_iteration']):
                print(f"    Iteration {i}: {avg_dist:.3f}")
        
        print(f"  Success Rate Progression: {[f'{sr:.3f}' for sr in data['success_rate_progression'][:10]]}{'...' if len(data['success_rate_progression']) > 10 else ''}")
    
    # Save detailed results to JSON
    output_file = '/home/minghao/workspace/dino_wm/planning_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()