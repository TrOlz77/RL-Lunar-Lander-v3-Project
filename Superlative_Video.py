from __future__ import annotations
import os
import re
import numpy as np
import glob
from utils import setup_environment, display_best_policy
from Semi_Grad_SARSA import HP as NstepHP
from Lambda_SARSA import HP as SarsaLambdaHP
from REINFORCE import HP as ReinforceHP

def parse_hp_file(filepath: str) -> dict:
    """
    Simple regex parser to extract necessary parameters from the BestHP text files.
    The files contain the __repr__ of the HP dataclass.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    params = {}
    # Extract integer parameters critical for Tile Coding setup
    for key in ['iht_size', 'n_tilings', 'Sintervals']:
        match = re.search(fr"{key}=(\d+)", content)
        if match:
            params[key] = int(match.group(1))
    
    # Extract float parameters (useful for logging/validation)
    for key in ['gamma', 'alpha_a', 'alpha_b', 'lmbda', 'epsilon']:
        match = re.search(fr"{key}=([\d\.eE\-]+)", content)
        if match:
            params[key] = float(match.group(1))
            
    return params

def main():
    """
    Loads the best saved models and runs the LunarLander environment 
    to visualize or record the agent's performance.
    """
    # Define the algorithms to visualize and their corresponding HP classes
    algorithms = [
        ("Semi_Grad_SARSA", NstepHP),
        ("Lambda_SARSA", SarsaLambdaHP),
        ("REINFORCE", ReinforceHP)
    ]

    print("Starting visualization of superlative policies...")

    for alg_name, hp_class in algorithms:
        # Find latest weights
        weights_pattern = f"./Saved Models/Best Weights_{alg_name}_*.npy"
        weight_files = sorted(glob.glob(weights_pattern))
        
        # Find latest HP
        hp_pattern = f"./Results/Superlative Hyperparameters/Superlative Hyperparameters_{alg_name}_*.txt"
        hp_files = sorted(glob.glob(hp_pattern))

        if not weight_files:
            print(f"\n[!] Skipping {alg_name}: weights not found matching {weights_pattern}")
            continue

        weights_path = weight_files[-1]
        print(f"\n--- Visualizing: {alg_name} ---")
        
        # Initialize default HP and overwrite with tuned values if available
        hp = hp_class()
        hp.alg_name = alg_name # Force consistency for video file naming and labeling
        if hp_files:
            hp_path = hp_files[-1]
            tuned_params = parse_hp_file(hp_path)
            for key, val in tuned_params.items():
                setattr(hp, key, val)
            print(f"Loaded tuned configuration: iht_size={hp.iht_size}, tilings={hp.n_tilings}, bins={hp.Sintervals}")

        # Load Weights and Setup Environment
        Q = np.load(weights_path)
        env, phi, _, _ = setup_environment(hp)
        
        # Run visualization: set record_video=True to save MP4s to ./Results/Superlative Video
        display_best_policy(env, Q, phi, hp, num_reps_show=3, record_video=True, video_folder="./Results/Superlative Video")
        if True: # Local scope check for record_video
            print(f"Video saved as: ./Results/Superlative Video/{alg_name}-episode-0.mp4")

if __name__ == "__main__":
    main()