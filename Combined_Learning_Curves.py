# =============================================================================
# Library Imports
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t
import os
import re
import glob

# =============================================================================
# Plotting Functions
# =============================================================================
def plot_combined_learning_curves(alg_names, results_dir="./Results"):
    """
    Loads saved learning curve CSVs for multiple algorithms and plots them on one graph.
    """
    plt.figure(figsize=(12, 7))
    
    # Consistent colors for each algorithm
    colors = {
        'Semi_Grad_SARSA': 'tab:red',
        'Lambda_SARSA': 'tab:purple',
        'REINFORCE': 'tab:blue'
    }
    
    for alg in alg_names:
        # Look for the latest Learning Curve file for this algorithm
        pattern = os.path.join(results_dir, f"Learning Curve_{alg}_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"Skipping {alg}: No files matching {pattern} found.")
            continue
            
        filepath = files[-1]
        # Load training history data: rep, episode, mean, hw
        data = pd.read_csv(filepath)
        
        # Dynamically determine Z (replications) from the data
        Z = int(data.iloc[:, 0].max() + 1)
        
        # Reshape the 'mean' column (index 2) to (Z, num_test_points)
        means_raw = data.iloc[:, 2].values
        # Logic Guard: Ensure data is divisible by Z for reshaping
        num_test_points = int(len(means_raw) // Z)
        if len(means_raw) % Z != 0:
            means_raw = means_raw[:Z * num_test_points]
        TestEETDR = np.reshape(means_raw, (Z, num_test_points))
        
        # Extract x-axis (episodes) from the first replication
        num_points = TestEETDR.shape[1]
        xs = data.iloc[:num_points, 1].values
        
        # Calculate aggregate statistics (Mean and 95% CI) across the Z replications
        avg_eetdr = np.mean(TestEETDR, axis=0)
        avg_se = np.std(TestEETDR, axis=0, ddof=1) / np.sqrt(Z)
        avg_hw = t.ppf(1 - 0.05 / 2, Z - 1) * avg_se
        
        # Extract key tuned hyperparameters from the saved BestHP text file for the legend
        hp_suffix = ""
        hp_pattern = os.path.join(results_dir, "Superlative Hyperparameters", f"Superlative Hyperparameters_{alg}_*.txt")
        hp_files = sorted(glob.glob(hp_pattern))
        if hp_files:
            hp_path = hp_files[-1]
            with open(hp_path, 'r') as f:
                hp_content = f.read()
                # Extract values using regex to handle the dataclass repr string
                a = re.search(r"alpha_a=([\d\.eE\-]+)", hp_content)
                ab = re.search(r"alpha_b=([\d\.eE\-]+)", hp_content)
                n = re.search(r"n_steps=(\d+)", hp_content)
                l = re.search(r"lambd_a=([\d\.eE\-]+)", hp_content)
                til = re.search(r"n_tilings=(\d+)", hp_content)
                wa = re.search(r"alpha_w_a=([\d\.eE\-]+)", hp_content)
                wb = re.search(r"alpha_w_b=([\d\.eE\-]+)", hp_content)
                g = re.search(r"gamma=([\d\.eE\-]+)", hp_content)

                # Adjust labels for policy gradient parameters (alpha_theta for REINFORCE)
                a_label = r"$\alpha_\theta$" if "REINFORCE" in alg else r"$\alpha_a$"
                hp_parts = [f"{a_label}={a.group(1)}" if a else None, f"b={ab.group(1)}" if ab else None]
                if n: hp_parts.append(f"n={n.group(1)}")
                if wa: 
                    hp_parts.append(f"$\\alpha_{{w,a}}$={wa.group(1)}" + (f", b={wb.group(1)}" if wb else ""))
                if l: hp_parts.append(f"$\\lambda$={l.group(1)}")
                if g: hp_parts.append(f"$\\gamma$={g.group(1)}")
                hp_parts.append(f"tilings={til.group(1)}" if til else None)
                hp_suffix = f" ({', '.join([p for p in hp_parts if p])})"

        color = colors.get(alg, None)
        plt.plot(xs, avg_eetdr, label=f'Superlative {alg}{hp_suffix}', linewidth=2, color=color)
        plt.fill_between(xs, avg_eetdr - avg_hw, avg_eetdr + avg_hw, alpha=0.15, color=color)

    # Visual aids for direct comparison
    plt.axhline(y=200, color='forestgreen', linestyle='--', alpha=0.4, label='Solved Threshold (200)')
    plt.ylim(-600, 300) 

    plt.xlabel('Episode')
    plt.ylabel('Estimated Expected Total Discounted Reward (EETDR)')
    plt.title('Algorithm Comparison: Superlative Learning Curves (95% CI)')
    plt.legend(loc='lower right', fontsize='small', framealpha=0.8)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    save_path = os.path.join(results_dir, "Combined_Superlative_Learning_Curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to {save_path}")
    plt.show()

def generate_executive_summary(alg_names, results_dir="./Results", raw_results_dir="./Raw Results"):
    """
    Aggregates the best design run results from all algorithms into a single comparison table
    for the project's Executive Summary.
    """
    summary_data = []
    for alg in alg_names:
        pattern = os.path.join(raw_results_dir, f"Raw Results_{alg}_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            continue
            
        # Load the latest raw results for this algorithm
        df = pd.read_csv(files[-1])
        
        # Find the configuration with the highest Algorithm Score
        best_idx = df['Algorithm Score'].idxmax()
        best_row = df.loc[best_idx]
        
        summary_data.append({
            'Algorithm': alg,
            'Best Run ID': best_row['Run Index'],
            'AlgScore': best_row['Algorithm Score'],
            'SP95LB (Superlative)': best_row['SP95LB'],
            'Mean Max EETDR': best_row['Mean Max EETDR'],
            'Mean Time-Avg EETDR': best_row['Mean Time-Avg EETDR'],
            'Wallclock (s)': best_row['Wallclock']
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(results_dir, "Executive_Summary_Results.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Executive Summary table saved to {summary_path}")

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # These names must match the alg_name defined in each algorithm's HP class exactly
    algorithms = [
        "Semi_Grad_SARSA", 
        "Lambda_SARSA",
        "REINFORCE"
    ]
    plot_combined_learning_curves(algorithms)
    generate_executive_summary(algorithms)