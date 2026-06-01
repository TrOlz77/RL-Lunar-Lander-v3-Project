# =============================================================================
# Library Imports
# =============================================================================
from __future__ import annotations

import os
import time
import random
import hashlib
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from math import floor
import statsmodels.api as sm

from datetime import datetime
from statsmodels.formula.api import ols
from scipy.stats import t
from scipy.stats.qmc import LatinHypercube
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from joblib import Parallel, delayed

# =============================================================================
# Seeding, set all seeds that may be used to a global seed of 42
# =============================================================================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# =============================================================================
# Environment
# =============================================================================
class IHT:
    """
    Index Hash Table (IHT) for Tile Coding.

    This structure maps high-dimensional coordinates (tuples) to a unique index 
    within a fixed-size range [0, size-1]. It is a critical component for 
    representing large or continuous state spaces with a fixed-size feature 
    vector, supporting linear value function approximation.
    """
    def __init__(self, sizeval):
        """
        Initializes the IHT with a fixed capacity.

        Args:
            sizeval (int): The total number of available indices (the size of 
                           the hash table and resulting feature vector).
        """
        self.size = sizeval                        
    
    def getindex(self, obj):
        """
        Retrieves the index for a given coordinate tuple using deterministic hashing.

        This implementation uses a hash-modulo approach rather than a traditional 
        mapping dictionary. This ensures that state features map to the same 
        indices across different sessions without needing to save the IHT state, 
        facilitating easier model loading/sharing. Collisions are possible but 
        mitigated by the large hash space.

        Args:
            obj (tuple): The high-dimensional coordinate tuple to index.

        Returns:
            int: The index for the coordinate.
        """
        h = hashlib.md5(str(obj).encode())
        return int(h.hexdigest(), 16) % self.size

def get_tiles(iht, num_tilings, floats, ints=[]):
    """Returns num-tilings tile indices for the given floats and ints."""
    qfloats = [floor(f * num_tilings) for f in floats]
    Tiles = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(iht.getindex(tuple(coords)))
    return Tiles

class TileDiscretizer:
    """
    Tile Coder for Value Function Approximation.
    Uses multiple overlapping tilings and hashing (IHT) to represent state.
    """

    def __init__(self, iht_size: int, n_tilings: int, low: np.ndarray, high: np.ndarray, n_bins: int):
        """
        Args:
            iht_size: Total size of the hash table (number of features).
            n_tilings: Number of overlapping tilings.
            low: Manually defined lower bounds for clipping/normalization.
            high: Manually defined upper bounds for clipping/normalization.
            n_bins: Desired resolution (number of tiles across the range).
        """
        self.iht = IHT(iht_size)
        self.n_tilings = n_tilings
        self.low = np.array(low)
        self.high = np.array(high)
        # Scaling factor: maps the range [low, high] to [0, n_bins]
        # We divide by n_bins because the tile() function internalizes tiling shifts
        self.scales = n_bins / (self.high - self.low) # Map [low, high] to [0, n_bins]

    def __call__(self, s):
        """
        Maps continuous state to a list of active feature indices.
        """
        s_clipped = np.clip(s, self.low, self.high)
        s_scaled = (s_clipped - self.low) * self.scales
        return get_tiles(self.iht, self.n_tilings, s_scaled)

class VideoLabelWrapper(gym.Wrapper):
    """
    Gymnasium environment wrapper that overlays the algorithm name 
    on the rendered RGB frames for visual identification in recorded videos.
    """
    def __init__(self, env, label):
        super().__init__(env)
        self.label = label

    def render(self):
        frame = self.env.render()
        # Only attempt to draw if we are in rgb_array mode and have a valid frame
        if self.render_mode == 'rgb_array' and isinstance(frame, np.ndarray):
            try:
                import cv2
                # Draw a dark background box for the label to ensure readability
                cv2.rectangle(frame, (5, 5), (280, 45), (0, 0, 0), -1)
                cv2.putText(frame, self.label, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)
            except ImportError:
                pass # Fallback if cv2 (opencv-python) is not installed
        return frame

def setup_environment(hp):
    """
    Initializes the environment and TileCoder with appropriate bounds.
    """
    env = gym.make(hp.env_name)
    env.action_space.seed(GLOBAL_SEED)
    
    # Custom bounds for LunarLander-v3 (to handle infinite observation space)
    if hp.env_name == "LunarLander-v3":
        # State: x, y, vx, vy, angle, ang_v, leg1, leg2
        low =  [-1.5, -0.5, -5.0, -5.0, -3.14, -5.0, 0.0, 0.0]
        high = [ 1.5,  1.5,  5.0,  5.0,  3.14,  5.0, 1.0, 1.0]
    else:
        # Default for MountainCar or others with finite bounds
        low, high = env.observation_space.low, env.observation_space.high

    iht_size = hp.iht_size
    n_tilings = hp.n_tilings

    phi = TileDiscretizer(iht_size, n_tilings, low, high, hp.Sintervals) # hp.Sintervals is used as n_bins

    # Dimensions of the feature-action weight matrix
    SAsize = (iht_size, env.action_space.n)
    num_actions = env.action_space.n

    return env, phi, SAsize, num_actions



# =============================================================================
# Latin Hypercube
# =============================================================================
def run_lhd_experiment(hp_class, run_episode_fn, hparam_names, factor_names, 
                       n_runs=50, n_cores=16, seed=GLOBAL_SEED,
                       z_search=None, m_search=None):
    """
    Executes a Latin Hypercube Design experiment in parallel.
    """
    sampler = LatinHypercube(len(hparam_names), scramble=False, optimization="lloyd", seed=seed)
    factor_table = sampler.random(n=n_runs)
    
    # Instantiate once to get the algorithm name
    alg_name = hp_class().alg_name
    
    print(f"\nInitializing parallel LHD experiment with {n_runs} runs...")
    start_time = time.time()

    # Create directories if they don't exist
    os.makedirs("./Saved Models", exist_ok=True)
    os.makedirs("./Raw Results", exist_ok=True)
    os.makedirs("./Results/Training Plots", exist_ok=True)
    os.makedirs("./Results/LHD Plots", exist_ok=True)
    os.makedirs("./Results/Top 10", exist_ok=True)
    os.makedirs("./Results/Top 5", exist_ok=True)
    os.makedirs("./Results/Superlative Hyperparameters", exist_ok=True)
    os.makedirs("./Results/Superlative Video", exist_ok=True)

    def parallel_task(run_idx, factors):
        hp = hp_class()
        for name, val in zip(hparam_names, factors):
            if name == 'n_steps':
                setattr(hp, name, int(val * 15) + 1)
            elif name == 'n_tilings':
                # Range [2, 32]
                setattr(hp, name, int(val * 30) + 2)
            elif name == 'Sintervals':
                # Range [4, 20]
                setattr(hp, name, int(val * 16) + 4)
            elif name in ['alpha_a', 'alpha_w_a']:
                # For REINFORCE, we need much smaller learning rates. 
                # Using log-uniform scaling for learning rates is often better.
                if "REINFORCE" in hp.alg_name:
                    setattr(hp, name, 10 ** (val * 3 - 4)) # Adjusted Range [1e-4, 1e-1]
                else:
                    setattr(hp, name, val * 1.5) # Standard scaling for SARSA
            else:
                setattr(hp, name, val)
        
        # Override Z and M for faster search if requested
        if z_search is not None: hp.Z = z_search
        if m_search is not None: hp.M = m_search

        env, phi, SAsize, num_actions = setup_environment(hp)
        results = run_experiment(env, phi, SAsize, hp, run_episode_fn, verbose=False)
        
        return (run_idx, results['supETDR_ho'], results['supETDRhw_ho'], results['SP95LB_ho'],
                results['meanMaxTestEETDR'], results['maxTestME'], 
                results['meanAULC'], results['meAULC'], 
                results['wallclock'], results['AlgScore'], hp)

    # Execute LHD runs in parallel. Wrapping the generator allows tqdm to track completion.
    print(f"Dispatching {n_runs} configurations to {n_cores} cores...")
    with tqdm(total=n_runs, desc=f"LHD {alg_name}") as pbar:
        results_generator = Parallel(n_jobs=n_cores, return_as="generator")(
            delayed(parallel_task)(i, factor_table[i]) for i in range(n_runs)
        )
        results_list = []
        for result in results_generator:
            results_list.append(result)
            pbar.update(1)

    # Extract metrics and result objects
    results_metrics = np.array([r[:9] for r in results_list])
    scores = np.array([r[9] for r in results_list])

    # Identify the configuration with the highest algorithm score
    best_idx = np.argmax(scores)
    best_hp = results_list[best_idx][10]
    ts = datetime.now().strftime('%m_%d_%Y_%H_%M')
    
    print(f"\nSearch complete. Re-running superlative configuration for final validation...")
    # Re-run the best configuration once in the main process to get the full results/weights
    env_best, phi_best, SA_best, _ = setup_environment(best_hp)
    best_res_data = run_experiment(env_best, phi_best, SA_best, best_hp, run_episode_fn, verbose=True)
    
    # Plot and save learning curve for the superlative policy
    plot_learning_curve(best_hp, best_res_data, f"LHD Plot_{alg_name}_{ts}", folder="./Results/LHD Plots", is_superlative=True)

    # Save superlative training run data for aggregate comparison plotting
    lc_data = np.array(best_res_data['GzmTest'])
    np.savetxt(f"./Results/Learning Curve_{alg_name}_{ts}.csv", lc_data, delimiter=",", 
               header="rep,episode,mean,hw", comments="")
    
    # Save the superlative policy (Q-table)
    np.save(f"./Saved Models/Best Weights_{alg_name}_{ts}.npy", best_res_data['Q'])
    
    # Save superlative hyperparameters for record keeping
    with open(f"./Results/Superlative Hyperparameters/Superlative Hyperparameters_{alg_name}_{ts}.txt", "w") as f:
        f.write(str(best_hp))

    # Combine factor table with results for full CSV export
    full_data = np.column_stack((results_metrics[:, 0], factor_table, results_metrics[:, 1:9], scores))
    column_names = ["Run Index"] + factor_names + \
                   ["Superlative EETDR", "Superlative 95HW", "SP95LB", "Mean Max EETDR", "Mean Max 95ME", 
                    "Mean Time-Avg EETDR", "Mean Time-Avg 95ME", "Wallclock", "Algorithm Score"]
    
    base_filename = f"Raw Results_{alg_name}_{ts}"
    filename = f"./Raw Results/{base_filename}.csv"
    output_table = np.row_stack((column_names, full_data))
    np.savetxt(filename, output_table, delimiter=",", fmt="%s")
    
    # --- Specialized Summary Table Generation ---
    offset = len(factor_names)

    # 1. Generate Top 10 Table (Hyperparameter Optimization Level - Robustness)
    # Sorted by AlgScore (offset + 9)
    t10_sort_idx = offset + 9
    t10_sorted = full_data[np.argsort(full_data[:, t10_sort_idx].astype(float))[::-1]]
    t10_subset = t10_sorted[:10]
    
    t10_headers = ["Rank", "Run Index"] + factor_names + \
                  ["AlgScore", "SP95LB", "Mean Max EETDR", "Max 95ME", "Mean Time-Avg EETDR", "Time-Avg 95ME", "Wallclock"]
    
    # Column Mapping: RunIdx(0), Factors(1..L), AlgScore(L+9), SP95LB(L+3), Max(L+4), MaxME(L+5), TimeAvg(L+6), TAME(L+7), Clock(L+8)
    t10_indices = [0] + list(range(1, offset + 1)) + [offset + 9, offset + 3, offset + 4, offset + 5, offset + 6, offset + 7, offset + 8]
    
    t10_ranks = np.arange(1, 11).reshape(-1, 1)
    t10_final = np.row_stack((t10_headers, np.column_stack((t10_ranks, t10_subset[:, t10_indices]))))
    t10_name = f"./Results/Top 10/Top 10_{alg_name}_{ts}.csv"
    np.savetxt(t10_name, t10_final, delimiter=",", fmt="%s")

    # 2. Generate Top 5 Table (Evaluation Level - Absolute Champions)
    # Sorted by SP95LB (offset + 3)
    t5_sort_idx = offset + 3
    t5_sorted = full_data[np.argsort(full_data[:, t5_sort_idx].astype(float))[::-1]]
    t5_subset = t5_sorted[:5]
    
    t5_headers = ["Rank", "Run Index", "Sup. EETDR (Stream-3)", "95% HW", "SP95LB"]
    # Map: Run Index (0), Superlative EETDR (L+1), Superlative HW (L+2), SP95LB (L+3)
    t5_indices = [0, offset + 1, offset + 2, offset + 3]
    t5_ranks = np.arange(1, 6).reshape(-1, 1)
    t5_final = np.row_stack((t5_headers, np.column_stack((t5_ranks, t5_subset[:, t5_indices]))))
    t5_name = f"./Results/Top 5/Top 5_{alg_name}_{ts}.csv"
    np.savetxt(t5_name, t5_final, delimiter=",", fmt="%s")

    print(f"\nCompleted LHD experiment in {time.time() - start_time:.1f}s.")
    print(f"Full results saved to {filename}")
    print(f"Top 10 (Robustness) saved to {t10_name}")
    print(f"Top 5 (Champions) saved to {t5_name}")

    return factor_table, scores, hparam_names


def analyze_lhd_results(factor_table, scores, hparam_names, alg_name):
    """
    Performs Response Surface Model (RSM) fit and ANOVA on LHD results.
    """
    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(factor_table)
    
    feature_names = [name.replace(' ', '_').replace('^', '_pow_').replace('*', '_times_')
                     for name in poly.get_feature_names_out(input_features=hparam_names)]
    
    df = pd.DataFrame(X_poly, columns=feature_names)
    df['AlgScore'] = scores
    
    predictors = '+'.join(df.columns[1:-1])
    formula = f'AlgScore ~ {predictors}' # type: ignore
    
    model = ols(formula, data=df).fit()
    print("\n\n" + "="*80)
    print("RSM Regression Summary")
    print("="*80)
    print(model.summary())

    anova_results = sm.stats.anova_lm(model, typ=2)
    print("\n\n" + "="*80)
    print("ANOVA Table")
    print("="*80)
    print(anova_results)

    # Save combined Analysis (Regression Summary + ANOVA Table) to a single file
    ts = datetime.now().strftime('%m_%d_%Y_%H_%M')
    analysis_filename = f"./Results/Superlative Hyperparameters/Analysis_{alg_name}_{ts}.txt"
    with open(analysis_filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"RSM REGRESSION SUMMARY: {alg_name}\n")
        f.write("="*80 + "\n")
        f.write(model.summary().as_text())
        f.write("\n\n" + "="*80 + "\n")
        f.write(f"ANOVA TABLE: {alg_name}\n")
        f.write("="*80 + "\n")
        f.write(anova_results.to_string())

    print(f"LHD Analysis results saved to {analysis_filename}")

    return model, anova_results # type: ignore


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_policy(env, Q, phi, num_reps, seed_mult=1):
    """
    Performs greedy evaluation of a Q-table over multiple episodes.

    Args:
        env: The Gymnasium environment.
        Q: The Q-table to evaluate.
        phi: The state discretizer.
        num_reps: Number of episodes to run.
        seed_mult: Multiplier for seeds to ensure evaluation on different data streams.

    Returns:
        A tuple (mean_reward, half_width) from the evaluation trials.
    """
    num_actions = env.action_space.n
    test_data = np.zeros(num_reps)
    for rep in range(num_reps):
        terminated = truncated = False
        Gtest = 0
        state, _ = env.reset(seed=GLOBAL_SEED + seed_mult * 500000 + rep)
        while not (terminated or truncated):
            state_features = phi(state)
            q_values = Q[state_features, :].sum(axis=0)
            action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
            
            state, reward, terminated, truncated, _ = env.step(action)
            Gtest += reward
        test_data[rep] = Gtest
    return confinterval(test_data)


def run_experiment(env, phi, SAsize, hp, run_episode_fn, verbose=True):
    """
    Orchestrates the training, selection, and validation streams of an RL experiment.

    Args:
        env: The Gymnasium environment.
        phi: The state discretizer.
        SAsize: Dimensions of the state-action space.
        hp: Hyperparameter configuration object.
        run_episode_fn: The algorithm-specific function to run one training episode.

    Returns:
        A dictionary containing the best policy (Q), learning curves (GzmTest),
        wallclock time, and final validation statistics.
    """
    GzmTest = []
    tracker = BestScoreTracker(k=2 * hp.Z)
    tic = time.perf_counter()

    # Ensure output directories exist before training starts
    os.makedirs("./Saved Models", exist_ok=True)
    os.makedirs("./Raw Results", exist_ok=True)
    os.makedirs("./Results/Training Plots", exist_ok=True)
    os.makedirs("./Results/LHD Plots", exist_ok=True)
    os.makedirs("./Results/Top 10", exist_ok=True)
    os.makedirs("./Results/Top 5", exist_ok=True)
    os.makedirs("./Results/Superlative Hyperparameters", exist_ok=True)
    os.makedirs("./Results/Superlative Video", exist_ok=True)

    for z in range(hp.Z):
        # Initialize weights such that the sum (Q-value) matches the intended qinit level
        w = (hp.qinit * (hp.qrange[1] - hp.qrange[0]) + hp.qrange[0]) / hp.n_tilings * np.ones(SAsize)
        C = np.zeros(SAsize)

        if verbose: print(f"\n{hp.alg_name} Rep {z}")
        for m in range(hp.M):
            # Seed training for reproducibility
            current_episode_seed = GLOBAL_SEED + z * 100000 + m + hp.offset * 1000
            np.random.seed(current_episode_seed)
            random.seed(current_episode_seed)
            Gm = run_episode_fn(env, w, C, phi, hp, episode_seed=current_episode_seed, episode_m=m) # Pass episode_m for epsilon decay

            if m % hp.test_freq == 0:
                mean, hw = evaluate_policy(env, w, phi, hp.num_test_reps)
                GzmTest.append((z, m, mean, hw))
                tracker.update(mean, hw, w) # Update the tracker with the latest evaluation scores
                if verbose:
                    is_top = " *** New Top" if mean >= tracker.scores[0]['ETDR'] else ""
                    print(f" Episode: {m:>4}, EETDR CI: {mean:>6.1f} +/- {hw:4.1f}{is_top}")

        # Last test of current algorithm run
        mean, hw = evaluate_policy(env, w, phi, hp.num_test_reps)
        GzmTest.append((z, hp.M, mean, hw))
        tracker.update(mean, hw, w) # Pass weights 'w' to tracker

    toc = time.perf_counter()
    wallclock_time = toc - tic
    
    # Stream-2 Selection
    mean_values, hw_values = [], []
    valid_scores = [s for s in tracker.scores if s['w'] is not None]
    for score in valid_scores:
        mean, hw = evaluate_policy(env, score['w'], phi, 2 * hp.num_test_reps, seed_mult=10) # Stream 2 (Selection)
        mean_values.append(mean); hw_values.append(hw)

    ind_best = np.argmax(mean_values) if mean_values else 0
    w_best = valid_scores[ind_best]['w']

    # Stream-3 Held-out Validation (Running 100 episodes for SP95LB reporting)
    # To match project requirements for "95% lower confidence bound", we use the 95th percentile 
    # of the t-distribution (one-sided) rather than the standard two-sided 97.5th.
    test_reps_ho = 100
    ho_mean, ho_hw_two_sided = evaluate_policy(env, w_best, phi, test_reps_ho, seed_mult=20)
    ho_se = ho_hw_two_sided / t.ppf(0.975, test_reps_ho - 1)
    supETDRhw_ho = t.ppf(0.95, test_reps_ho - 1) * ho_se
    sp95lb_ho = ho_mean - supETDRhw_ho

    results = {
        'env': env, 'phi': phi, 'Q': w_best, 'GzmTest': GzmTest, # Store weights and test results
        'wallclock': wallclock_time, 'supETDR': mean_values[ind_best], 
        'supETDRhw': hw_values[ind_best], 'supETDR_ho': ho_mean, 
        'supETDRhw_ho': supETDRhw_ho, 'SP95LB_ho': sp95lb_ho, 'sup_rank': ind_best + 1
    }
    results.update(compute_metrics(hp, GzmTest))
    return results


# =============================================================================
# Utilities
# =============================================================================
def compute_metrics(hp, GzmTest):
    """
    Calculates performance metrics from experiment results.
    """
    npGzmTest = np.array(GzmTest)
    # GzmTest: (rep_z, episode_m, mean, hw)
    TestEETDR = np.reshape(npGzmTest[:, 2], (hp.Z, -1))
    
    # Max EETDR metrics
    max_vals = np.max(TestEETDR, axis=1)
    meanMaxTestEETDR = np.mean(max_vals)
    maxTestSE = np.std(max_vals, ddof=1) / np.sqrt(hp.Z)
    maxTestME = t.ppf(0.95, hp.Z - 1) * maxTestSE

    # Time-averaged EETDR (AULC)
    # Extract actual recorded episode numbers from GzmTest for one repetition
    num_points = TestEETDR.shape[1]
    xs = npGzmTest[:num_points, 1]
    AULC = [metrics.auc(xs[:num_points], TestEETDR[z]) / hp.M for z in range(hp.Z)]
    meanAULC = np.mean(AULC)
    meAULC = t.ppf(0.95, hp.Z - 1) * np.std(AULC, ddof=1) / np.sqrt(hp.Z)
    
    # Score calculation (weighted average of lower bounds)
    score = 0.6 * (meanMaxTestEETDR - maxTestME) + 0.4 * (meanAULC - meAULC)
    
    return {
        'meanMaxTestEETDR': meanMaxTestEETDR,
        'maxTestME': maxTestME,
        'meanAULC': meanAULC,
        'meAULC': meAULC,
        'AlgScore': score
    }

def plot_learning_curve(hp, result, fig_name, folder="./Results/Training Plots", is_superlative=False):
    """
    Generates a learning curve plot with confidence intervals and training metrics.

    Args:
        hp: Hyperparameter configuration object.
        result: The results dictionary returned by run_experiment.
        is_superlative: Boolean to indicate if this is the best LHD run.
    """
    npGzmTest = np.array(result['GzmTest'])
    TestEETDR = np.reshape(npGzmTest[:, 2], (hp.Z, -1))

    avgTestEETDR = np.mean(TestEETDR, axis=0)
    avgTestSE = np.std(TestEETDR, axis=0, ddof=1) / np.sqrt(hp.Z)
    avgTestHW = t.ppf(1 - 0.05 / 2, hp.Z - 1) * avgTestSE

    meanMaxTestEETDR = result['meanMaxTestEETDR']
    maxTestME = result['maxTestME']
    meanAULC = result['meanAULC']
    meAULC = result['meAULC']
    AlgScore = result['AlgScore']
    
    # Consistent colors for each algorithm matching combined plots
    colors = {
        'Semi_Grad_SARSA': 'tab:red',
        'Lambda_SARSA': 'tab:purple',
        'REINFORCE': 'tab:blue'
    }
    color = colors.get(hp.alg_name, 'tab:blue')

    plt.figure()
    num_points = TestEETDR.shape[1]
    xs = npGzmTest[:num_points, 1]
    plt.plot(xs, avgTestEETDR, marker='o', ms=3, mec='k', linewidth=1, color=color, label='Mean EETDR')
    plt.fill_between(xs, avgTestEETDR - avgTestHW, avgTestEETDR + avgTestHW, alpha=0.2, color=color, label='95% Confidence Interval')

    plt.xlabel('Episode', fontsize=9)
    plt.ylabel('Estimated Expected\nTotal Discounted Reward (EETDR)', fontsize=9)

    # Construct dynamic hyperparameter string for the title
    a_label = r"$\alpha_\theta$" if "REINFORCE" in hp.alg_name else r"$\alpha_a$"
    details = [f"$\\gamma$={hp.gamma}", f"{a_label}={hp.alpha_a}"]
    if hasattr(hp, 'alpha_b'): details.append(f"$\\alpha_b$={hp.alpha_b}")
    if hasattr(hp, 'eps_a'): details.append(f"$\\epsilon_a$={hp.eps_a}")
    if hasattr(hp, 'eps_b'): details.append(f"$\\epsilon_b$={hp.eps_b}")
    if hasattr(hp, 'alpha_w_a'): 
        wa_str = f"$\\alpha_{{w,a}}$={hp.alpha_w_a}"
        if hasattr(hp, 'alpha_w_b'): wa_str += f", $b_w$={hp.alpha_w_b}"
        details.append(wa_str)
    
    if hasattr(hp, 'lambd_a'): 
        details.append(f"$\\lambda_a$={hp.lambd_a}")
        details.append(f"$\\lambda_b$={hp.lambd_b}")
    
    if hasattr(hp, 'n_steps'): details.append(f"$n$={hp.n_steps}")
    details.append(f"$q_0$={hp.qinit}")
    details.append(f"$T$={hp.n_tilings}")
    details.append(f"$d$={hp.Sintervals}")
    
    hp_details = ", ".join(details)

    plt.title(
        f"{hp.alg_name} Performance, {hp.Z} reps, "
        f"{np.round(result['wallclock'] / (hp.Z) / 60, 1)} m/rep, {hp.env_name}\n"
        f"{hp_details}\n"
        f"Max EETDR: {meanMaxTestEETDR:>6.1f} - {maxTestME:4.1f}, "
        f"Mean EETDR = {meanAULC:>6.1f} - {meAULC:4.1f}, "
        f"Alg Score ={AlgScore:6.1f}\n"
        f"Superlative Policy EETDR: {result['supETDR_ho']:>6.2f} +/- {result['supETDRhw_ho']:4.2f}",
        fontsize=8)
    plt.legend(loc='upper left', fontsize=7, framealpha=0.8)
    plt.xlim([-0.05 * hp.M, 1.05 * hp.M])
    y_min = -600 if "REINFORCE" in hp.alg_name else -400
    plt.ylim([y_min, 300])
    plt.grid(which='both')
    plt.savefig(f"{folder}/{fig_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Utility plot successfully saved to {folder} folder.")



def display_best_policy(env, Q, phi, hp, num_reps_show=5, record_video=False, video_folder="./Results/Superlative Video"):
    """
    Renders the environment to visually demonstrate the performance of the best policy.

    Args:
        env: The Gymnasium environment.
        Q: The weight matrix/Q-table to visualize.
        phi: The state discretizer.
        hp: Hyperparameter configuration object.
        num_reps_show: Number of episodes to render.
        record_video: If True, saves MP4 files instead of rendering to screen.
        video_folder: Directory to save recorded videos.
    """
    render_mode = 'rgb_array' if record_video else 'human'
    render_env = gym.make(hp.env_name, render_mode=render_mode)
    
    if record_video:
        # Wrap the environment to add the text label before recording starts
        render_env = VideoLabelWrapper(render_env, hp.alg_name)
        render_env = gym.wrappers.RecordVideo(render_env, video_folder, episode_trigger=lambda x: True, name_prefix=hp.alg_name)

    num_actions = render_env.action_space.n
    for rep in range(num_reps_show):
        terminated = truncated = False
        state, _ = render_env.reset(seed=GLOBAL_SEED + 1000 + rep + hp.offset)
        while not (terminated or truncated):
            state_features = phi(state)
            q_values = Q[state_features, :].sum(axis=0)
            action = np.random.choice(np.flatnonzero(q_values == q_values.max()))

            state, reward, terminated, truncated, _ = render_env.step(action)
        time.sleep(0.5)
    render_env.close()

def confinterval(data, alpha=0.05):
    """ 
    Calculates the mean and the half-width of a 1-alpha confidence interval. 

    Args: 
        data: Array-like containing the sample data. 
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        A tuple (mean, half_width).
    """
    n = np.size(data)
    if n <= 1:
        return np.mean(data), 0.0
    se = np.std(data, ddof=1) / np.sqrt(n)
    ts = t.ppf(1 - alpha / 2, n - 1)
    return np.mean(data), ts * se

class BestScoreTracker:
    """Maintains the top-K (mean EETDR, half-width, Q) triples seen so far."""

    def __init__(self, k):
        """
        Initializes the tracker with a capacity of k records.
 
        Args: 
            k: Number of top records to maintain.
        """
        self.k = k
        self.scores = [{'ETDR': -np.inf, 'ETDR_hw': np.inf, 'w': None} # Initialize score and weight records
                       for _ in range(k)]

    def update(self, mean, hw, w):
        """ 
        Attempts to insert a new policy result into the top-K list if it outperforms existing ones. 
 
        Args: 
            mean: The mean EETDR of the policy. 
            hw: The confidence interval half-width. 
            w: The weight matrix (policy) associated with the score.

        Returns:
            True if the record was added to the top-K list, False otherwise.
        """
        # Ensure we don't store a reference to the mutating weight array
        w_copy = np.copy(w)
        for i, e in enumerate(self.scores):
            if mean > e['ETDR']: # Compare based on mean EETDR
                self.scores.insert(i, {'ETDR': mean,
                                       'ETDR_hw': hw,
                                       'w': w_copy}) # Store a copy of the weights
                self.scores.pop()
                return True
        return False

    def __getitem__(self, i):
        """Accesses a record from the tracker by index."""
        return self.scores[i]
