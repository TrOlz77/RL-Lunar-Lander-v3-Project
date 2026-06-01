# =============================================================================
# Library Imports
# =============================================================================
from __future__ import annotations

import numpy as np
import collections
from datetime import datetime
from dataclasses import dataclass
# Import utilities
from utils import setup_environment, run_experiment, plot_learning_curve, analyze_lhd_results, run_lhd_experiment


# =============================================================================
# Hyperparameters
# =============================================================================
@dataclass
class HP:
    alg_name: str = "REINFORCE"
    env_name: str = "LunarLander-v3"
    alpha_a: float = 0.0005         # Policy learning rate numerator (alpha_theta)
    alpha_b: float = 0.0            # Policy learning rate decay exponent
    alpha_w_a: float = 0.005        # Baseline learning rate numerator
    alpha_w_b: float = 0.0          # Baseline learning rate decay exponent
    qinit: float = 0.0              # Initial preference scaling factor
    qrange: tuple = (-1, 1)         # Theoretical preference range for initialization
    Sintervals: int = 9             # Discretization resolution per dimension
    gamma: float = 0.999            # Discount factor for return calculation
    iht_size: int = 262144          # Allowed hash table size for tile coding
    n_tilings: int = 8              # Number of tilings for tile coding
    Z: int = 10                     # Number of independent statistical replications
    delta_clip: float = 50.0        # Increased clipping for LunarLander reward scales
    M: int = 500                    # Total episodes per replication
    test_freq: int = 25             # Frequency of evaluation milestones
    num_test_reps: int = 30         # Episodes per greedy policy evaluation
    offset: int = 0                 # Seed offset for experimental control

    def alpha(self, c): return self.alpha_a / (1 + c) ** self.alpha_b
    def alpha_w(self, c): return self.alpha_w_a / (1 + c) ** self.alpha_w_b


# =============================================================================
# REINFORCE with Baseline Semi-Gradient with linear VFA Algorithm
# =============================================================================
def run_episode(env, w, w_baseline, phi, hp: HP, episode_seed: int, episode_m: int) -> float:
    """Single episode update logic for REINFORCE with Baseline (On-Policy)."""
    num_actions = env.action_space.n

    def get_action_and_probs(state_features):
        # Calculate preferences by summing weights for active features
        h = w[state_features, :].sum(axis=0)
        # Softmax with numerical stability
        h_max = np.max(h)
        exp_h = np.exp(h - h_max)
        # pi(a|s, theta)
        probs = exp_h / np.sum(exp_h)
        action = np.random.choice(num_actions, p=probs)
        return action, probs

    # Initialize queues for state-action-reward sequence (Matches Professor's structure)
    state_queue = collections.deque([])
    action_queue = collections.deque([])
    reward_queue = collections.deque([])
    
    state_cont, _ = env.reset(seed=episode_seed)
    terminated = truncated = False
    Gm = 0
    
    # 1. Forward loop: Generate a full episode following policy pi(a|s, theta)
    while not (terminated or truncated):
        state_f = phi(state_cont)
        action, _ = get_action_and_probs(state_f)
        
        state_queue.append(state_f)
        action_queue.append(action)

        state_cont, reward, terminated, truncated, _ = env.step(action)
        reward_queue.append(reward)
        Gm += reward # Cumulative undiscounted reward for monitoring

    # 2. Backward loop: REINFORCE update with Baseline
    # Step sizes scaled by n_tilings for Linear VFA
    alpha_theta = hp.alpha(episode_m) / hp.n_tilings
    alpha_w = hp.alpha_w(episode_m) / hp.n_tilings
    
    G_discounted = 0
    # Process the episode backwards to correctly associate returns with states
    while len(state_queue) > 0:
        sf = state_queue.pop()          # S_t (starting from T-1)
        a_t = action_queue.pop()
        r_t = reward_queue.pop()        # R_{t+1}

        # Iteratively calculate the return G from time t (Matches Professor's structure)
        G_discounted = r_t + (hp.gamma * G_discounted)
        
        # 1. Compute TD error (delta) = G - Vhat(s, w)
        v_hat = w_baseline[sf, 0].sum()
        delta = G_discounted - v_hat
        
        # Stability clipping
        delta_clip = np.clip(delta, -hp.delta_clip, hp.delta_clip)
        
        # 2. Update baseline weights
        w_baseline[sf, 0] += alpha_w * delta_clip
        
        # 3. Update policy weights (theta)
        _, pi = get_action_and_probs(sf)
        grad_log_pi = -pi
        grad_log_pi[a_t] += 1.0
        
        # The index t is represented by len(state_queue) in the backward pass
        w[sf, :] += alpha_theta * (hp.gamma ** len(state_queue)) * delta_clip * grad_log_pi
        
    return Gm


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    hp = HP()
    ts = datetime.now().strftime('%m_%d_%Y_%H_%M')
    env, phi, SAsize, num_actions = setup_environment(hp)
    results = run_experiment(env, phi, SAsize, hp, run_episode)
    plot_learning_curve(hp, results, f"Training Plot_{hp.alg_name}_{ts}", folder="./Results/Training Plots")

    factors, scores, names = run_lhd_experiment(
        HP, run_episode, 
        # Removed alpha_b/alpha_w_b from LHD to prevent vanishing gradients
        hparam_names=['alpha_a', 'alpha_w_a', 'qinit', 'n_tilings', 'Sintervals'], 
        n_runs=100, # Total configurations for hyperparameter search
        factor_names=['Alpha_a', 'Alpha_W_a', 'Q_Init', 'N_Tilings', 'S_Intervals'],
    )
    analyze_lhd_results(factors, scores, names, hp.alg_name)