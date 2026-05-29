# =============================================================================
# Library Imports
# =============================================================================
from __future__ import annotations

import numpy as np
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
    delta_clip: float = 10.0        # TD error clipping for baseline stability
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

    # 1. Generate an episode following the current policy
    states_features = []
    actions = []
    rewards = []
    
    state_cont, _ = env.reset(seed=episode_seed)
    terminated = truncated = False
    Gm = 0
    
    while not (terminated or truncated):
        state_f = phi(state_cont)
        action, _ = get_action_and_probs(state_f)
        
        next_state_cont, reward, terminated, truncated, _ = env.step(action)
        
        states_features.append(state_f)
        actions.append(action)
        rewards.append(reward)
        
        state_cont = next_state_cont
        Gm += reward

    # 2. Update policy (theta) and baseline (w_baseline)
    T = len(rewards)
    
    # Pre-calculate discounted returns G_t
    returns = np.zeros(T)
    G = 0
    # G_t is calculated as the sum of discounted rewards from time t+1 onwards
    for t in reversed(range(T)):
        G = rewards[t] + (hp.gamma * G)
        returns[t] = G
        
    # Step sizes scaled by n_tilings for Linear VFA
    alpha_theta = hp.alpha(episode_m) / hp.n_tilings
    alpha_w = hp.alpha_w(episode_m) / hp.n_tilings
    
    for t in range(T):
        G_t = returns[t]          # G
        sf = states_features[t]   # x(S_t)
        a_t = actions[t]          # A_t
        
        # 1. delta = G - v(S_t, w)
        v_hat = w_baseline[sf, 0].sum()
        delta = G_t - v_hat
        
        # Stability clipping for LunarLander-v3
        delta_clip = np.clip(delta, -hp.delta_clip, hp.delta_clip)
        
        # 2. Update baseline weights: w <- w + alpha_w * delta * grad(v(S_t, w))
        w_baseline[sf, 0] += alpha_w * delta_clip
        
        # 3. Update policy weights: theta <- theta + alpha_theta * gamma^t * delta * grad(ln pi(A_t|S_t, theta))
        _, pi = get_action_and_probs(sf)
        
        grad_log_pi = -pi
        grad_log_pi[a_t] += 1.0
        
        w[sf, :] += alpha_theta * (hp.gamma ** t) * delta_clip * grad_log_pi
        
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
        hparam_names=['alpha_a', 'alpha_b', 'alpha_w_a', 'alpha_w_b', 'qinit', 'n_tilings', 'Sintervals'], 
        n_runs=100, # Total configurations for hyperparameter search
        factor_names=['Alpha_a', 'Alpha_b', 'Alpha_W_a', 'Alpha_W_b', 'Q_Init', 'N_Tilings', 'S_Intervals'],
    )
    analyze_lhd_results(factors, scores, names, hp.alg_name)