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
    alg_name: str = "Lambda_SARSA"   # Identifier used for plot labels and logging
    env_name: str = "LunarLander-v3" # Farama Gymnasium environment name
    alpha_a: float = 1.0            # Learning rate numerator (alpha = a / (1 + c)^b)
    alpha_b: float = 0.5            # Learning rate decay exponent
    eps_a: float = 1.0              # Epsilon numerator (epsilon = a / (1 + c)^b)
    eps_b: float = 0.5              # Epsilon decay exponent
    qinit: float = 1.0              # Initial Q-value scaling factor (0.0 to 1.0)
    qrange: tuple = (-100, 100)     # Theoretical min/max return for initialization
    Sintervals: int = 9             # Discretization resolution per observation dimension
    gamma: float = 0.999            # Discount factor for temporal difference updates 
    iht_size: int = 262144          # allowed hash table size for tile coding
    n_tilings: int = 8              # Number of tilings for tile coding
    lambd_a: float = 0.9            # Lambda numerator
    lambd_b: float = 0.0            # Lambda decay exponent
    Z: int = 10                     # Number of independent statistical replications
    delta_clip: float = 10.0        # TD error clipping for stability
    M: int = 500                    # Total episodes per replication
    test_freq: int = 25             # Frequency of evaluation milestones
    num_test_reps: int = 30         # Episodes per greedy policy evaluation
    offset: int = 0                 # Seed offset for experimental control

    def alpha(self, c): return self.alpha_a / (1 + c) ** self.alpha_b
    def epsilon(self, c): return self.eps_a / (1 + c) ** self.eps_b
    def lambd(self, c): return self.lambd_a / (1 + c) ** self.lambd_b


# =============================================================================
# SARSA(lambda) Semi-Gradient with linear VFA Algorithm
# =============================================================================
def run_episode(env, w, C, phi, hp: HP, episode_seed: int, episode_m: int) -> float:
    """Single episode update logic for SARSA(lambda) (On-Policy)."""
    num_actions = env.action_space.n
    
    # Calculate the lambda value for this episode based on the schedule
    lmbda = hp.lambd(episode_m)
    # Initialize eligibility traces
    z = np.zeros_like(w)
    active_indices = set()

    def get_action(state_features, current_m):
        if np.random.rand() > hp.epsilon(current_m): 
            q_values_for_state = w[state_features, :].sum(axis=0)
            return np.random.choice(np.flatnonzero(q_values_for_state == q_values_for_state.max()))
        else:
            return np.random.randint(0, num_actions)

    # 1. Initialize state and first action
    state = env.reset(seed=episode_seed)[0]
    state_f = phi(state)
    action = get_action(state_f, episode_m)

    terminated = truncated = False
    Gm = 0

    # 2. SARSA(lambda) main loop
    while not (terminated or truncated):
        # Apply action and observe system information
        next_state, reward, terminated, truncated, _ = env.step(action)
        Gm += reward
        next_state_f = phi(next_state)
        
        # Select next action (Epsilon-greedy)
        next_action = get_action(next_state_f, episode_m)

        # Compute qhat (target) and TD error delta (clipped)
        q_curr = w[state_f, action].sum()
        q_next = w[next_state_f, next_action].sum()
        qhat = reward + (1 - terminated) * hp.gamma * q_next
        delta = np.clip(qhat - q_curr, -hp.delta_clip, hp.delta_clip)
        
        # Update eligibility traces (Decay and then Increment)
        decay_factor = hp.gamma * lmbda
        new_active = set()
        for idx in active_indices:
            z[idx] *= decay_factor
            if abs(z[idx]) > 1e-4: new_active.add(idx)
            else: z[idx] = 0
        active_indices = new_active

        for f in state_f:
            idx = (f, action)
            z[idx] += 1.0 # Accumulating traces
            C[idx] += 1
            active_indices.add(idx)

        # Update weight vector using semi-gradient update (Vectorized)
        if active_indices:
            rows, cols = zip(*active_indices)
            lrs = hp.alpha(C[rows, cols]) / hp.n_tilings
            w[rows, cols] += lrs * delta * z[rows, cols]

        # Transition state and action
        state_f = next_state_f
        action = next_action

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
        hparam_names=['alpha_a', 'alpha_b', 'eps_a', 'eps_b', 'qinit', 'lambd_a', 'lambd_b', 'n_tilings', 'Sintervals'], 
        n_runs=100, # Total configurations for hyperparameter search
        factor_names=['Alpha_a', 'Alpha_b', 'Eps_a', 'Eps_b', 'Q_Init', 'Lambda_a', 'Lambda_b', 'N_Tilings', 'S_Intervals'],
    )
    analyze_lhd_results(factors, scores, names, hp.alg_name)