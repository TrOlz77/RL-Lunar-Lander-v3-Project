# =============================================================================
# Library Imports
# =============================================================================
from __future__ import annotations

import numpy as np
import collections
from datetime import datetime
from dataclasses import dataclass
# Import utilities
from utils import setup_environment, run_experiment, plot_learning_curve, display_best_policy, analyze_lhd_results, run_lhd_experiment


# =============================================================================
# Hyperparameters
# =============================================================================
@dataclass
class HP:
    alg_name: str = "Semi_Grad_SARSA"   # Identifier used for plot labels and logging
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
    n_steps: int = 4                # The 'n' in n-step bootstrapping
    Z: int = 10                     # Number of independent statistical replications
    M: int = 500                    # Total episodes per replication
    test_freq: int = 25             # Frequency of evaluation milestones
    num_test_reps: int = 30         # Episodes per greedy policy evaluation
    offset: int = 0                 # Seed offset for experimental control

    def alpha(self, c): return self.alpha_a / (1 + c) ** self.alpha_b
    def epsilon(self, c): return self.eps_a / (1 + c) ** self.eps_b


# =============================================================================
# n-step SARSA Semi-Gradient with linear VFA Algorithm
# =============================================================================
def run_episode(env, w, C, phi, hp: HP, episode_seed: int, episode_m: int) -> float:
    """Single episode update logic for n-step SARSA (On-Policy)."""
    num_actions = env.action_space.n

    def get_action(state_features, current_m):
        """
        Epsilon-greedy action selection using approximated Q-values and 
        an epsilon schedule.
        """
        if np.random.rand() > hp.epsilon(current_m): 
            q_values_for_state = w[state_features, :].sum(axis=0)
            return np.random.choice(np.flatnonzero(q_values_for_state == q_values_for_state.max()))
        else:
            return np.random.randint(0, num_actions)

    # 1. Initialize environment and queues
    state = env.reset(seed=episode_seed)[0]
    state_queue = collections.deque([state])
    
    state_f = phi(state)
    action = get_action(state_f, episode_m)
    action_queue = collections.deque([action])
    reward_queue = collections.deque([])
    
    terminated = truncated = False
    Gm = 0  # Total episodic reward metric

    # 2. SARSA main loop - first n-1 transitions
    for _ in range(hp.n_steps - 1):
        if not (terminated or truncated):
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward_queue.append(reward)
            Gm += reward
            state_queue.append(next_state)
            
            next_state_f = phi(next_state)
            action = get_action(next_state_f, episode_m)
            action_queue.append(action)

    # 3. SARSA main loop - until episode is complete
    while not (terminated or truncated):
        next_state, reward, terminated, truncated, _ = env.step(action_queue[-1])
        state_queue.append(next_state)
        reward_queue.append(reward)
        Gm += reward
        
        next_state_f = phi(next_state)
        next_action = get_action(next_state_f, episode_m)
        action_queue.append(next_action)

        # Temporal difference learning mechanism using rewards and bootstrapped value
        discounted_rewards = np.dot(reward_queue, hp.gamma**np.arange(len(reward_queue)))
        bootstrap = (1 - terminated) * (hp.gamma ** len(reward_queue)) * w[next_state_f, next_action].sum()
        qhat = discounted_rewards + bootstrap

        # Compute TD Error (Delta) and update weights
        active_tiles = phi(state_queue[0])
        current_action = action_queue[0]
        delta = qhat - w[active_tiles, current_action].sum()
        
        # Update counter and weights
        C[active_tiles, current_action] += 1
        lrs = hp.alpha(C[active_tiles, current_action]) / hp.n_tilings
        w[active_tiles, current_action] += lrs * delta
        
        state_queue.popleft()
        action_queue.popleft()
        reward_queue.popleft()

    # 4. SARSA cleanup loop - remaining steps in buffer
    while len(reward_queue) > 0:
        qhat = np.dot(reward_queue, hp.gamma**np.arange(len(reward_queue)))
        
        active_tiles = phi(state_queue[0])
        current_action = action_queue[0]
        delta = qhat - w[active_tiles, current_action].sum()
        
        C[active_tiles, current_action] += 1
        lrs = hp.alpha(C[active_tiles, current_action]) / hp.n_tilings
        w[active_tiles, current_action] += lrs * delta

        state_queue.popleft()
        action_queue.popleft()
        reward_queue.popleft()

    return Gm


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    hp = HP()
    ts = datetime.now().strftime('%m_%d_%Y_%H_%M')
    env, phi, SAsize, num_actions = setup_environment(hp) # Dimensions of the state-action weight matrix
    results = run_experiment(env, phi, SAsize, hp, run_episode) # Run the experiment using the episode index for epsilon
    plot_learning_curve(hp, results, f"Training Plot_{hp.alg_name}_{ts}", folder="./Results/Training Plots")
    # display_best_policy(results['env'], results['Q'], results['phi'], hp) # Visual demonstration of the policy

    factors, scores, names = run_lhd_experiment(
        HP, run_episode, 
        hparam_names=['alpha_a', 'alpha_b', 'eps_a', 'eps_b', 'qinit', 'n_steps', 'n_tilings', 'Sintervals'], 
        n_runs=100, # Total configurations for hyperparameter search
        factor_names=['Alpha_a', 'Alpha_b', 'Eps_a', 'Eps_b', 'Q_Init', 'N_Steps', 'N_Tilings', 'S_Intervals'],
    )
    analyze_lhd_results(factors, scores, names, hp.alg_name)
