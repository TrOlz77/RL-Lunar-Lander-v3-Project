# Lunar Lander Reinforcement Learning Project

This project implements and compares three Reinforcement Learning algorithms using Linear Value Function Approximation (Tile Coding) on the `LunarLander-v3` environment:
1. **n-step SARSA Semi-Gradient**
2. **SARSA(lambda) Semi-Gradient**
3. **REINFORCE with Baseline Semi-Gradient**

## Project Outputs

The code generates an organized directory structure for outputs used for performance analysis and reporting.

### 1. Saved Models (`./Saved Models`)
*   **`Best Weights_{Algorithm}_{Timestamp}.npy`**: The saved weight matrix for the best-performing policy discovered during tuning.

### 2. Raw Results (`./Raw Results`)
*   **`Raw Results_{Algorithm}_{Timestamp}.csv`**: The complete raw dataset from the parallelized LHD search.

### 3. Results Directory (`./Results`)
*   **`Learning Curve_{Algorithm}_{Timestamp}.csv`**: Learning curve data used for aggregate plotting.
*   **`./Results/Top 10/Top 10_{Algorithm}_{Timestamp}.csv`**: Ranked configurations based on the Algorithm Score.
*   **`./Results/Top 5/Top 5_{Algorithm}_{Timestamp}.csv`**: Absolute champions ranked by SP95LB.
*   **`./Results/Executive_Summary_Results.csv`**: Aggregated performance of the superlative policies across all algorithms.
*   **`./Results/Superlative Hyperparameters/Superlative Hyperparameters_{Algorithm}_{Timestamp}.txt`**: Record of tuned hyperparameters.
*   **`./Results/Superlative Hyperparameters/Analysis_{Algorithm}_{Timestamp}.txt`**: Response Surface Methodology (RSM) regression and ANOVA output.
*   **`./Results/Training Plots/Training Plot_{Algorithm}_{Timestamp}.png`**: Standard learning curves generated during training.
*   **`./Results/LHD Plots/LHD Plot_{Algorithm}_{Timestamp}.png`**: Learning curves specifically for the superlative configurations.
*   **`Combined_Superlative_Learning_Curves.png`**: Comparative plot of all superlative algorithm configurations.
*   **`./Results/Superlative Video/`**: MP4 recordings of the agent playing in the environment using the superlative weights.

---

## How to Use

### Prerequisites
Install the required dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Step 1: Run Semi-Gradient n-step SARSA
Execute the Semi Grad SARSA script. This trains the agent, performs a Latin Hypercube Design (LHD) search for hyperparameters, and saves the superlative results.
```bash
python3 Semi_Grad_SARSA.py
```

### Step 2: Run SARSA(lambda)
Execute the Lambda SARSA script to perform the same kind of tuning and training process for step 1.
```bash
python3 Lambda_SARSA.py
```
### Step 3: Run REINFORCE with Baseline
Execute the REINFORCE script to perform the same kind of tuning and training process for steps 1, 2.
'''bash
python3 REINFORCE.py
'''

### Step 4: Generate Comparison Plots
Once all three algorithms have been run and their respective `LC_*.csv` files exist in the `Results` folder, generate the combined comparison graph:
```bash
python3 Combined_Learning_Curves.py
```
### Step 5: Generate Videos of the best policies from each algorithm
Execute superlative video generation to show the best policies in action.
```bash
python3 Superlative_Video.py
'''

### Bash Command to run all files in order
```bash
python3 Semi_Grad_SARSA.py && python3 Lambda_SARSA.py && python3 REINFORCE.py && python3 Combined_Learning_Curves.py && python3 Superlative_Video.py
'''

---

### Configuration
Hyperparameters (such as learning rate decay, $n$-steps, or $\lambda$), training iterations ($M$), and statistical replications ($Z$) can be modified within the `HP` dataclass located at the top of `Nstep.py` and `SarsaLambda.py`.

To visually see the agent play after training, uncomment the `display_best_policy` call in the `__main__` block of the algorithm scripts.

## Project Structure & File Descriptions

*   **`Nstep.py`**: Implements the n-step SARSA Semi-Gradient algorithm, including bootstrapping logic and hyperparameter definitions.
*   **`SarsaLambda.py`**: Implements the SARSA(lambda) Semi-Gradient algorithm using replacing eligibility traces.
*   **`utils.py`**: The project's core engine; contains the Tile Coding implementation, experimental pipelines, LHD sampling logic, and statistical analysis functions.
*   **`Algorithm Pseudo Code.txt`**: Contains the pseudocode for the implemented algorithms (n-step SARSA, SARSA(lambda) and Reinforce with baseline).
*   **`Handouts/tiles3.py`**: A local implementation of the Tile Coding software (Version 3.0) by Rich Sutton.
*   **`plot_comparison.py`**: Utility script to aggregate results and generate comparative superlative learning curves.
*   **`requirements.txt`**: Lists necessary libraries including `gymnasium[lunar-lander]`, `statsmodels`, `scipy`, and `joblib`.
*   **`Results/`**: (Folder) Stores processed learning curves, superlative hyperparameter records, and statistical analysis text files.
*   **`Figures/`**: (Folder) Stores all generated PNG visualizations and plots.
*   **`models/`**: (Folder) Contains the saved `.npy` weight matrices for the best-performing policies.
*   **`Raw Results/`**: (Folder) Contains the full CSV datasets generated by the parallel hyperparameter search.
*   **`Videos/`**: (Folder) Stores MP4 recordings of the superlative policies.

---

## Runtime Optimizations

Several optimizations have been implemented to ensure the experiments remain tractable while maintaining algorithmic integrity:

1.  **Sparse Eligibility Traces (SARSA($\lambda$))**: Instead of performing dense matrix operations over the entire Index Hash Table ($2^{18}$ entries) every timestep, the implementation tracks only the indices of non-zero traces. This reduces the complexity of the trace update from $O(\text{IHT Size})$ to $O(\text{Active Traces})$. This optimization is **functionally identical** to the standard algorithm but provides a massive speedup.
2.  **Parallel Hyperparameter Search**: The Latin Hypercube Design (LHD) experiment in `utils.py` uses `joblib` to distribute different hyperparameter configurations across multiple CPU cores. Since each experimental run is independent, this **does not change** the results but allows for significantly more exhaustive tuning within the same timeframe.
3.  **Vectorized Weight Updates**: Across all algorithms, weight updates and Q-value calculations use NumPy's vectorized operations (e.g., `w[active_features, action].sum()`). This leverages optimized C-level implementations for linear algebra, ensuring the linear value function approximation is calculated efficiently without changing the underlying math.
4.  **Numerical Stability in Softmax**: In the REINFORCE implementation, the preference calculation subtracts the maximum preference value before exponentiation. This prevents floating-point overflow/underflow without altering the resulting probability distribution.
