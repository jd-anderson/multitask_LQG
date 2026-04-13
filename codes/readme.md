
# How to Run

The repository currently supports three dynamical system families:

- `cartpole` — a 4-state partially observed cart-pole system with nominal parameters  
  `m_p = 0.1`, `m_c = 1.0`, `ell = 0.5`, `g = 9.81`, and `dt = 0.05`
- `pendulum` — a 2-state partially observed inverted pendulum with nominal parameters  
  `m = 0.5`, `l = 0.3`, `g = 9.81`, and `dt = 0.05`
- `synthetic` — a randomly generated 4-state synthetic linear time-invariant benchmark system

In all three cases, task variability is introduced by perturbing the nominal physical or system parameters, which generates a family of related LQG tasks. The experiments then learn a **shared lifted controller** across these tasks.

### 1. Model-based multitask training

Run:

```bash
python pg_lqg_multitask_model_based.py
````

Inside the script, choose the system family by setting

```python
SYSTEM = "cartpole"   # or "pendulum" or "synthetic"
```

This script runs multitask policy gradient on a shared lifted controller, computes task-specific performance metrics, and evaluates bisimulation-based heterogeneity quantities during training.

System-specific defaults used in the code include:

* **Cart-pole**: partially observed 4-state system with nominal parameters
  `m_p = 0.1`, `m_c = 1.0`, `ell = 0.5`, `g = 9.81`, `dt = 0.05`
* **Pendulum**: partially observed 2-state inverted pendulum with nominal parameters
  `m = 0.5`, `l = 0.3`, `g = 9.81`, `dt = 0.05`
* **Synthetic**: randomly generated linear system with configurable dimensions

You may also adjust the main experiment settings in the script, such as the number of tasks, history length, number of policy-gradient iterations, step size, and perturbation level.

**Important:** The bisimulation computation uses a fast closed-form upper bound by default since the real SDP solver's calculation is based on multithreading. If you want the **SDP-based bisimulation values** instead of the closed-form estimate, set the corresponding option

```python
use_closed_form = False
```

in the bisimulation computation path. The current implementation of `compute_all_bisim_pairs(...)` uses `use_closed_form=True` by default, so this must be turned off explicitly to obtain the SDP-based quantities.

### 2. Generalization experiment

Run:

```bash
python pg_lqg_generalization_demo.py
```

Inside the script, choose the system family by setting

```python
SYSTEM = "cartpole"   # or "pendulum" or "synthetic"
```

This script trains a shared lifted controller on a training set of tasks and evaluates its performance on held-out tasks drawn from the same distribution.

The same system parameterizations are used here:

* **Cart-pole**: nominal parameters `m_p = 0.1`, `m_c = 1.0`, `ell = 0.5`, `g = 9.81`, `dt = 0.05`
* **Pendulum**: nominal parameters `m = 0.5`, `l = 0.3`, `g = 9.81`, `dt = 0.05`
* **Synthetic**: randomly generated LTI system

You may also adjust the experiment settings in the main block, such as:

* history length `p`
* total task pool size `M_pool`
* number of training tasks `n_train`
* number of test tasks `n_test`
* number of iterations `iters`
* step size `eta`
* perturbation level `perturb`

### 3. Variance-reduction experiment

Run:

```bash
python pg_lqg_variance_reduction_demo.py
```

Inside the script, choose the system family by setting

```python
SYSTEM = "cartpole"   # or "pendulum" or "synthetic"
```

This script evaluates the multitask one-point zeroth-order policy gradient estimator and measures how the relative gradient RMSE scales with the number of tasks.

The available system setups are again:

* **Cart-pole**: nominal parameters `m_p = 0.1`, `m_c = 1.0`, `ell = 0.5`, `g = 9.81`, `dt = 0.05`
* **Pendulum**: nominal parameters `m = 0.5`, `l = 0.3`, `g = 9.81`, `dt = 0.05`
* **Synthetic**: randomly generated LTI benchmark

You may also adjust the experiment settings in the main block, including:

* history length `p`
* task pool size `M_pool`
* task counts `task_counts`
* number of Monte Carlo trials `trials`
* number of zeroth-order perturbation directions `n_s`
* smoothing radius `r`

To include the optional out-of-distribution experiment, set

```python
include_ood = True
```

---

## Notes

* The scripts are configured around a **shared lifted controller** learned across multiple tasks.
* The codebase supports both **model-based** and **model-free** experiments.
* The newer versions of the scripts also include experiments on the **inverted pendulum**, in addition to the original cart-pole setup.
* If you only want to reproduce the main figures, the simplest workflow is to run the three scripts above separately and collect the saved outputs.
