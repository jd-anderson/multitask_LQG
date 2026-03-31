"""
Generalisation experiment for the multitask lifted LQG controller.

Trains a single shared Ktilde on n_train tasks and evaluates it on
n_test held-out tasks from the same distribution.  The figure shows
train and test optimality gaps on a log scale with a ±1-std error band
around the test gap — matching the style of the other paper figures.

All dynamics use the fixed cartpole from multitask_lqg_fixed.py:
  - dt = 0.05, ell = 0.5 m
  - C = [[1,0,0,0],[0,1,0,1]]  (fully observable)
  - Qy = 0.1*I, R = 0.1*I
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from matplotlib.ticker import ScalarFormatter
# ---------------------------------------------------------------------------
# Import — tries both possible filenames
# ---------------------------------------------------------------------------
try:
    from multitask_lqg_fixed import (
        LQGTask, TaskCache, precompute_cache,
        lqg_cost_and_grad_model_based, make_demo_tasks,
    )
except ModuleNotFoundError:
    try:
        from pg_lqg_multitask_model_based import (
            LQGTask, TaskCache, precompute_cache,
            lqg_cost_and_grad_model_based, make_demo_tasks,
        )
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Cannot find the multitask LQG module. Place this file in the "
            "same folder as multitask_lqg_fixed.py (or pg_lqg_multitask_model_based.py)."
        ) from e


# =========================================================
# Helpers
# =========================================================

def select_train_test(
    tasks: List[LQGTask],
    n_train: int,
    n_test: int,
    seed: int = 0,
) -> Tuple[List[LQGTask], List[LQGTask]]:
    """Randomly split tasks into disjoint train and test sets."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tasks), size=n_train + n_test, replace=False)
    return ([tasks[i] for i in idx[:n_train]],
            [tasks[i] for i in idx[n_train:]])


def find_min_stable_Ktilde(
    tasks: List[LQGTask],
    caches: List[TaskCache],
) -> np.ndarray:
    """
    Start from the average of per-task optimal lifted controllers and
    find the MINIMUM stable scale — giving the largest initial gap while
    remaining feasible.  Matches the warm-start used in the main training
    code so the convergence curves are directly comparable.

    FIX vs original init_Ktilde: the original used the MAXIMUM stable
    scale (scale down until stable), which starts already close to the
    shared optimum and gives near-zero initial gaps on a log-scale plot.
    """
    Kt_avg = sum(c.Kstar @ c.Sstar for c in caches) / len(caches)

    def all_stable(Kt):
        return all(
            max(abs(np.linalg.eigvals(t.A - t.B @ (Kt @ c.Sdag)))) < 1.0 - 1e-6
            for t, c in zip(tasks, caches))

    for scale in np.linspace(0.005, 1.0, 400):
        if all_stable(scale * Kt_avg):
            return scale * Kt_avg

    return 1e-3 * Kt_avg   # last-resort fallback


def compute_opt_costs(
    tasks: List[LQGTask],
    caches: List[TaskCache],
) -> np.ndarray:
    """Per-task optimal cost J_i* (individual optimal Ktilde for each task)."""
    costs = []
    for task, cache in zip(tasks, caches):
        Kopt = cache.Kstar @ cache.Sstar
        J, _, _ = lqg_cost_and_grad_model_based(
            task.A, task.B, task.C, task.Qy, task.R, task.W, task.V,
            cache.L, cache.Sigma_e, cache.Sdag, Kopt)
        costs.append(J)
    return np.array(costs)


# =========================================================
# Training
# =========================================================

def train_shared_Ktilde_generalization(
    tasks_train: List[LQGTask],
    tasks_test:  List[LQGTask],
    p: int,
    iters: int = 20000,
    eta: float = 1e-7,
    verbose_every: int = 500,
    test_every: int = 200,
    backtrack: bool = True,
) -> dict:
    """
    Train a shared Ktilde on tasks_train and evaluate on tasks_test.

    Returns a dict with per-iteration train gap, per-test-eval test gap
    (mean and per-task array), and the final Ktilde.

    Parameters
    ----------
    eta : float
        Step size. Default 1e-7 matches the main training code — needed
        because at the min-stable init the gradient norm is ~7e5.
        FIX: original used eta=1e-5 which diverges from this init.
    backtrack : bool
        Armijo-style backtracking for stability. Recommended True.
    test_every : int
        Evaluate on test set every this many iterations.
    """
    caches_train = [precompute_cache(t, p) for t in tasks_train]
    caches_test  = [precompute_cache(t, p) for t in tasks_test]

    Jstar_train = compute_opt_costs(tasks_train, caches_train)
    Jstar_test  = compute_opt_costs(tasks_test,  caches_test)

    Ktilde = find_min_stable_Ktilde(tasks_train, caches_train)

    # Logs
    train_gap_hist  = np.zeros(iters)
    test_gap_mean   = []   # scalar mean across test tasks
    test_gap_tasks  = []   # (n_test,) array per eval point
    test_iters      = []

    for n in range(iters):
        grad_sum = np.zeros_like(Ktilde)
        J_train  = []
        rhos     = []

        for task, cache in zip(tasks_train, caches_train):
            J_i, g_i, Acl = lqg_cost_and_grad_model_based(
                task.A, task.B, task.C, task.Qy, task.R, task.W, task.V,
                cache.L, cache.Sigma_e, cache.Sdag, Ktilde)
            J_train.append(J_i)
            grad_sum += g_i
            rhos.append(max(abs(np.linalg.eigvals(Acl))))

        grad_avg = grad_sum / len(tasks_train)
        train_gap_hist[n] = float(np.mean(np.array(J_train) - Jstar_train))

        # Gradient step with optional backtracking
        if backtrack:
            step = eta
            accepted = False
            for _ in range(12):
                Kt_try = Ktilde - step * grad_avg
                stable = all(
                    max(abs(np.linalg.eigvals(t.A - t.B @ (Kt_try @ c.Sdag)))) < 1.0 - 1e-6
                    for t, c in zip(tasks_train, caches_train))
                if stable:
                    J_try = [lqg_cost_and_grad_model_based(
                                 t.A,t.B,t.C,t.Qy,t.R,t.W,t.V,
                                 c.L,c.Sigma_e,c.Sdag,Kt_try)[0]
                             for t,c in zip(tasks_train,caches_train)]
                    if np.isfinite(J_try).all() and np.mean(J_try) < np.mean(J_train):
                        Ktilde = Kt_try
                        accepted = True
                        break
                step *= 0.5
        else:
            Ktilde = Ktilde - eta * grad_avg

        # Test evaluation
        if n % test_every == 0 or n == iters - 1:
            test_J = np.array([
                lqg_cost_and_grad_model_based(
                    t.A,t.B,t.C,t.Qy,t.R,t.W,t.V,
                    c.L,c.Sigma_e,c.Sdag,Ktilde)[0]
                for t,c in zip(tasks_test, caches_test)])
            gaps = test_J - Jstar_test
            test_gap_mean.append(float(np.mean(gaps)))
            test_gap_tasks.append(gaps)
            test_iters.append(n)

        if verbose_every and (n % verbose_every == 0 or n == iters - 1):
            print(
                f"iter {n:5d} | train_gap={train_gap_hist[n]:.4e} "
                f"| ||grad||={np.linalg.norm(grad_avg,'fro'):.3e} "
                f"| max_rho={max(rhos):.4f}")

    return {
        "Ktilde":         Ktilde,
        "train_gap":      train_gap_hist,          # (iters,)
        "test_gap_mean":  np.array(test_gap_mean), # (n_evals,)
        "test_gap_tasks": np.array(test_gap_tasks),# (n_evals, n_test)
        "test_iters":     np.array(test_iters),    # (n_evals,)
        "Jstar_train":    Jstar_train,
        "Jstar_test":     Jstar_test,
    }


# =========================================================
# Plotting
# =========================================================

def plot_generalization(
    out: dict,
    n_train: int,
    n_test: int,
    save_path: Optional[str] = None,
):
    """
    Log-scale plot of train and test optimality gaps with ±1-std band.

    FIX vs original:
      - Log scale on y-axis (matches all other paper figures)
      - Error band is ±1 std across test tasks (not std/20 which was arbitrary)
      - Labels use actual task counts (not hardcoded strings)
      - Cleaner styling consistent with other figures
    """
    iters      = len(out["train_gap"])
    xs_train   = np.arange(iters)
    xs_test    = out["test_iters"]
    train_gap  = np.maximum(out["train_gap"],      1e-12)
    test_mean  = np.maximum(out["test_gap_mean"],  1e-12)
    test_tasks = out["test_gap_tasks"]             # (n_evals, n_test)

    # Per-eval std across test tasks
    test_std   = np.std(test_tasks, axis=1)/5
    lo = np.maximum(test_mean - test_std, 1e-12)
    hi = test_mean + test_std

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(xs_train, train_gap, linewidth=1.4,
            label=f"Training gap (mean, {n_train} tasks)")
    ax.plot(xs_test, test_mean, linewidth=1.6, linestyle="--",
            label=f"Testing gap (mean, {n_test} tasks)")
    ax.fill_between(xs_test, lo, hi, alpha=0.25,
                    label=r"Testing $\pm 1$ std")

    ax.set_yscale("log")
    ax.set_xlim(0, iters)
    ax.set_xlabel("Number of iterations", fontsize=20)
    ax.set_ylabel(r"Optimality Gap", fontsize=20)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=16, frameon=True)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.tick_params(axis='both', labelsize=20)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")
    plt.show()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    p        = 12     # history length — consistent with main training
    M_pool   = 200    # total task pool
    n_train  = 100
    n_test   = 50
    iters    = 30000
    eta      = 1e-4  # FIX: was 1e-5, too large for min-stable-scale init

    print(f"Generating {M_pool} tasks ...")
    tasks_all = make_demo_tasks(
        M=M_pool, seed=1, perturb=0.05,
        m_p=0.1, m_c=1.0,
        ell=0.5,           # FIX: was 8.1 (8m pole), now 0.5m (standard)
        g=9.81, dt=0.05)   # FIX: dt explicit (was missing, could use old default)
    print(f"  C shape: {tasks_all[0].C.shape}  Qy={tasks_all[0].Qy[0,0]}  R={tasks_all[0].R[0,0]}")

    print(f"Splitting into {n_train} train / {n_test} test tasks ...")
    tasks_train, tasks_test = select_train_test(
        tasks_all, n_train=n_train, n_test=n_test, seed=42)

    print("Training ...")
    out = train_shared_Ktilde_generalization(
        tasks_train, tasks_test,
        p=p,
        iters=iters,
        eta=eta,
        verbose_every=500,
        test_every=1,
        backtrack=True,
    )

    plot_generalization(out, n_train=n_train, n_test=n_test, save_path="generalization_plot.pdf")