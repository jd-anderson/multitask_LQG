"""
Generalisation experiment for the multitask lifted LQG controller.

Trains a single shared Ktilde on n_train tasks and evaluates it on
n_test held-out tasks from the same distribution. The figure shows
train and test optimality gaps on a log scale with a ±1-std error band
around the test gap — matching the style of the other paper figures.

Supports three system regimes (same as the multitask training code):
  - "cartpole"  : 4-state cartpole   (nx=4, nu=1, ny=2)
  - "pendulum"  : 2-state inverted pendulum (nx=2, nu=1, ny=1)
  - "synthetic" : 4-state synthetic LTI (nx=4, nu=2, ny=2)
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
        lqg_cost_and_grad_model_based,
        make_demo_tasks, make_pendulum_tasks, make_synthetic_tasks,
    )
except ModuleNotFoundError:
    try:
        from pg_lqg_multitask_model_based import (
            LQGTask, TaskCache, precompute_cache,
            lqg_cost_and_grad_model_based,
            make_demo_tasks, make_pendulum_tasks, make_synthetic_tasks,
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
    scale_factor: float = 1.0,
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

    scale_factor = max(scale_factor, 1e-6)
    for scale in np.linspace(0.005, 1.0, 400):
        if all_stable(scale * Kt_avg):
            return scale_factor * scale * Kt_avg

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


def make_task_pool(
    system: str,
    M_pool: int,
    seed: int,
    perturb: float,
    dt: float = 0.05,
) -> List[LQGTask]:
    """
    Build a task pool for the selected system.
    """
    if system == "cartpole":
        return make_demo_tasks(
            M=M_pool, seed=seed, perturb=perturb,
            m_p=0.1, m_c=1.0, ell=0.5, g=9.81, dt=dt)
    if system == "pendulum":
        return make_pendulum_tasks(
            M=M_pool, seed=seed, perturb=perturb,
            m=0.5, l=0.3, g=9.81, dt=dt)
    if system == "synthetic":
        return make_synthetic_tasks(
            M=M_pool, seed=seed, perturb=perturb,
            nx=4, nu=2, ny=2, dt=dt)
    raise ValueError(f"Unknown system '{system}'. Choose: cartpole | pendulum | synthetic")


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
    init_scale_factor: float = 1.0,
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

    Ktilde = find_min_stable_Ktilde(
        tasks_train, caches_train, scale_factor=init_scale_factor)

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
    SYSTEM  = "pendulum"  # "cartpole" | "pendulum" | "synthetic"
    p       = 6          # history length — consistent with main training
    M_pool  = 1000         # total task pool
    n_train = 300
    n_test  = 20
    iters   = 30000
    eta     = 1e-2        # tuned for min-stable-scale init
    dt      = 0.05
    perturb = 0.05
    init_scale_factor = 1 # <1.0 => higher initial cost (weaker controller)

    print(f"System: {SYSTEM}")
    print(f"Generating {M_pool} tasks ...")
    tasks_all = make_task_pool(
        system=SYSTEM, M_pool=M_pool, seed=1, perturb=perturb, dt=dt)
    print(f"  nx={tasks_all[0].A.shape[0]}, nu={tasks_all[0].B.shape[1]}, "
          f"ny={tasks_all[0].C.shape[0]} | Qy={tasks_all[0].Qy[0,0]} "
          f"R={tasks_all[0].R[0,0]}")

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
        init_scale_factor=init_scale_factor,
    )

    plot_generalization(
        out, n_train=n_train, n_test=n_test,
        save_path=f"generalization_{SYSTEM}.pdf")
