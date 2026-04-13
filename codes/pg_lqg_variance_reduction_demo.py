"""
Variance reduction experiment for the multitask lifted LQG gradient estimator.

Supports three dynamical systems:
  - "cartpole"  : 4-state cartpole   (nx=4, nu=1, ny=2)
  - "pendulum"  : 2-state inverted pendulum  (nx=2, nu=1, ny=1)
  - "synthetic" : 4-state synthetic LTI  (nx=4, nu=2, ny=2)

Shows that averaging the ZO gradient estimate across M tasks reduces the
relative RMSE at approximately the 1/sqrt(M) rate.

Optional OOD test set: 50 tasks drawn at 2x the training perturbation.
The OOD curve illustrates that higher task heterogeneity inflates the
estimator variance, connecting to the paper's variance bound.
Toggle with  include_ood = True / False  in the main block.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Import — tries both possible filenames.
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
            "Cannot find the multitask LQG module. "
            "Place this file in the same folder as multitask_lqg_fixed.py."
        ) from e


# =========================================================
# Task factory
# =========================================================

def make_tasks(system: str, M: int, seed: int, perturb: float,
               p: int = 10) -> Tuple[List[LQGTask], List[TaskCache]]:
    """
    Generate M tasks + caches for the requested system.

    system : "cartpole" | "pendulum" | "synthetic"
    """
    if system == "cartpole":
        tasks = make_demo_tasks(M=M, seed=seed, perturb=perturb,
                                m_p=0.1, m_c=1.0, ell=0.5, g=9.81, dt=0.05)
    elif system == "pendulum":
        tasks = make_pendulum_tasks(M=M, seed=seed, perturb=perturb,
                                    m=0.5, l=0.3, g=9.81, dt=0.05)
    elif system == "synthetic":
        tasks = make_synthetic_tasks(M=M, seed=seed, perturb=perturb,
                                     nx=4, nu=2, ny=2, dt=0.05)
    else:
        raise ValueError(f"Unknown system '{system}'. "
                         "Choose: cartpole | pendulum | synthetic")
    caches = [precompute_cache(t, p) for t in tasks]
    return tasks, caches


# =========================================================
# ZO gradient estimator
# =========================================================

def gradient_modelfree(
    task: LQGTask, cache: TaskCache, Ktilde: np.ndarray,
    r: float = 0.005, n_s: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    One-point ZO gradient estimator with baseline subtraction.

        g_hat = (d / (n_s * r)) * sum_k [J(K + r*U_k) - J(K)] * U_k

    d = numel(Ktilde),  U_k ~ Uniform(unit Frobenius sphere).
    Baseline J(K) is evaluated once to reduce variance.
    """
    if rng is None:
        rng = np.random.default_rng()

    d = Ktilde.size
    g_hat = np.zeros_like(Ktilde)
    count = 0

    J_base, _, _ = lqg_cost_and_grad_model_based(
        task.A, task.B, task.C, task.Qy, task.R, task.W, task.V,
        cache.L, cache.Sigma_e, cache.Sdag, Ktilde)
    if not np.isfinite(J_base):
        return g_hat

    for _ in range(n_s):
        U = rng.standard_normal(Ktilde.shape)
        U /= np.linalg.norm(U, "fro")
        J_plus, _, _ = lqg_cost_and_grad_model_based(
            task.A, task.B, task.C, task.Qy, task.R, task.W, task.V,
            cache.L, cache.Sigma_e, cache.Sdag, Ktilde + r * U)
        if np.isfinite(J_plus):
            g_hat += (J_plus - J_base) * U
            count += 1

    if count > 0:
        g_hat *= d / (count * r)
    return g_hat


# =========================================================
# Helpers
# =========================================================

def find_min_stable_Ktilde(tasks: List[LQGTask],
                           caches: List[TaskCache]) -> np.ndarray:
    """Smallest scale of averaged K*S* that stabilises all tasks."""
    Kt_avg = sum(c.Kstar @ c.Sstar for c in caches) / len(caches)

    def all_stable(Kt):
        return all(
            max(abs(np.linalg.eigvals(t.A - t.B @ (Kt @ c.Sdag)))) < 1.0 - 1e-6
            for t, c in zip(tasks, caches))

    for scale in np.linspace(0.005, 1.0, 400):
        if all_stable(scale * Kt_avg):
            return scale * Kt_avg
    return Kt_avg if all_stable(Kt_avg) else 1e-3 * Kt_avg


def filter_stable(tasks, caches, Ktilde):
    """Keep only tasks stable under Ktilde."""
    ok = [max(abs(np.linalg.eigvals(t.A - t.B @ (Ktilde @ c.Sdag)))) < 1.0 - 1e-6
          for t, c in zip(tasks, caches)]
    return ([t for t, o in zip(tasks, ok) if o],
            [c for c, o in zip(caches, ok) if o])


# =========================================================
# Core experiment
# =========================================================

def variance_reduction_experiment(
    tasks: List[LQGTask], caches: List[TaskCache], Ktilde: np.ndarray,
    task_counts: List[int], trials: int = 30,
    r: float = 0.005, n_s: int = 50, seed: int = 0, label: str = "",
) -> List[dict]:
    """
    For each M in task_counts, run `trials` Monte Carlo estimates of the
    averaged ZO gradient and record the relative RMSE:

        Relative RMSE = ||g_hat_avg - g_true_avg||_F / ||g_true_avg||_F

    Returns list of dicts: {m, rmse_mean, rmse_std, label}
    """
    rng = np.random.default_rng(seed)
    tag = f"[{label}] " if label else ""

    print(f"{tag}Pre-computing true gradients ({len(tasks)} tasks) ...")
    all_true_grads = [
        lqg_cost_and_grad_model_based(
            t.A, t.B, t.C, t.Qy, t.R, t.W, t.V,
            c.L, c.Sigma_e, c.Sdag, Ktilde)[1]
        for t, c in zip(tasks, caches)
    ]
    print(f"{tag}  Done.")

    results = []
    for M in task_counts:
        if M > len(tasks):
            print(f"{tag}  Skipping M={M} — only {len(tasks)} tasks available")
            continue

        g_true_avg = sum(all_true_grads[:M]) / M
        g_norm     = np.linalg.norm(g_true_avg, "fro")
        print(f"{tag}  M={M:>3} | ||g_true||={g_norm:.3e}", end="", flush=True)

        rmse_trials = []
        for _ in range(trials):
            g_hat_avg = np.zeros_like(Ktilde)
            for t, c in zip(tasks[:M], caches[:M]):
                g_hat_avg += gradient_modelfree(t, c, Ktilde, r=r, n_s=n_s, rng=rng)
            g_hat_avg /= M
            rmse_trials.append(
                float(np.linalg.norm(g_hat_avg - g_true_avg, "fro") / g_norm))

        arr = np.array(rmse_trials)
        print(f" | RMSE = {arr.mean():.4f} ± {arr.std():.4f}")
        results.append({"m": M, "rmse_mean": float(arr.mean()),
                        "rmse_std": float(arr.std()), "label": label})
    return results


# =========================================================
# Plotting
# =========================================================

def plot_variance_reduction(
    results_in: List[dict],
    results_ood: Optional[List[dict]] = None,
    save_path: Optional[str] = None,
    show_ood: bool = True,
):
    """
    Plot relative RMSE vs M with error bars and 1/sqrt(M) reference.

    show_ood : set False to hide the OOD curve without removing the data.
               This is the master toggle — equivalent to passing
               results_ood=None but keeps the data available for printing.
    """
    ms_in    = np.array([r["m"]         for r in results_in])
    means_in = np.array([r["rmse_mean"] for r in results_in])
    stds_in  = np.array([r["rmse_std"]  for r in results_in])
    ref      = means_in[0] / np.sqrt(ms_in)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.errorbar(ms_in, means_in, yerr=stds_in,
                fmt="o-", capsize=4, linewidth=2,
                label="ZO estimator")

    if show_ood and results_ood is not None:
        ms_ood    = np.array([r["m"]         for r in results_ood])
        means_ood = np.array([r["rmse_mean"] for r in results_ood])
        stds_ood  = np.array([r["rmse_std"]  for r in results_ood])
        ax.errorbar(ms_ood, means_ood, yerr=stds_ood,
                    fmt="s--", capsize=4, linewidth=2,
                    label=r"OOD ($2\times$ perturb)")

    ax.plot(ms_in, ref, "k--", linewidth=1.2,
            label=r"$1/\sqrt{N}$ reference")

    ax.set_yscale("log")
    ax.set_xlabel("Number of tasks $N$", fontsize=20)
    ax.set_ylabel("Relative gradient RMSE", fontsize=20)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=14, frameon=True)
    ax.tick_params(axis="both", labelsize=20)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    # -----------------------------------------------------------
    # Configuration — change these to switch system / behaviour
    # -----------------------------------------------------------
    SYSTEM      = "pendulum"   # "cartpole" | "pendulum" | "synthetic"
    p           = 15           # history length
    M_pool      = 2000          # in-distribution task pool size
    M_ood       = 50           # OOD task pool size
    perturb     = 0.01         # in-distribution perturbation level
    task_counts = [1, 2, 5, 10,20,40]
    trials      = 10
    n_s         = 200           # ZO directions per gradient estimate
    r           = 0.01        # ZO perturbation radius

    # Master OOD flag — set False to skip OOD experiment entirely
    include_ood = False

    # -----------------------------------------------------------
    # In-distribution tasks
    # -----------------------------------------------------------
    print(f"System : {SYSTEM}  (perturb={perturb})")
    print(f"Generating {M_pool} in-distribution tasks ...")
    tasks_all, caches_all = make_tasks(SYSTEM, M=M_pool, seed=1,
                                       perturb=perturb, p=p)
    nx = tasks_all[0].A.shape[0]
    nu = tasks_all[0].B.shape[1]
    ny = tasks_all[0].C.shape[0]
    print(f"  nx={nx}, nu={nu}, ny={ny}, lifted dim={p*(nu+ny)}")

    print("Finding shared stabilising Ktilde ...")
    Ktilde0 = find_min_stable_Ktilde(tasks_all, caches_all)
    rhos    = [max(abs(np.linalg.eigvals(t.A - t.B @ (Ktilde0 @ c.Sdag))))
               for t, c in zip(tasks_all, caches_all)]
    print(f"  max rho(Acl) = {max(rhos):.5f}")

    tasks_stable, caches_stable = filter_stable(tasks_all, caches_all, Ktilde0)
    print(f"  Stable tasks : {len(tasks_stable)} / {M_pool}")

    task_counts_in = [m for m in task_counts if m <= len(tasks_stable)]
    if not task_counts_in:
        raise RuntimeError("Not enough stable in-distribution tasks. Reduce perturb.")

    # -----------------------------------------------------------
    # OOD tasks (2x perturbation, independent seed)
    # -----------------------------------------------------------
    results_ood    = None
    task_counts_ood = []
    if include_ood:
        perturb_ood = 2.0 * perturb
        print(f"\nGenerating {M_ood} OOD tasks (perturb={perturb_ood}) ...")
        tasks_ood_all, caches_ood_all = make_tasks(
            SYSTEM, M=M_ood, seed=99, perturb=perturb_ood, p=p)
        tasks_ood_s, caches_ood_s = filter_stable(
            tasks_ood_all, caches_ood_all, Ktilde0)
        print(f"  OOD stable under Ktilde0 : {len(tasks_ood_s)} / {M_ood}")
        task_counts_ood = [m for m in task_counts if m <= len(tasks_ood_s)]

    # -----------------------------------------------------------
    # Run experiments
    # -----------------------------------------------------------
    print(f"\nIn-distribution: task_counts={task_counts_in}, "
          f"trials={trials}, n_s={n_s}, r={r}")
    results_in = variance_reduction_experiment(
        tasks_stable, caches_stable, Ktilde0,
        task_counts=task_counts_in, trials=trials,
        r=r, n_s=n_s, seed=123, label="in-dist")

    if include_ood and task_counts_ood:
        print(f"\nOOD: task_counts={task_counts_ood}")
        results_ood = variance_reduction_experiment(
            tasks_ood_s, caches_ood_s, Ktilde0,
            task_counts=task_counts_ood, trials=trials,
            r=r, n_s=n_s, seed=456, label="OOD")

    # -----------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------
    print("\n--- In-distribution ---")
    base = results_in[0]["rmse_mean"]
    print(f"{'M':>4}  {'RMSE_mean':>10}  {'RMSE_std':>9}  "
          f"{'ratio':>8}  {'1/sqrt(M)':>10}")
    for res in results_in:
        M = res["m"]
        print(f"{M:>4}  {res['rmse_mean']:>10.4f}  {res['rmse_std']:>9.4f}  "
              f"{base/res['rmse_mean']:>8.2f}x  {M**0.5:>10.2f}x")

    if include_ood and results_ood:
        print("\n--- OOD (2x perturb) ---")
        base_ood = results_ood[0]["rmse_mean"]
        for res in results_ood:
            M = res["m"]
            print(f"{M:>4}  {res['rmse_mean']:>10.4f}  {res['rmse_std']:>9.4f}  "
                  f"{base_ood/res['rmse_mean']:>8.2f}x  {M**0.5:>10.2f}x")

    # -----------------------------------------------------------
    # Plot  (show_ood mirrors include_ood but can be set independently)
    # -----------------------------------------------------------
    plot_variance_reduction(
        results_in  = results_in,
        results_ood = results_ood,
        save_path   = f"variance_reduction_{SYSTEM}.pdf",
        show_ood    = include_ood,   # <-- flip False to hide OOD curve
    )