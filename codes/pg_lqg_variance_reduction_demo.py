"""
Variance reduction experiment for the multitask lifted LQG gradient estimator.

Shows that averaging the zeroth-order (model-free) gradient estimate across
M tasks reduces the relative RMSE at approximately the 1/sqrt(M) rate.

All dynamics use the fixed cartpole setup from multitask_lqg_fixed.py:
  - dt = 0.05  (was 0.001)
  - C = [[1,0,0,0],[0,1,0,1]]  (fully observable, was rank-deficient)
  - Qy = 0.1*I, R = 0.1*I     (matches paper cost scale)
  - ell = 0.5 m                (was 6.5 / 20.1)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

# ---------------------------------------------------------------------------
# Import from whichever filename you are using.
# If you renamed the file, update the module name below accordingly.
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
            "Cannot find the multitask LQG module. "
            "Place variance_reduction.py in the same folder as your "
            "multitask model-based file, or update the import above."
        ) from e


# =========================================================
# Zeroth-order (model-free) gradient estimator
# =========================================================

def gradient_modelfree(
    task: LQGTask,
    cache: TaskCache,
    Ktilde: np.ndarray,
    r: float = 0.005,
    n_s: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    One-point zeroth-order gradient estimator for the lifted LQG cost.

    Approximates  ∇_{Ktilde} J  using only forward cost evaluations:

        g_hat = (d / (n_s * r)) * sum_{k=1}^{n_s} [J(K + r*U_k) - b] * U_k

    where:
        d   = numel(Ktilde)
        U_k ~ Uniform(unit Frobenius sphere)
        b   = J(K)  -- baseline evaluated once at the current K

    The baseline b = J(K) is a control variate that subtracts the large
    constant offset from each sample, reducing variance without introducing
    bias (since E[b * U_k] = 0 for zero-mean U_k).

    Without the baseline, the raw J(K+rU) values (~100) would dominate
    the gradient signal (~0.1), inflating the estimator variance by
    a factor of (J/||∇J||*r)^2 ~ 10^6.

    Cost evaluations per call: n_s + 1  (n_s forward + 1 baseline).
    Compare to two-point: 2*n_s evaluations for lower variance.

    Parameters
    ----------
    task, cache : LQGTask / TaskCache
        Task definition and precomputed Kalman/LQR quantities.
    Ktilde : (nu, nz) array
        Current lifted controller.
    r : float
        Perturbation radius. Small r -> less bias, more variance.
        Typical range: 1e-3 to 1e-2.
    n_s : int
        Number of perturbation directions.
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    g_hat : (nu, nz) array
        ZO gradient estimate.
    """
    if rng is None:
        rng = np.random.default_rng()

    d = Ktilde.size
    g_hat = np.zeros_like(Ktilde)
    count = 0

    # Baseline: evaluate J at current Ktilde once
    J_base, _, _ = lqg_cost_and_grad_model_based(
        task.A, task.B, task.C, task.Qy, task.R, task.W, task.V,
        cache.L, cache.Sigma_e, cache.Sdag, Ktilde)

    if not np.isfinite(J_base):
        return g_hat   # current point is already unstable

    for _ in range(n_s):
        U = rng.standard_normal(Ktilde.shape)
        U = U / np.linalg.norm(U, "fro")

        J_plus, _, _ = lqg_cost_and_grad_model_based(
            task.A, task.B, task.C, task.Qy, task.R, task.W, task.V,
            cache.L, cache.Sigma_e, cache.Sdag, Ktilde + r * U)

        if np.isfinite(J_plus):
            g_hat += (J_plus - J_base) * U
            count += 1

    if count > 0:
        g_hat *= d / (count * r)   # factor d/(n_s*r), not d/(2*n_s*r)

    return g_hat


# =========================================================
# Helpers
# =========================================================

def find_min_stable_Ktilde(tasks: List[LQGTask],
                           caches: List[TaskCache]) -> np.ndarray:
    """
    Start from the average of per-task optimal lifted controllers and
    find the minimum stable scale — giving the largest initial gap while
    remaining feasible (matches the training code's warm-start).
    """
    Kt_avg = sum(c.Kstar @ c.Sstar for c in caches) / len(caches)

    def all_stable(Kt):
        return all(
            max(abs(np.linalg.eigvals(t.A - t.B @ (Kt @ c.Sdag)))) < 1.0 - 1e-6
            for t, c in zip(tasks, caches))

    for scale in np.linspace(0.005, 1.0, 400):
        if all_stable(scale * Kt_avg):
            return scale * Kt_avg

    # Fallback: full scale or tiny
    if all_stable(Kt_avg):
        return Kt_avg
    return 1e-3 * Kt_avg


def compute_true_grad(task: LQGTask, cache: TaskCache,
                      Ktilde: np.ndarray) -> np.ndarray:
    """Model-based true gradient for one task."""
    _, grad, _ = lqg_cost_and_grad_model_based(
        task.A, task.B, task.C, task.Qy, task.R, task.W, task.V,
        cache.L, cache.Sigma_e, cache.Sdag, Ktilde)
    return grad


# =========================================================
# Main experiment
# =========================================================

def variance_reduction_experiment(
    tasks: List[LQGTask],
    caches: List[TaskCache],
    Ktilde: np.ndarray,
    task_counts: List[int],
    trials: int = 30,
    r: float = 0.005,
    n_s: int = 50,
    seed: int = 0,
) -> List[dict]:
    """
    For each M in task_counts, estimate the relative RMSE of the
    averaged ZO gradient estimator over `trials` Monte Carlo runs.

    Relative RMSE = ||g_hat_avg - g_true_avg||_F / ||g_true_avg||_F

    The true gradient g_true_avg is fixed (computed model-based) at the
    given Ktilde. Only the ZO estimates vary across trials.

    Returns list of dicts with keys: m, rmse_mean, rmse_std.
    """
    rng = np.random.default_rng(seed)

    # Pre-compute ALL true gradients at this fixed Ktilde
    # (avoids recomputing them inside the trial loop)
    print("Pre-computing true gradients ...")
    all_true_grads = []
    for i, (task, cache) in enumerate(zip(tasks, caches)):
        g = compute_true_grad(task, cache, Ktilde)
        all_true_grads.append(g)
    print(f"  Done. Max M available = {len(all_true_grads)}")

    results = []
    for M in task_counts:
        if M > len(tasks):
            print(f"  Skipping M={M} — not enough tasks ({len(tasks)} available)")
            continue

        # True gradient for this M: average of first M task gradients
        g_true_avg = sum(all_true_grads[:M]) / M
        g_norm = np.linalg.norm(g_true_avg, "fro")

        print(f"  M={M:>3} | ||g_true||={g_norm:.4e}", end="", flush=True)

        rmse_trials = []
        for trial in range(trials):
            # Averaged ZO estimate across M tasks
            g_hat_avg = np.zeros_like(Ktilde)
            for task, cache in zip(tasks[:M], caches[:M]):
                g_hat = gradient_modelfree(task, cache, Ktilde,
                                           r=r, n_s=n_s, rng=rng)
                g_hat_avg += g_hat
            g_hat_avg /= M

            rel_err = np.linalg.norm(g_hat_avg - g_true_avg, "fro") / g_norm
            rmse_trials.append(float(rel_err))

        rmse_arr = np.array(rmse_trials)
        rmse_mean = float(np.mean(rmse_arr))
        rmse_std  = float(np.std(rmse_arr))
        print(f" | rel_RMSE = {rmse_mean:.4f} ± {rmse_std:.4f}")

        results.append({
            "m":         M,
            "rmse_mean": rmse_mean,
            "rmse_std":  rmse_std,
        })

    return results


# =========================================================
# Plotting
# =========================================================

def plot_variance_reduction(results: List[dict],
                            save_path: Optional[str] = None):
    """
    Plot relative RMSE vs number of tasks M, with error bars.
    Overlays the theoretical 1/sqrt(M) scaling for reference.
    """
    ms    = np.array([r["m"]         for r in results])
    means = np.array([r["rmse_mean"] for r in results])
    stds  = np.array([r["rmse_std"]  for r in results])

    # 1/sqrt(M) reference line scaled to M=1 value
    ref = means[0] / np.sqrt(ms)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(ms, means, yerr=stds, fmt="o-", capsize=4,
                linewidth=2, label="ZO estimator")
    ax.plot(ms, ref, "k--", linewidth=1.2,
            label=r"$1/\sqrt{N}$ reference")

    ax.set_yscale("log")
    #ax.set_xscale("log")
    ax.set_xlabel("Number of tasks $N$", fontsize=20)
    ax.set_ylabel("Relative gradient RMSE", fontsize=20)
    #ax.set_xticks(ms)
    #ax.set_xticklabels([str(m) for m in ms])
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=14, frameon=True)
    fig.tight_layout()
    ax.tick_params(axis='both', labelsize=20)
    #ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0.1, 1)



    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")
    plt.show()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    p = 10          # history length — consistent with main training
    M_pool = 50     # total task pool size
    perturb = 0.01  # tight perturbation so tasks share a common Ktilde

    print("Generating tasks ...")
    tasks_all = make_demo_tasks(
        M=M_pool, seed=1, perturb=perturb,
        m_p=0.1, m_c=1.0, ell=0.5,   # ell=0.5 m (was 6.5 / 20.1)
        g=9.81, dt=0.05)              # dt=0.05 (was 0.001)
    print(f"  {len(tasks_all)} tasks generated.")
    print(f"  C shape: {tasks_all[0].C.shape}  "
          f"(should be (2,4) for full observability)")

    print("Pre-computing caches ...")
    caches_all = [precompute_cache(t, p) for t in tasks_all]

    print("Finding shared stabilising Ktilde ...")
    Ktilde0 = find_min_stable_Ktilde(tasks_all, caches_all)
    rhos = [max(abs(np.linalg.eigvals(t.A - t.B @ (Ktilde0 @ c.Sdag))))
            for t, c in zip(tasks_all, caches_all)]
    print(f"  max rho(Acl) over all tasks = {max(rhos):.5f}  (must be < 1)")

    # How many tasks are stable under this Ktilde?
    stable_mask = [r < 1.0 - 1e-6 for r in rhos]
    n_stable = sum(stable_mask)
    print(f"  Stable tasks: {n_stable} / {M_pool}")

    # Filter to stable tasks only
    tasks_stable  = [t for t, ok in zip(tasks_all,  stable_mask) if ok]
    caches_stable = [c for c, ok in zip(caches_all, stable_mask) if ok]

    task_counts = [m for m in [1, 2, 5, 10, 20, 40] if m <= n_stable]
    if not task_counts:
        raise RuntimeError(
            f"Not enough stable tasks ({n_stable}) for any task_count. "
            "Reduce perturb or increase M_pool.")

    print(f"\nRunning variance reduction experiment ...")
    print(f"  task_counts = {task_counts}, trials=30, n_s=50, r=0.005\n")

    results = variance_reduction_experiment(
        tasks_stable,
        caches_stable,
        Ktilde0,
        task_counts=task_counts,
        trials=30,
        r=0.005,
        n_s=50,
        seed=123,
    )

    print("\nResults:")
    print(f"{'M':>4}  {'rel_RMSE_mean':>14}  {'rel_RMSE_std':>13}  {'ratio_vs_M1':>12}  {'1/sqrt(M)':>10}")
    base = results[0]["rmse_mean"]
    for res in results:
        M = res["m"]
        print(f"{M:>4}  {res['rmse_mean']:>14.4f}  "
              f"{res['rmse_std']:>13.4f}  "
              f"{base/res['rmse_mean']:>12.2f}x  "
              f"{np.sqrt(M):>10.2f}x")

    plot_variance_reduction(results,"variance_reduction_demo.pdf")