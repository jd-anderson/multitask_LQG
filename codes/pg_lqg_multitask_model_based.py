import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional
from matplotlib.ticker import ScalarFormatter
try:
    import cvxpy as cp
except Exception:
    cp = None

try:
    from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
except Exception:
    solve_discrete_are = None
    solve_discrete_lyapunov = None

from concurrent.futures import ProcessPoolExecutor, as_completed
import os as _os


def _pick_bisim_solver(requested: str) -> str:
    if requested is None or requested.upper() == "AUTO":
        if cp is None:
            return "SCS"
        installed = cp.installed_solvers()
        for preferred in ("MOSEK", "CLARABEL", "CVXOPT"):
            if preferred in installed:
                return preferred
        return "SCS"
    return requested.upper()


BISIM_SCS_EPS          = 1e-4
BISIM_SCS_MAX_ITERS    = 50000
BISIM_LAM_BACKOFFS     = 8
BISIM_LAM_MIN          = 1e-8
BISIM_USE_FIXED_LAMBDA = False
BISIM_FIXED_LAMBDA     = 0.1

# =============================================================
# 1) Numerics
# =============================================================

def dare_lqr_iter(A, B, Q, R, max_it=50_000, tol=1e-12):
    P = Q.copy()
    for _ in range(max_it):
        G  = R + B.T @ P @ B
        K  = np.linalg.solve(G, B.T @ P @ A)
        Pn = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.linalg.norm(Pn - P, "fro") <= tol * (1.0 + np.linalg.norm(P, "fro")):
            return Pn
        P = Pn
    raise RuntimeError("DARE(LQR) did not converge.")


def kalman_dare_iter(A, C, W, V, max_it=80_000, tol=1e-12):
    Sigma = W.copy()
    for _ in range(max_it):
        S      = C @ Sigma @ C.T + V
        Sinv   = np.linalg.inv(S)
        SigmaN = W + A @ Sigma @ A.T - A @ Sigma @ C.T @ Sinv @ C @ Sigma @ A.T
        if np.linalg.norm(SigmaN - Sigma, "fro") <= tol * (1.0 + np.linalg.norm(Sigma, "fro")):
            Sigma = SigmaN
            L = (Sigma @ C.T) @ np.linalg.inv(C @ Sigma @ C.T + V)
            return L, Sigma
        Sigma = SigmaN
    if solve_discrete_are is not None:
        Sigma = solve_discrete_are(A.T, C.T, W, V)
        L = (Sigma @ C.T) @ np.linalg.inv(C @ Sigma @ C.T + V)
        return L, Sigma
    raise RuntimeError("DARE(Kalman) did not converge.")


def dlyap_iter_ATXA(A, Q, max_it=200_000, tol=1e-12):
    """Solve X = Q + A^T X A. Uses scipy direct solver when available."""
    if solve_discrete_lyapunov is not None:
        return solve_discrete_lyapunov(A.T, Q)
    X = np.zeros_like(Q)
    for _ in range(max_it):
        Xn = Q + A.T @ X @ A
        if np.linalg.norm(Xn - X, "fro") <= tol * (1.0 + np.linalg.norm(X, "fro")):
            return Xn
        X = Xn
    raise RuntimeError("Lyapunov iteration did not converge.")


# =============================================================
# 1.5) Helpers
# =============================================================

def vecF(X: np.ndarray) -> np.ndarray:
    return X.reshape(-1, order="F")


def block_diag(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    m1, n1 = A.shape; m2, n2 = B.shape
    out = np.zeros((m1 + m2, n1 + n2))
    out[:m1, :n1] = A; out[m1:, n1:] = B
    return out


def _bisim_eta_zeta_lambda_prime(lam: float):
    """
    Post-SDP slack parameter and derived quantities (Steps 3-4 of note).

    Given contraction parameter lam from the SDP, choose eta at the
    midpoint of its allowed range (0, lam/(1-lam)):

        eta    = lam / (2*(1-lam))
        zeta   = 1 + 1/eta   =  (2-lam)/lam
        lam'   = lam - eta*(1-lam)  =  lam/2

    Both zeta and lam' are strictly positive for any lam in (0,1).
    """
    eta       = (1 / np.sqrt(1.0 - lam))- 1
    zeta      = 1.0 + 1.0 / eta           # = (2 - lam) / lam
    lam_prime = lam - eta * (1.0 - lam)   # = lam / 2
    return eta, zeta, lam_prime


# =============================================================
# 2) S* construction
# =============================================================

def build_Fu_Fy(Atil, Btil, L, p):
    nx = Atil.shape[0]; nu = Btil.shape[1]; ny = L.shape[1]
    Fu = np.zeros((nx, p*nu)); Fy = np.zeros((nx, p*ny))
    for k in range(p):
        Ak = np.linalg.matrix_power(Atil, k)
        Fu[:, k*nu:(k+1)*nu] = Ak @ Btil
        Fy[:, k*ny:(k+1)*ny] = Ak @ L
    return Fu, Fy


def build_Ox(Atil, Kstar, p):
    nu = Kstar.shape[0]; nx = Kstar.shape[1]
    Ox = np.zeros((p*nu, nx))
    for i in range(p):
        Ox[i*nu:(i+1)*nu, :] = Kstar @ np.linalg.matrix_power(Atil, i)
    return Ox


def build_Tu(Kstar, Atil, Btil, p):
    nu = Kstar.shape[0]; Tu = np.zeros((p*nu, p*nu))
    for r in range(p):
        for c in range(r+1, p):
            Tu[r*nu:(r+1)*nu, c*nu:(c+1)*nu] = \
                Kstar @ (np.linalg.matrix_power(Atil, c-r-1) @ Btil)
    return Tu


def build_Ty(Kstar, Atil, L, p):
    nu = Kstar.shape[0]; ny = L.shape[1]; Ty = np.zeros((p*nu, p*ny))
    for r in range(p):
        for c in range(r+1, p):
            Ty[r*nu:(r+1)*nu, c*ny:(c+1)*ny] = \
                Kstar @ (np.linalg.matrix_power(Atil, c-r-1) @ L)
    return Ty


def compute_S_star(A, B, C, Qy, R, W, V, p: int):
    nx   = A.shape[0]; Qtil = C.T @ Qy @ C
    P    = dare_lqr_iter(A, B, Qtil, R)
    Kstar = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    L, Sigma_e = kalman_dare_iter(A, C, W, V)
    I = np.eye(nx)
    Atil = (I - L @ C) @ A; Btil = (I - L @ C) @ B
    Fu, Fy = build_Fu_Fy(Atil, Btil, L, p)
    Tu = build_Tu(Kstar, Atil, Btil, p)
    Ty = build_Ty(Kstar, Atil, L, p)
    Ox = build_Ox(Atil, Kstar, p)
    Ox_dag = np.linalg.pinv(Ox); Atil_p = np.linalg.matrix_power(Atil, p)
    S1 = Fu + Atil_p @ Ox_dag @ (np.eye(Tu.shape[0]) - Tu)
    S2 = Fy - Atil_p @ Ox_dag @ Ty
    Sstar = np.hstack([S1, S2])
    Sdag  = Sstar.T @ np.linalg.inv(Sstar @ Sstar.T)
    return Kstar, L, Sigma_e, Sstar, Sdag


# =============================================================
# 3) Model-based LQG cost + gradient
# =============================================================

def lqg_cost_and_grad_model_based(A, B, C, Qy, R, W, V, L, Sigma_e, Sdag, Ktilde):
    Qtil = C.T @ Qy @ C; K = Ktilde @ Sdag; Acl = A - B @ K
    rho  = max(abs(np.linalg.eigvals(Acl)))
    if rho >= 1.0 - 1e-6:
        return float("inf"), np.zeros_like(Ktilde), Acl
    Sigma_nu = L @ (C @ Sigma_e @ C.T + V) @ L.T
    Qcl      = Qtil + K.T @ R @ K
    P        = dlyap_iter_ATXA(Acl, Qcl)
    Sigma_K  = dlyap_iter_ATXA(Acl.T, Sigma_nu)
    E        = (R + B.T @ P @ B) @ K - (B.T @ P @ A)
    grad     = 2.0 * (E @ Sigma_K @ Sdag.T)
    J        = float(np.trace(P @ Sigma_nu))
    return J, grad, Acl


# =============================================================
# 3.5) Bisimulation heterogeneity  — NEW FORMULATION
# =============================================================

def compute_pairwise_bisim_lqg(task_i, cache_i, task_j, cache_j,
                                Ktilde, eps_lambda=1e-6, eps_s=1e-8,
                                solver="SCS"):
    """
    Pairwise bisimulation b_ij(K̃) — NEW formula from Leo's note.

    WHAT STAYS THE SAME
    -------------------
    • F^{(ij)} = blkdiag(Acl_i⊗Acl_i, Acl_j⊗Acl_j)
    • H^{(ij)} = [H_i, -H_j]  with  H_i = 2*(S*†_i ⊗ E_i)
    • nu^{(ij)} = [vec(Σ_ν^i); vec(Σ_ν^j)]
    • SDP constraints (22a) M >= H^T H  and  (22b) F^T M F - M << -λM

    WHAT CHANGES
    ------------
    SDP objective: minimise trace(M)  [was: minimise SOC epigraph u]
    Final score:   b_ij = ζ · (nu^T M nu) / λ'   [was: sqrt(2u)]

    Post-SDP steps (Steps 3-6 of note):
        η  = λ/(2(1−λ))           [midpoint of allowed (0, λ/(1−λ))]
        ζ  = 1 + 1/η = (2−λ)/λ
        λ' = λ − η(1−λ) = λ/2
        q  = nu^{ij,T} M nu^{ij}
        b_ij = ζ · q / λ'

    Returns
    -------
    (bij, Mval, lam, lam_prime, zeta, eta, aux)
    """
    if cp is None:
        raise RuntimeError("cvxpy is not available.")

    K_i = Ktilde @ cache_i.Sdag; K_j = Ktilde @ cache_j.Sdag
    Acl_i = task_i.A - task_i.B @ K_i
    Acl_j = task_j.A - task_j.B @ K_j

    Sigma_nu_i = cache_i.L @ (task_i.C @ cache_i.Sigma_e @ task_i.C.T + task_i.V) @ cache_i.L.T
    Sigma_nu_j = cache_j.L @ (task_j.C @ cache_j.Sigma_e @ task_j.C.T + task_j.V) @ cache_j.L.T

    Qtil_i = task_i.C.T @ task_i.Qy @ task_i.C
    Qtil_j = task_j.C.T @ task_j.Qy @ task_j.C
    P_i = dlyap_iter_ATXA(Acl_i, Qtil_i + K_i.T @ task_i.R @ K_i)
    P_j = dlyap_iter_ATXA(Acl_j, Qtil_j + K_j.T @ task_j.R @ K_j)

    E_i = (task_i.R + task_i.B.T @ P_i @ task_i.B) @ K_i - task_i.B.T @ P_i @ task_i.A
    E_j = (task_j.R + task_j.B.T @ P_j @ task_j.B) @ K_j - task_j.B.T @ P_j @ task_j.A

    # F^{(ij)} — UNCHANGED
    F_i  = np.kron(Acl_i, Acl_i); F_j = np.kron(Acl_j, Acl_j)
    F_ij = block_diag(F_i, F_j)

    # H^{(ij)} = C_K^{(ij)} — UNCHANGED
    H_i  = 2.0 * np.kron(cache_i.Sdag, E_i)
    H_j  = 2.0 * np.kron(cache_j.Sdag, E_j)
    H_ij = np.hstack([H_i, -H_j])

    # nu^{(ij)} = phi^{(ij)} in paper — UNCHANGED
    nu_ij = np.concatenate([vecF(Sigma_nu_i), vecF(Sigma_nu_j)])

    rho_F = max(abs(np.linalg.eigvals(F_ij)))
    if rho_F >= 1.0 - 1e-9:
        return np.inf, None, 0.0, 0.0, 0.0, 0.0, {"rho_F": rho_F}

    lam_max = 0.5 * (1.0 - rho_F ** 2)
    lam0 = (min(BISIM_FIXED_LAMBDA, 0.9 * lam_max) if BISIM_USE_FIXED_LAMBDA
            else max(BISIM_LAM_MIN, 0.5 * lam_max - eps_lambda))

    def solve_sdp(lam):
        """
        NEW SDP: minimise trace(M) subject to (22a) and (22b) only.
        OLD SDP had extra variables t, u, s and a SOC epigraph constraint;
        b_ij was computed indirectly as sqrt(2*u).  That whole apparatus
        is removed.  The quadratic nu^T M nu is now computed analytically.
        """
        n   = F_ij.shape[0]
        M   = cp.Variable((n, n), PSD=True)
        constraints = [
            M - H_ij.T @ H_ij >> 0,              # (22a) — UNCHANGED
            F_ij.T @ M @ F_ij - M << -lam * M,   # (22b) — UNCHANGED
        ]
        prob = cp.Problem(cp.Minimize(cp.trace(M)), constraints)

        solver_use   = _pick_bisim_solver(solver)
        solver_chain = ["SCS"] if solver_use == "SCS" else [solver_use, "SCS"]
        last_status  = "not_run"; last_err = None

        for solver_try in solver_chain:
            try:
                import warnings as _w
                with _w.catch_warnings():
                    _w.filterwarnings("ignore", category=UserWarning)
                    if solver_try.upper() == "SCS":
                        prob.solve(solver=solver_try, verbose=False,
                                   eps=BISIM_SCS_EPS, max_iters=BISIM_SCS_MAX_ITERS)
                    else:
                        prob.solve(solver=solver_try, verbose=False)
                last_status = prob.status
            except Exception as e:
                last_err = str(e); last_status = "exception"; continue

            if (prob.status in ("optimal", "optimal_inaccurate")
                    and M.value is not None
                    and np.isfinite(M.value).all()):
                return M.value, {"status": prob.status, "solver": solver_try}

        return None, {"status": last_status,
                      "error": last_err if last_err else "infeasible_or_unbounded"}

    lam = lam0; Mval = None; aux_try = {}
    for _ in range(BISIM_LAM_BACKOFFS):
        Mvar, aux_try = solve_sdp(lam)
        if Mvar is not None: Mval = Mvar; break
        lam = max(lam * 0.5, BISIM_LAM_MIN)

    if Mval is None:
        return np.inf, None, lam, 0.0, 0.0, 0.0, {"rho_F": rho_F, **aux_try}

    # Steps 3-6: compute b_ij = zeta * (nu^T M nu) / lambda'
    eta, zeta, lam_prime = _bisim_eta_zeta_lambda_prime(lam)
    quad = float(nu_ij @ Mval @ nu_ij)   # q = nu^{ij,T} M nu^{ij}
    bij  = zeta * quad / lam_prime        # NEW formula

    return bij, Mval, lam, lam_prime, zeta, eta, {
        "rho_F": rho_F, "quad": quad, "status": aux_try.get("status", "ok")}


def compute_bisim_closed_form(tasks, caches, Ktilde, per_task_cache=None):
    """
    Fast closed-form upper bound using the NEW formula.

    Substitutes M = sigma_max(H_ij)^2 * I  (always feasible for the SDP).
    Then  nu^T M nu = sigma_max^2 * ||nu||^2, and:

        b_ij <= zeta * sigma_max(H_ij)^2 * ||nu^{ij}||^2 / lambda'

    OLD formula: bij = sigma_max(H_ij) * sqrt(2/(1-rho_F^2))
    NEW formula: bij = zeta * sigma_max^2 * ||nu||^2 / lam'
    """
    M_tasks = len(tasks)
    B_mat   = np.zeros((M_tasks, M_tasks))

    if per_task_cache is None:
        per_task_cache = []
        for task, cache in zip(tasks, caches):
            K    = Ktilde @ cache.Sdag; Acl = task.A - task.B @ K
            Qtil = task.C.T @ task.Qy @ task.C
            P    = dlyap_iter_ATXA(Acl, Qtil + K.T @ task.R @ K)
            E    = (task.R + task.B.T @ P @ task.B) @ K - task.B.T @ P @ task.A
            per_task_cache.append({"Acl": Acl, "E": E})

    for i in range(M_tasks):
        for j in range(i + 1, M_tasks):
            Acl_i = per_task_cache[i]["Acl"]; Acl_j = per_task_cache[j]["Acl"]
            E_i   = per_task_cache[i]["E"];   E_j   = per_task_cache[j]["E"]

            H_i   = 2.0 * np.kron(caches[i].Sdag, E_i)
            H_j   = 2.0 * np.kron(caches[j].Sdag, E_j)
            H_ij  = np.hstack([H_i, -H_j])
            F_ij  = block_diag(np.kron(Acl_i, Acl_i), np.kron(Acl_j, Acl_j))
            rho_F = max(abs(np.linalg.eigvals(F_ij)))

            if rho_F >= 1.0 - 1e-9:
                B_mat[i, j] = B_mat[j, i] = np.inf; continue

            # nu^{ij}
            Sigma_nu_i = (caches[i].L
                          @ (tasks[i].C @ caches[i].Sigma_e @ tasks[i].C.T + tasks[i].V)
                          @ caches[i].L.T)
            Sigma_nu_j = (caches[j].L
                          @ (tasks[j].C @ caches[j].Sigma_e @ tasks[j].C.T + tasks[j].V)
                          @ caches[j].L.T)
            nu_ij = np.concatenate([vecF(Sigma_nu_i), vecF(Sigma_nu_j)])

            # Feasible point: M = sigma_max^2 * I
            sigma_max = float(np.linalg.norm(H_ij, ord=2))
            quad_cf   = sigma_max**2 * float(np.dot(nu_ij, nu_ij))

            # lam same derivation as SDP path (use 0.5*lam_max)
            lam_cf = max(BISIM_LAM_MIN, 0.5 * 0.5 * (1.0 - rho_F**2) - 1e-6)
            eta_cf, zeta_cf, lam_prime_cf = _bisim_eta_zeta_lambda_prime(lam_cf)

            # NEW formula
            bij = zeta_cf * quad_cf / lam_prime_cf
            B_mat[i, j] = B_mat[j, i] = float(bij)

    b_i = np.zeros(M_tasks)
    if M_tasks > 1:
        for i in range(M_tasks):
            b_i[i] = np.sum(B_mat[i, :]) / (M_tasks - 1)
    return B_mat, b_i


def _bisim_worker(args):
    # Module-level worker — must not be a closure so ProcessPoolExecutor can pickle it.
    # args = (i, j, task_i, cache_i, task_j, cache_j, Ktilde, eps_lambda, eps_s, solver)
    i, j, task_i, cache_i, task_j, cache_j, Ktilde, eps_lam, eps_s, solver = args
    bij, _, _, _, _, _, _ = compute_pairwise_bisim_lqg(
        task_i, cache_i, task_j, cache_j, Ktilde,
        eps_lambda=eps_lam, eps_s=eps_s, solver=solver)
    return i, j, bij


def compute_all_bisim_pairs(tasks, caches, Ktilde,
                            eps_lambda=1e-6, eps_s=1e-8, solver="AUTO",
                            use_closed_form=False, per_task_cache=None,
                            n_workers=None):
    """
    Compute all pairwise bisimulation bounds.

    use_closed_form=True  (default): fast closed-form, no SDP (~8 ms total).
    use_closed_form=False: full SDP, parallelised across pairs.

    n_workers : int or None
        Number of parallel worker processes for the SDP path.
        None  -> os.cpu_count()  (use all cores)
        1     -> serial  (safe for debugging / profiling)
        k > 1 -> k parallel workers

    Speed example (M=6, 15 pairs, MOSEK ~5 s/pair):
        n_workers=1   ->  ~75 s  (serial)
        n_workers=4   ->  ~19 s
        n_workers=8   ->   ~9 s

    ProcessPoolExecutor spawns independent Python processes, so
    cvxpy/BLAS state is never shared — safe on Linux, macOS, Windows.
    """
    if use_closed_form:
        return compute_bisim_closed_form(tasks, caches, Ktilde, per_task_cache)

    M_tasks  = len(tasks)
    B_mat    = np.zeros((M_tasks, M_tasks))
    n_w      = n_workers if n_workers is not None else _os.cpu_count()

    pair_args = [
        (i, j, tasks[i], caches[i], tasks[j], caches[j],
         Ktilde, eps_lambda, eps_s, solver)
        for i in range(M_tasks)
        for j in range(i + 1, M_tasks)
    ]

    if n_w == 1 or len(pair_args) <= 1:
        # Serial path — avoids subprocess overhead for tiny problems
        for args in pair_args:
            ii, jj, bij = _bisim_worker(args)
            B_mat[ii, jj] = bij; B_mat[jj, ii] = bij
            print(f"Computed bisim pair ({ii}, {jj}): b_ij = {bij:.4f}")
    else:
        # Parallel path — each SDP in its own process
        with ProcessPoolExecutor(max_workers=n_w) as pool:
            futures = {pool.submit(_bisim_worker, a): a for a in pair_args}
            for future in as_completed(futures):
                try:
                    ii, jj, bij = future.result()
                    B_mat[ii, jj] = bij; B_mat[jj, ii] = bij
                except Exception as e:
                    a = futures[future]
                    print(f"bisim pair ({a[0]},{a[1]}) failed: {e}")
                    B_mat[a[0], a[1]] = B_mat[a[1], a[0]] = np.inf

    b_i = np.zeros(M_tasks)
    if M_tasks > 1:
        for i in range(M_tasks):
            b_i[i] = np.sum(B_mat[i, :]) / (M_tasks - 1)
            print(f"Computed b_i for task {i}: b_i = {b_i[i]:.4f}")
    return B_mat, b_i


# =============================================================
# 4) Data structures
# =============================================================

@dataclass
class LQGTask:
    A: np.ndarray; B: np.ndarray; C: np.ndarray
    Qy: np.ndarray; R: np.ndarray; W: np.ndarray; V: np.ndarray


@dataclass
class TaskCache:
    Kstar: np.ndarray; L: np.ndarray; Sigma_e: np.ndarray
    Sstar: np.ndarray; Sdag: np.ndarray


def precompute_cache(task: LQGTask, p: int) -> TaskCache:
    Kstar, L, Sigma_e, Sstar, Sdag = compute_S_star(
        task.A, task.B, task.C, task.Qy, task.R, task.W, task.V, p)
    return TaskCache(Kstar=Kstar, L=L, Sigma_e=Sigma_e, Sstar=Sstar, Sdag=Sdag)


# =============================================================
# 5) Multitask training
# =============================================================

def train_multitask_shared_Ktilde(tasks: List[LQGTask], p: int,
                                  iters: int = 200, eta: float = 1e-4,
                                  Ktilde_init: Optional[np.ndarray] = None,
                                  verbose_every: int = 10,
                                  backtrack: bool = True,
                                  compute_bisim: bool = False,
                                  log_bisim: bool = False,
                                  bisim_every: int = 50,
                                  bisim_solver: str = "AUTO"):
    caches       = [precompute_cache(t, p) for t in tasks]
    Ktilde_stars = [c.Kstar @ c.Sstar for c in caches]
    Ktilde0      = sum(Ktilde_stars) / len(Ktilde_stars)
    rng          = np.random.default_rng(0)
    Ktilde0      = Ktilde0 + 0.003 * rng.standard_normal(Ktilde0.shape)

    def all_stable(Kt):
        return all(
            max(abs(np.linalg.eigvals(t.A - t.B @ (Kt @ c.Sdag)))) < 1.0 - 1e-6
            for t, c in zip(tasks, caches))

    if Ktilde_init is None:
        best_scale = None
        for scale in np.linspace(0.005, 1.0, 400):
            if all_stable(scale * Ktilde0): best_scale = scale; break
        Ktilde0 = (best_scale * Ktilde0) if best_scale is not None else 1e-3 * Ktilde0

    Ktilde = Ktilde0.copy() if Ktilde_init is None else Ktilde_init.copy()

    J_hist    = np.zeros((iters, len(tasks))); J_avg = np.zeros(iters)
    grad_norm = np.zeros(iters); max_rho = np.zeros(iters)
    bisim_pair_hist, bisim_task_hist, bisim_iters = [], [], []
    last_bisim = last_bisim_iter = None

    for n in range(iters):
        grad_sum = np.zeros_like(Ktilde); rhos = []
        per_task_cache = []

        for i, (task, cache) in enumerate(zip(tasks, caches)):
            J_i, g_i, Acl = lqg_cost_and_grad_model_based(
                task.A, task.B, task.C, task.Qy, task.R, task.W, task.V,
                cache.L, cache.Sigma_e, cache.Sdag, Ktilde)
            J_hist[n, i] = J_i; grad_sum += g_i
            rhos.append(max(abs(np.linalg.eigvals(Acl))))
            K    = Ktilde @ cache.Sdag; Qtil = task.C.T @ task.Qy @ task.C
            from scipy.linalg import solve_discrete_lyapunov as _sdl
            P = _sdl(Acl.T, Qtil + K.T @ task.R @ K)
            E = (task.R + task.B.T @ P @ task.B) @ K - task.B.T @ P @ task.A
            per_task_cache.append({"Acl": Acl, "E": E})

        grad_avg      = grad_sum / len(tasks)
        J_avg[n]      = float(np.mean(J_hist[n, :]))
        grad_norm[n]  = float(np.linalg.norm(grad_avg, "fro"))
        max_rho[n]    = float(np.max(rhos))

        if backtrack:
            step = eta; accepted = False
            for _ in range(12):
                K_try = Ktilde - step * grad_avg; stable = True; J_try_vals = []
                for task, cache in zip(tasks, caches):
                    K_sf = K_try @ cache.Sdag
                    if max(abs(np.linalg.eigvals(task.A - task.B @ K_sf))) >= 1.0 - 1e-6:
                        stable = False; break
                    J_t, _, _ = lqg_cost_and_grad_model_based(
                        task.A, task.B, task.C, task.Qy, task.R, task.W, task.V,
                        cache.L, cache.Sigma_e, cache.Sdag, K_try)
                    J_try_vals.append(J_t)
                if stable and len(J_try_vals) and np.isfinite(J_try_vals).all():
                    if float(np.mean(J_try_vals)) < J_avg[n]:
                        Ktilde = K_try; accepted = True; break
                step *= 0.5
        else:
            Ktilde = Ktilde - eta * grad_avg

        if verbose_every and (n % verbose_every == 0 or n == iters - 1):
            msg = (f"iter {n:4d} | J_avg={J_avg[n]:.6e} | "
                   f"||grad||F={grad_norm[n]:.3e} | max rho(Acl)={max_rho[n]:.4f}")
            if compute_bisim and log_bisim and last_bisim is not None:
                msg += f" | bisim_max={last_bisim:.3e} (iter {last_bisim_iter})"
            print(msg)

        if compute_bisim and log_bisim and (n % bisim_every == 0 or n == iters - 1):
            try:
                Bmat, b_i = compute_all_bisim_pairs(
                    tasks, caches, Ktilde, solver=bisim_solver,
                    per_task_cache=per_task_cache)
            except Exception as e:
                print(f"bisim warning at iter {n}: {e}")
                Mn = len(tasks); Bmat = np.full((Mn,Mn),np.inf); b_i = np.full(Mn,np.inf)
            finite_vals     = b_i[np.isfinite(b_i)]
            last_bisim      = float(np.max(finite_vals)) if finite_vals.size else np.inf
            last_bisim_iter = n
            bisim_pair_hist.append(Bmat); bisim_task_hist.append(b_i); bisim_iters.append(n)

    return {"Ktilde": Ktilde, "caches": caches, "J_hist": J_hist, "J_avg": J_avg,
            "grad_norm": grad_norm, "max_rho": max_rho,
            "bisim_pair_hist": bisim_pair_hist, "bisim_task_hist": bisim_task_hist,
            "bisim_iters": bisim_iters}


# =============================================================
# 6) Cartpole + task generation
# =============================================================

def build_cartpole_system(m_p, m_c, ell, g=9.81, dt=0.05):
    A_c = np.array([[0,1,0,0],[0,0,(m_p/m_c)*g,0],[0,0,0,1],
                    [0,0,((m_p+m_c)/(ell*m_c))*g,0]])
    B_c = np.array([[0],[1/m_c],[0],[1/(ell*m_c)]])
    A_d = np.eye(4) + dt*A_c; B_d = dt*B_c
    C   = np.array([[1.,0.,0.,0.],[0.,1.,0.,1.]])
    return A_d, B_d, C


def _kalman_feasible(A, C, W, V):
    try:
        if solve_discrete_are is None: return True
        solve_discrete_are(A.T, C.T, W, V); return True
    except Exception: return False


def make_demo_tasks(M=4, seed=0, perturb=0.05,
                    m_p=0.1, m_c=1.0, ell=0.5, g=9.81, dt=0.05) -> List[LQGTask]:
    rng = np.random.default_rng(seed); tasks = []; tries = 0
    while len(tasks) < M and tries < M*50:
        tries += 1
        m_p_i = max(1e-3, m_p*(1+perturb*rng.standard_normal()))
        m_c_i = max(1e-3, m_c*(1+perturb*rng.standard_normal()))
        ell_i = max(1e-3, ell*(1+perturb*rng.standard_normal()))
        A, B, C = build_cartpole_system(m_p_i, m_c_i, ell_i, g=g, dt=dt)
        nx, nu, ny = A.shape[0], B.shape[1], C.shape[0]
        Qy=0.1*np.eye(ny); R=0.1*np.eye(nu); W=0.12*np.eye(nx); V=0.15*np.eye(ny)
        if not _kalman_feasible(A, C, W, V): continue
        tasks.append(LQGTask(A=A,B=B,C=C,Qy=Qy,R=R,W=W,V=V))
    if len(tasks) < M:
        raise RuntimeError(f"Could not generate {M} feasible tasks.")
    return tasks


# =============================================================
# 7) Plotting
# =============================================================

def compute_task_opt_costs(tasks, caches):
    costs = []
    for task, cache in zip(tasks, caches):
        Kopt = cache.Kstar @ cache.Sstar
        J, _, _ = lqg_cost_and_grad_model_based(
            task.A,task.B,task.C,task.Qy,task.R,task.W,task.V,
            cache.L,cache.Sigma_e,cache.Sdag,Kopt)
        costs.append(J)
    return np.array(costs)


def plot_task_costs(J_hist, J_avg, tasks, caches,
                    save_path: Optional[str] = "fig_optimality_gap.pdf"):
    iters = J_hist.shape[0]; xs = np.arange(iters)
    opt_costs = compute_task_opt_costs(tasks, caches)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(J_hist.shape[1]):
        gap = np.maximum(J_hist[:,i] - opt_costs[i], 1e-12)
        ax.plot(xs, gap, label=f"Task {i+1}", linewidth=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("Number of iterations", fontsize=20)
    ax.set_ylabel("Optimality Gap", fontsize=20)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=20, frameon=True)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    ax.tick_params(axis="both", labelsize=20)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight"); print(f"Saved: {save_path}")
    plt.show()


def _ema(x: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    out = np.empty_like(x); out[0] = x[0]
    for t in range(1, len(x)): out[t] = alpha*x[t] + (1-alpha)*out[t-1]
    return out


def plot_bisim_task_measures(out: dict,
                             save_path: Optional[str] = "fig_bisim_task_measures.pdf",
                             smooth_alpha: float = 0.3):
    if not out.get("bisim_task_hist"):
        print("No bisimulation history."); return
    b_hist = np.array(out["bisim_task_hist"]); bisim_iters = out["bisim_iters"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(b_hist.shape[1]):
        smooth = _ema(np.maximum(b_hist[:,i], 1e-12), alpha=smooth_alpha)
        ax.plot(bisim_iters, smooth, linewidth=1.4, label=f"Task {i+1}")
    ax.set_yscale("log")
    ax.set_xlabel("Number of iterations", fontsize=20)
    ax.set_ylabel(r"$b_i(\tilde{K}_n)$", fontsize=20)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=20, frameon=True)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    ax.tick_params(axis="both", labelsize=20)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight"); print(f"Saved: {save_path}")
    plt.show()


def plot_bisim_max_measure(out: dict,
                           save_path: Optional[str] = "fig_bisim_max.pdf",
                           smooth_alpha: float = 0.3):
    if not out.get("bisim_task_hist"):
        print("No bisimulation history."); return
    b_hist = np.array(out["bisim_task_hist"]); bisim_iters = out["bisim_iters"]
    raw    = np.maximum(np.max(b_hist, axis=1), 1e-12)
    smooth = _ema(raw, alpha=smooth_alpha)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bisim_iters, smooth, color="C0", linewidth=1.4,
            label=r"$\max_i\, b_i(\tilde{K}_n)$")
    ax.set_yscale("log")
    ax.set_xlabel("Number of iterations", fontsize=20)
    ax.set_ylabel(r"$\max_i\, b_i$", fontsize=20)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=20, frameon=True)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
    ax.tick_params(axis="both", labelsize=20)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight"); print(f"Saved: {save_path}")
    plt.show()


def plot_bisim_heatmap(out: dict, save_path: Optional[str] = None):
    if not out.get("bisim_pair_hist"): print("No bisimulation history."); return
    B = out["bisim_pair_hist"][-1]
    plt.figure(figsize=(5,4)); plt.imshow(B, cmap="viridis")
    plt.colorbar(label=r"$b_{ij}(\tilde{K})$")
    plt.xlabel("Task $j$"); plt.ylabel("Task $i$")
    if save_path: plt.tight_layout(); plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# =============================================================
# 8) Main
# =============================================================

if __name__ == "__main__":
    M = 6; p = 10; iters = 100000; eta = 1e-3
    tasks = make_demo_tasks(M=M, seed=1, perturb=0.05,
                            m_p=0.1, m_c=1.0, ell=0.5, g=9.81, dt=0.05)
    out = train_multitask_shared_Ktilde(
        tasks, p=p, iters=iters, eta=eta,
        verbose_every=1000, backtrack=True,
        compute_bisim=True, log_bisim=True, bisim_every=5000,
        bisim_solver="AUTO")
    plot_task_costs(out["J_hist"], out["J_avg"], tasks, out["caches"],
                    "fig_optimality_gap.pdf")
    plot_bisim_task_measures(out, "fig_bisim_task_measures.pdf")
    plot_bisim_max_measure(out,   "fig_bisim_max.pdf")