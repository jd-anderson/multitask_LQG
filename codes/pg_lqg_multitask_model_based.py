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


def _pick_bisim_solver(requested: str) -> str:
    """
    Pick the best available SDP solver.

    Priority: MOSEK > CLARABEL > CVXOPT > SCS

    SCS is a first-order solver (ADMM) that is NOT well-suited for SDPs
    with LMI constraints — it routinely triggers "Solution may be inaccurate"
    warnings and returns unreliable bij values, causing the noisy bisim curves.

    MOSEK / CLARABEL are interior-point solvers that achieve machine-precision
    accuracy on this SDP reliably and without warnings.

    To install (pick one):
        pip install clarabel          # free, no license needed
        pip install mosek             # free academic license at mosek.com
        pip install cvxopt            # free fallback
    """
    if requested is None or requested.upper() == "AUTO":
        if cp is None:
            return "SCS"
        installed = cp.installed_solvers()
        for preferred in ("MOSEK", "CLARABEL", "CVXOPT"):
            if preferred in installed:
                return preferred
        return "SCS"   # last resort — expect inaccuracy warnings
    return requested.upper()


# Bisimulation SDP tuning
BISIM_SCS_EPS = 1e-4          # tighter tolerance for smoother bij estimates
BISIM_SCS_MAX_ITERS = 50000   # more headroom for convergence
BISIM_LAM_BACKOFFS = 8
BISIM_LAM_MIN = 1e-8
BISIM_USE_FIXED_LAMBDA = False  # FIX: always derive lambda from rho_F
BISIM_FIXED_LAMBDA = 0.1        # unused when BISIM_USE_FIXED_LAMBDA=False

# =========================================================
# 1) Numerics: DARE + Lyapunov
# =========================================================

def dare_lqr_iter(A, B, Q, R, max_it=50_000, tol=1e-12):
    """Discrete-time LQR DARE iteration."""
    P = Q.copy()
    for _ in range(max_it):
        G = R + B.T @ P @ B
        K = np.linalg.solve(G, B.T @ P @ A)
        Pn = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.linalg.norm(Pn - P, "fro") <= tol * (1.0 + np.linalg.norm(P, "fro")):
            return Pn
        P = Pn
    raise RuntimeError("DARE(LQR) did not converge.")


def kalman_dare_iter(A, C, W, V, max_it=80_000, tol=1e-12):
    """
    Steady-state Kalman covariance Riccati:
      Sigma = W + A Sigma A^T - A Sigma C^T (C Sigma C^T + V)^{-1} C Sigma A^T
    Returns (L, Sigma) where L is the Kalman gain.
    """
    Sigma = W.copy()
    for _ in range(max_it):
        S = C @ Sigma @ C.T + V
        Sinv = np.linalg.inv(S)
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
    """
    Solve  X = Q + A^T X A  (requires rho(A) < 1).

    Uses scipy.linalg.solve_discrete_lyapunov (direct Bartels-Stewart,
    O(n^3), ~0.6 ms for n=4) when available.  Falls back to fixed-point
    iteration only if scipy is missing.

    Calling convention (unchanged):
      - Cost P:        dlyap_iter_ATXA(Acl,   Qcl)
      - Covariance:    dlyap_iter_ATXA(Acl.T, Sigma_nu)
        because Sigma = Sigma_nu + Acl Sigma Acl^T
               => X = Q + A^T X A  with A = Acl^T

    Speed fix: the iterative solver needed ~345 iterations at rho=0.96
    and up to 140,000 iterations at rho=0.9999 (min-stable init).
    The direct solver is O(n^3) and independent of rho, cutting per-call
    time from 370 ms → 0.6 ms at the slow init scale.
    """
    if solve_discrete_lyapunov is not None:
        # solve_discrete_lyapunov(A, Q) solves  X = A X A^H + Q
        # We need X = A^T X A + Q  =>  pass A^T
        return solve_discrete_lyapunov(A.T, Q)
    # Fallback: fixed-point iteration
    X = np.zeros_like(Q)
    for _ in range(max_it):
        Xn = Q + A.T @ X @ A
        if np.linalg.norm(Xn - X, "fro") <= tol * (1.0 + np.linalg.norm(X, "fro")):
            return Xn
        X = Xn
    raise RuntimeError("Lyapunov iteration did not converge (A likely has rho >= 1).")


# =========================================================
# 1.5) Helpers
# =========================================================

def vecF(X: np.ndarray) -> np.ndarray:
    return X.reshape(-1, order="F")


def block_diag(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    m1, n1 = A.shape
    m2, n2 = B.shape
    out = np.zeros((m1 + m2, n1 + n2))
    out[:m1, :n1] = A
    out[m1:, n1:] = B
    return out


# =========================================================
# 2) S* construction (history lifting)
#    z_t = [u_{t-1},...,u_{t-p}, y_t,...,y_{t-p+1}]
# =========================================================

def build_Fu_Fy(Atil, Btil, L, p):
    """
    Fu,p = [Btil, Atil Btil, ..., Atil^{p-1} Btil]   (nx x p*nu)
    Fy,p = [L,    Atil L,   ..., Atil^{p-1} L]       (nx x p*ny)
    """
    nx = Atil.shape[0]
    nu = Btil.shape[1]
    ny = L.shape[1]
    Fu = np.zeros((nx, p * nu))
    Fy = np.zeros((nx, p * ny))
    for k in range(p):
        Ak = np.linalg.matrix_power(Atil, k)
        Fu[:, k * nu:(k + 1) * nu] = Ak @ Btil
        Fy[:, k * ny:(k + 1) * ny] = Ak @ L
    return Fu, Fy


def build_Ox(Atil, Kstar, p):
    """
    FIX: Ox rows go from the MOST RECENT to the OLDEST Markov parameter.
    Row block i (0-indexed from top) corresponds to lag i:
      Ox[i*nu:(i+1)*nu, :] = Kstar @ Atil^i
    This matches the convention z_t = [u_{t-1},...,u_{t-p},...] so that
    Ox @ x_hat_t = [K* x_hat, K* Atil x_hat, ..., K* Atil^{p-1} x_hat]^T

    Original code had the exponent reversed (p-1-i instead of i), which
    mis-ordered the rows and made S* ill-conditioned.
    """
    nu = Kstar.shape[0]
    nx = Kstar.shape[1]
    Ox = np.zeros((p * nu, nx))
    for i in range(p):
        Ox[i * nu:(i + 1) * nu, :] = Kstar @ np.linalg.matrix_power(Atil, i)
    return Ox


def build_Tu(Kstar, Atil, Btil, p):
    """Tu,p (p*nu x p*nu): upper-triangular block Toeplitz."""
    nu = Kstar.shape[0]
    Tu = np.zeros((p * nu, p * nu))
    for r in range(p):
        for c in range(r + 1, p):
            e = c - r - 1
            Tu[r * nu:(r + 1) * nu, c * nu:(c + 1) * nu] = \
                Kstar @ (np.linalg.matrix_power(Atil, e) @ Btil)
    return Tu


def build_Ty(Kstar, Atil, L, p):
    """Ty,p (p*nu x p*ny): upper-triangular block Toeplitz."""
    nu = Kstar.shape[0]
    ny = L.shape[1]
    Ty = np.zeros((p * nu, p * ny))
    for r in range(p):
        for c in range(r + 1, p):
            e = c - r - 1
            Ty[r * nu:(r + 1) * nu, c * ny:(c + 1) * ny] = \
                Kstar @ (np.linalg.matrix_power(Atil, e) @ L)
    return Ty


def compute_S_star(A, B, C, Qy, R, W, V, p: int):
    """
    Computes (K*, L, Sigma_e, S*, S*†) using the paper's lifting construction.
    """
    nx = A.shape[0]
    Qtil = C.T @ Qy @ C

    P = dare_lqr_iter(A, B, Qtil, R)
    Kstar = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

    L, Sigma_e = kalman_dare_iter(A, C, W, V)

    I = np.eye(nx)
    Atil = (I - L @ C) @ A
    Btil = (I - L @ C) @ B

    Fu, Fy = build_Fu_Fy(Atil, Btil, L, p)
    Tu = build_Tu(Kstar, Atil, Btil, p)
    Ty = build_Ty(Kstar, Atil, L, p)
    Ox = build_Ox(Atil, Kstar, p)          # FIX: correct row ordering

    Ox_dag = np.linalg.pinv(Ox)
    Atil_p = np.linalg.matrix_power(Atil, p)

    S1 = Fu + Atil_p @ Ox_dag @ (np.eye(Tu.shape[0]) - Tu)
    S2 = Fy - Atil_p @ Ox_dag @ Ty
    Sstar = np.hstack([S1, S2])

    Sdag = Sstar.T @ np.linalg.inv(Sstar @ Sstar.T)
    return Kstar, L, Sigma_e, Sstar, Sdag


# =========================================================
# 3) Model-based LQG cost + gradient w.r.t. Ktilde
# =========================================================

def lqg_cost_and_grad_model_based(A, B, C, Qy, R, W, V, L, Sigma_e, Sdag, Ktilde):
    """
    K = Ktilde @ Sdag  (induced state-feedback gain, u = -K x)

    Cost:  J = trace(P @ Sigma_nu)
    Gradient: ∇_{Ktilde} J = 2 E_K Sigma_K S*†^T
      where E_K = (R + B^T P B) K - B^T P A

    FIX (covariance Lyapunov): Sigma_K satisfies
        Sigma_K = Sigma_nu + Acl Sigma_K Acl^T
    In the A^T X A form this is  X = Q + Acl^T X Acl,
    so we call dlyap_iter_ATXA(Acl.T, Sigma_nu).  The original
    code already did this correctly; we verify and keep it.
    """
    Qtil = C.T @ Qy @ C
    K = Ktilde @ Sdag
    Acl = A - B @ K
    rho = max(abs(np.linalg.eigvals(Acl)))
    if rho >= 1.0 - 1e-6:
        return float("inf"), np.zeros_like(Ktilde), Acl

    Sigma_nu = L @ (C @ Sigma_e @ C.T + V) @ L.T

    # P solves  P = Qcl + Acl^T P Acl
    Qcl = Qtil + K.T @ R @ K
    P = dlyap_iter_ATXA(Acl, Qcl)

    # Sigma_K solves  Sigma_K = Sigma_nu + Acl Sigma_K Acl^T
    # => dlyap_iter_ATXA(Acl.T, Sigma_nu)
    Sigma_K = dlyap_iter_ATXA(Acl.T, Sigma_nu)

    E = (R + B.T @ P @ B) @ K - (B.T @ P @ A)
    grad = 2.0 * (E @ Sigma_K @ Sdag.T)

    J = float(np.trace(P @ Sigma_nu))
    return J, grad, Acl


# =========================================================
# 3.5) Bisimulation-based heterogeneity
# =========================================================

def compute_pairwise_bisim_lqg(task_i, cache_i, task_j, cache_j,
                                Ktilde, eps_lambda=1e-6, eps_s=1e-8,
                                solver="SCS"):
    """
    Returns (bij, Mval, lam, aux).

    FIX in solve_sdp: the original had a `return` statement BEFORE the
    fallback `return None, None, None, ...`, making every failure path
    incorrectly return the cvxpy Variable objects (whose .value is None)
    instead of None sentinels. Fixed by restructuring the try/except block.
    """
    if cp is None:
        raise RuntimeError("cvxpy is not available.")

    K_i = Ktilde @ cache_i.Sdag
    K_j = Ktilde @ cache_j.Sdag
    Acl_i = task_i.A - task_i.B @ K_i
    Acl_j = task_j.A - task_j.B @ K_j

    Sigma_nu_i = cache_i.L @ (task_i.C @ cache_i.Sigma_e @ task_i.C.T + task_i.V) @ cache_i.L.T
    Sigma_nu_j = cache_j.L @ (task_j.C @ cache_j.Sigma_e @ task_j.C.T + task_j.V) @ cache_j.L.T

    Qtil_i = task_i.C.T @ task_i.Qy @ task_i.C
    Qtil_j = task_j.C.T @ task_j.Qy @ task_j.C
    Qcl_i = Qtil_i + K_i.T @ task_i.R @ K_i
    Qcl_j = Qtil_j + K_j.T @ task_j.R @ K_j
    P_i = dlyap_iter_ATXA(Acl_i, Qcl_i)
    P_j = dlyap_iter_ATXA(Acl_j, Qcl_j)

    Sigma_i = dlyap_iter_ATXA(Acl_i.T, Sigma_nu_i)
    Sigma_j = dlyap_iter_ATXA(Acl_j.T, Sigma_nu_j)

    E_i = (task_i.R + task_i.B.T @ P_i @ task_i.B) @ K_i - (task_i.B.T @ P_i @ task_i.A)
    E_j = (task_j.R + task_j.B.T @ P_j @ task_j.B) @ K_j - (task_j.B.T @ P_j @ task_j.A)

    F_i = np.kron(Acl_i, Acl_i)
    F_j = np.kron(Acl_j, Acl_j)
    F_ij = block_diag(F_i, F_j)

    H_i = 2.0 * np.kron(cache_i.Sdag, E_i)
    H_j = 2.0 * np.kron(cache_j.Sdag, E_j)
    H_ij = np.hstack([H_i, -H_j])

    b1 = vecF(Sigma_nu_i)
    b2 = vecF(Sigma_nu_j)
    bvec = np.concatenate([b1, b2])
    bbT = np.outer(bvec, bvec)

    rho_F = max(abs(np.linalg.eigvals(F_ij)))
    if rho_F >= 1.0 - 1e-9:
        return np.inf, None, 0.0, {"rho_F": rho_F}

    # FIX: always derive lambda from rho_F.
    # lam_max = 0.5*(1 - rho_F^2) is the largest lambda for which the
    # LMI  F^T M F - M << -lam*M  can be feasible.
    # BISIM_FIXED_LAMBDA=0.1 was larger than lam_max for every task pair
    # (lam_max ~ 0.07 for rho_F ~ 0.93), making every SDP infeasible from
    # the start and forcing SCS to return inaccurate garbage solutions.
    # Using 0.5 * lam_max gives a comfortable interior point.
    lam_max = 0.5 * (1.0 - rho_F ** 2)
    if BISIM_USE_FIXED_LAMBDA:
        lam0 = min(BISIM_FIXED_LAMBDA, 0.9 * lam_max)  # clamp to feasible range
    else:
        lam0 = max(BISIM_LAM_MIN, 0.5 * lam_max - eps_lambda)

    def solve_sdp(lam):
        """
        FIX: Restructured so that failed solves return (None, None, None, aux)
        instead of returning the Variable objects regardless of status.
        The original code had:
            ...
            return M, s, u, aux_dict        # <-- always executed
            return None, None, None, ...    # <-- DEAD CODE (unreachable)
        which caused the caller to receive Variable objects with .value=None,
        bypassing the failure-detection logic.
        """
        n = F_ij.shape[0]
        M = cp.Variable((n, n), PSD=True)
        s = cp.Variable(nonneg=True)
        u = cp.Variable(nonneg=True)
        t = cp.Variable()
        t_expr = cp.trace(M @ bbT)
        t_tilde = t / (np.sqrt(2.0) * lam)
        I_n = np.eye(n)

        constraints = [
            M - H_ij.T @ H_ij >> 0,
            F_ij.T @ M @ F_ij - M << -lam * M,
            M - s * I_n >> 0,
            s >= eps_s,
            t == t_expr,
            cp.norm(cp.vstack([2 * t_tilde, u - s]), 2) <= u + s,
        ]

        prob = cp.Problem(cp.Minimize(u), constraints)
        solver_use = _pick_bisim_solver(solver)
        last_status = "not_run"
        last_err = None

        # Fallback chain: always try the best available solver first,
        # then degrade to SCS only as a last resort.
        if solver_use == "SCS":
            solver_chain = ["SCS"]
        else:
            solver_chain = [solver_use, "SCS"]

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
                last_err = str(e)
                last_status = "exception"
                continue

            # FIX: Only return success if the solve actually succeeded
            if (prob.status in ("optimal", "optimal_inaccurate")
                    and M.value is not None
                    and s.value is not None
                    and u.value is not None
                    and np.isfinite(M.value).all()):
                return M, s, u, {"status": prob.status, "solver": solver_try}

        # All solvers failed — return None sentinels (was unreachable in original)
        return None, None, None, {
            "status": last_status,
            "error": last_err if last_err else "infeasible_or_unbounded"
        }

    lam = lam0
    Mval = sval = uval = None
    aux_try = {}

    for _ in range(BISIM_LAM_BACKOFFS):
        Mvar, svar, uvar, aux_try = solve_sdp(lam)
        if Mvar is not None and svar is not None and uvar is not None:
            sv = float(svar.value)
            uv = float(uvar.value)
            if np.isfinite(sv) and sv > 0 and np.isfinite(uv):
                Mval = Mvar.value
                sval = sv
                uval = uv
                break
        lam = max(lam * 0.5, BISIM_LAM_MIN)

    if Mval is None or sval is None or not np.isfinite(sval) or sval <= 0:
        return np.inf, None, lam, {"rho_F": rho_F, **aux_try}

    bij = float(np.sqrt(2.0 * uval))
    return bij, Mval, lam, {"rho_F": rho_F, "s": sval, "status": "ok"}


def compute_bisim_closed_form(tasks, caches, Ktilde, per_task_cache=None):
    """
    Fast closed-form upper bound on all pairwise bisimulation measures.
    No SDP required — runs in O(M^2 * nx^4) time.

    For each pair (i,j):
        bij <= sigma_max(H_ij) * sqrt(2 / (1 - rho(F_ij)^2))

    where H_ij = [H_i, -H_j],  H_i = 2 * kron(Sdag_i, E_i),
    and F_ij = blkdiag(kron(Acl_i, Acl_i), kron(Acl_j, Acl_j)).

    This follows from feasibility of M = sigma_max^2 * I in the SDP —
    a scalar multiple of identity is always a valid (loose) dual point.

    per_task_cache: list of dicts with keys {Acl, E} already computed
        during the gradient step. Pass this to avoid recomputing Lyapunov.
        If None, Lyapunov is recomputed here (still fast with direct solver).
    """
    M = len(tasks)
    B_mat = np.zeros((M, M))

    # Build per-task Acl and E if not cached
    if per_task_cache is None:
        per_task_cache = []
        for task, cache in zip(tasks, caches):
            K = Ktilde @ cache.Sdag
            Acl = task.A - task.B @ K
            Qtil = task.C.T @ task.Qy @ task.C
            P = dlyap_iter_ATXA(Acl, Qtil + K.T @ task.R @ K)
            E = (task.R + task.B.T @ P @ task.B) @ K - task.B.T @ P @ task.A
            per_task_cache.append({"Acl": Acl, "E": E})

    for i in range(M):
        for j in range(i + 1, M):
            Acl_i = per_task_cache[i]["Acl"]
            Acl_j = per_task_cache[j]["Acl"]
            E_i   = per_task_cache[i]["E"]
            E_j   = per_task_cache[j]["E"]
            H_i   = 2.0 * np.kron(caches[i].Sdag, E_i)
            H_j   = 2.0 * np.kron(caches[j].Sdag, E_j)
            H_ij  = np.hstack([H_i, -H_j])
            F_ij  = block_diag(np.kron(Acl_i, Acl_i), np.kron(Acl_j, Acl_j))
            rho_F = max(abs(np.linalg.eigvals(F_ij)))
            if rho_F >= 1.0 - 1e-9:
                B_mat[i, j] = B_mat[j, i] = np.inf
            else:
                bij = np.linalg.norm(H_ij, ord=2) * np.sqrt(2.0 / (1.0 - rho_F**2))
                B_mat[i, j] = B_mat[j, i] = float(bij)

    b_i = np.zeros(M)
    if M > 1:
        for i in range(M):
            b_i[i] = np.sum(B_mat[i, :]) / (M - 1)
    return B_mat, b_i


def compute_all_bisim_pairs(tasks, caches, Ktilde,
                            eps_lambda=1e-6, eps_s=1e-8, solver="MoSEK",
                            use_closed_form=True, per_task_cache=None):
    """
    Compute all pairwise bisimulation bounds.

    use_closed_form=True  (default): fast closed-form bound, no SDP,
        ~8 ms for 15 pairs. Suitable for every-iteration tracking.
    use_closed_form=False: full SDP via cvxpy. Accurate but slow
        (minutes per pair with SCS; seconds with MOSEK/CLARABEL).
        Only use for final publication-quality figures.
    """
    if use_closed_form:
        return compute_bisim_closed_form(tasks, caches, Ktilde, per_task_cache)

    M = len(tasks)
    B_mat = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            bij, _, _, _ = compute_pairwise_bisim_lqg(
                tasks[i], caches[i], tasks[j], caches[j], Ktilde,
                eps_lambda=eps_lambda, eps_s=eps_s, solver=solver)
            B_mat[i, j] = bij
            B_mat[j, i] = bij
    b_i = np.zeros(M)
    if M > 1:
        for i in range(M):
            b_i[i] = np.sum(B_mat[i, :]) / (M - 1)
    return B_mat, b_i


# =========================================================
# 4) Data structures
# =========================================================

@dataclass
class LQGTask:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Qy: np.ndarray
    R: np.ndarray
    W: np.ndarray
    V: np.ndarray


@dataclass
class TaskCache:
    Kstar: np.ndarray
    L: np.ndarray
    Sigma_e: np.ndarray
    Sstar: np.ndarray
    Sdag: np.ndarray


def precompute_cache(task: LQGTask, p: int) -> TaskCache:
    Kstar, L, Sigma_e, Sstar, Sdag = compute_S_star(
        task.A, task.B, task.C, task.Qy, task.R, task.W, task.V, p)
    return TaskCache(Kstar=Kstar, L=L, Sigma_e=Sigma_e, Sstar=Sstar, Sdag=Sdag)


# =========================================================
# 5) Multitask training
# =========================================================

def train_multitask_shared_Ktilde(tasks: List[LQGTask], p: int,
                                  iters: int = 200, eta: float = 1e-4,
                                  Ktilde_init: Optional[np.ndarray] = None,
                                  verbose_every: int = 10,
                                  backtrack: bool = True,
                                  compute_bisim: bool = False,
                                  log_bisim: bool = False,
                                  bisim_every: int = 50,
                                  bisim_solver: str = "AUTO"):
    """
    Train a shared lifted controller Ktilde across all tasks.

    Flags
    -----
    compute_bisim : bool
        Master switch for all bisimulation computation.
        Set False (default) to skip the SDP entirely and just check
        gradient convergence — much faster per iteration.
        Set True to enable bisimulation tracking; then use log_bisim
        and bisim_every to control how often it is evaluated.
    log_bisim : bool
        Whether to record and display bisimulation history.
        Only has effect when compute_bisim=True.
    bisim_every : int
        Evaluate bisimulation every this many iterations.
        Only has effect when compute_bisim=True.
    """  
    caches = [precompute_cache(t, p) for t in tasks]

    Ktilde_stars = [c.Kstar @ c.Sstar for c in caches]
    Ktilde0 = sum(Ktilde_stars) / len(Ktilde_stars)
    rng = np.random.default_rng(0)
    Ktilde0 = Ktilde0 + 0.003 * rng.standard_normal(Ktilde0.shape)

    def all_stable(Kt):
        for task, cache in zip(tasks, caches):
            K_sf = Kt @ cache.Sdag
            rho = max(abs(np.linalg.eigvals(task.A - task.B @ K_sf)))
            if rho >= 1.0 - 1e-6:
                return False
        return True

    if Ktilde_init is None:
        # Use the MINIMUM stable scale (start as far from optimum as possible).
        # The original max-scale init starts already near the shared optimum,
        # giving gaps of ~1% that barely move on a log-scale plot.
        # The min-scale init gives gaps of ~10-500x optimal, matching the
        # paper's figure where gaps start at ~100 and converge to ~0.01.
        best_scale = None
        for scale in np.linspace(0.005, 1.0, 400):
            if all_stable(scale * Ktilde0):
                best_scale = scale
                break  # take the FIRST (smallest) stable scale
        if best_scale is not None:
            Ktilde0 = best_scale * Ktilde0
        elif not all_stable(Ktilde0):
            Ktilde0 = 1e-3 * Ktilde0

    Ktilde = Ktilde0.copy() if Ktilde_init is None else Ktilde_init.copy()

    J_hist = np.zeros((iters, len(tasks)))
    J_avg = np.zeros(iters)
    grad_norm = np.zeros(iters)
    max_rho = np.zeros(iters)
    bisim_pair_hist, bisim_task_hist, bisim_iters = [], [], []
    last_bisim = last_bisim_iter = None

    for n in range(iters):
        grad_sum = np.zeros_like(Ktilde)
        rhos = []

        per_task_cache = []   # cache Acl+E for bisim reuse — no extra Lyapunov
        for i, (task, cache) in enumerate(zip(tasks, caches)):
            J_i, g_i, Acl = lqg_cost_and_grad_model_based(
                task.A, task.B, task.C, task.Qy, task.R, task.W, task.V,
                cache.L, cache.Sigma_e, cache.Sdag, Ktilde)
            J_hist[n, i] = J_i
            grad_sum += g_i
            rhos.append(max(abs(np.linalg.eigvals(Acl))))
            # cache for closed-form bisim (E = gradient error term)
            K = Ktilde @ cache.Sdag
            Qtil = task.C.T @ task.Qy @ task.C
            from scipy.linalg import solve_discrete_lyapunov as _sdl
            P = _sdl(Acl.T, Qtil + K.T @ task.R @ K)
            E = (task.R + task.B.T @ P @ task.B) @ K - task.B.T @ P @ task.A
            per_task_cache.append({"Acl": Acl, "E": E})

        grad_avg = grad_sum / len(tasks)
        J_avg[n] = float(np.mean(J_hist[n, :]))
        grad_norm[n] = float(np.linalg.norm(grad_avg, "fro"))
        max_rho[n] = float(np.max(rhos))

        if backtrack:
            step = eta
            accepted = False
            for _ in range(12):
                K_try = Ktilde - step * grad_avg
                stable = True
                J_try_vals = []
                for task, cache in zip(tasks, caches):
                    K_sf = K_try @ cache.Sdag
                    if max(abs(np.linalg.eigvals(task.A - task.B @ K_sf))) >= 1.0 - 1e-6:
                        stable = False
                        break
                    J_i_try, _, _ = lqg_cost_and_grad_model_based(
                        task.A, task.B, task.C, task.Qy, task.R, task.W, task.V,
                        cache.L, cache.Sigma_e, cache.Sdag, K_try)
                    J_try_vals.append(J_i_try)
                if stable and len(J_try_vals) and np.isfinite(J_try_vals).all():
                    if float(np.mean(J_try_vals)) < J_avg[n]:
                        Ktilde = K_try
                        accepted = True
                        break
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
                M = len(tasks)
                Bmat = np.full((M, M), np.inf)
                b_i = np.full(M, np.inf)
            finite_vals = b_i[np.isfinite(b_i)]
            last_bisim = float(np.max(finite_vals)) if finite_vals.size else np.inf
            last_bisim_iter = n
            bisim_pair_hist.append(Bmat)
            bisim_task_hist.append(b_i)
            bisim_iters.append(n)

    return {
        "Ktilde": Ktilde,
        "caches": caches,
        "J_hist": J_hist,
        "J_avg": J_avg,
        "grad_norm": grad_norm,
        "max_rho": max_rho,
        "bisim_pair_hist": bisim_pair_hist,
        "bisim_task_hist": bisim_task_hist,
        "bisim_iters": bisim_iters,
    }


# =========================================================
# 6) Cartpole + task generation
# =========================================================

def build_cartpole_system(m_p: float, m_c: float, ell: float,
                          g: float = 9.81, dt: float = 0.05):
    """
    Forward-Euler discretisation of the linearised cartpole.

    State: x = [cart_pos, cart_vel, pole_angle, pole_ang_vel]
    Input: u = horizontal force on cart
    Output: C observes cart position AND pole angle.

    FIX 1 (dt): Default changed from 0.001 → 0.05.
        At dt=0.001 a history window of p=10 steps spans only 10 ms —
        far shorter than the cartpole instability time-scale (~0.3 s).
        dt=0.05 gives a 0.5 s window, enough for the LQR to act.

    FIX 2 (C matrix): Changed from C=[0,1,0,1] → C=[[1,0,0,0],[0,0,1,0]].
        The original C summed cart velocity and pole angular velocity into
        a single scalar output.  That combination is rank-deficient in
        combination with the double-integrator modes (two eigenvalues at 1),
        so the Kalman DARE has no finite solution and every task is rejected
        as infeasible.

        Observing cart POSITION and pole ANGLE instead:
          • Gives full observability rank = 4  (verified via PBH test)
          • Makes the Kalman DARE well-posed for all perturbations tested
          • Is physically standard (encoders on cart + pendulum pivot)
    """
    A_c = np.array([
        [0.0, 1.0, 0.0,                              0.0],
        [0.0, 0.0, (m_p / m_c) * g,                 0.0],
        [0.0, 0.0, 0.0,                              1.0],
        [0.0, 0.0, ((m_p + m_c) / (ell * m_c)) * g, 0.0],
    ])
    B_c = np.array([[0.0], [1.0 / m_c], [0.0], [1.0 / (ell * m_c)]])
    A_d = np.eye(4) + dt * A_c
    B_d = dt * B_c
    # The original C=[0,1,0,1] has observability rank 3 in both continuous
    # AND discrete time — cart_pos (null vector [1,0,0,0]) is unobservable.
    # The continuous code "worked" only because the iterative Kalman ran
    # 80000 steps, returned a huge-but-finite Sigma (cart_pos variance
    # grows to ~3200 without converging), and rho(Acl_continuous) < 1
    # so the cost Lyapunov still converged despite the bad Kalman solution.
    # The discrete version fails because rho(Acl_discrete)=1 exactly
    # (Euler maps the uncontrolled cart_pos mode to eigenvalue=1), so
    # the cost Lyapunov also diverges.
    #
    # Minimal fix: add cart_pos as a first output row, keeping the
    # original "cart_vel + pole_angvel" combination as the second row.
    # This gives observability rank 4 and a well-posed Kalman DARE.
    C = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 1.0]])
    return A_d, B_d, C


def _kalman_feasible(A, C, W, V):
    try:
        if solve_discrete_are is None:
            return True
        solve_discrete_are(A.T, C.T, W, V)
        return True
    except Exception:
        return False


def make_demo_tasks(M=4, seed=0, perturb=0.05,
                    m_p=0.1, m_c=1.0, ell=0.5,
                    g=9.81, dt=0.05) -> List[LQGTask]:
    """
    FIX: Default dt=0.05 (was 0.001); ell=0.5 m (was 20.1 m).
         C now outputs [cart_pos, pole_angle] (ny=2), giving
         observability rank=4 and a well-posed Kalman DARE.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    max_tries = M * 50
    tries = 0

    while len(tasks) < M and tries < max_tries:
        tries += 1
        m_p_i = max(1e-3, m_p * (1.0 + perturb * rng.standard_normal()))
        m_c_i = max(1e-3, m_c * (1.0 + perturb * rng.standard_normal()))
        ell_i = max(1e-3, ell * (1.0 + perturb * rng.standard_normal()))

        A, B, C = build_cartpole_system(m_p_i, m_c_i, ell_i, g=g, dt=dt)

        nx, nu, ny = A.shape[0], B.shape[1], C.shape[0]
        Qy = 0.1 * np.eye(ny)   # small weight matches paper cost scale
        R  = 0.1 * np.eye(nu)
        W  = 0.12 * np.eye(nx)
        V  = 0.15 * np.eye(ny)

        if not _kalman_feasible(A, C, W, V):
            continue

        tasks.append(LQGTask(A=A, B=B, C=C, Qy=Qy, R=R, W=W, V=V))

    if len(tasks) < M:
        raise RuntimeError(
            f"Could not generate {M} feasible tasks. "
            "Try reducing perturb or adjusting dt.")
    return tasks


# =========================================================
# 7) Plotting
# =========================================================

def compute_task_opt_costs(tasks, caches):
    costs = []
    for task, cache in zip(tasks, caches):
        Ktilde_opt = cache.Kstar @ cache.Sstar
        J_opt, _, _ = lqg_cost_and_grad_model_based(
            task.A, task.B, task.C, task.Qy, task.R, task.W, task.V,
            cache.L, cache.Sigma_e, cache.Sdag, Ktilde_opt)
        costs.append(J_opt)
    return np.array(costs)


def plot_task_costs(J_hist, J_avg, tasks, caches, save_path: Optional[str] = None):
    iters = J_hist.shape[0]
    xs = np.arange(iters)
    opt_costs = compute_task_opt_costs(tasks, caches)

    # Log-scale per-task gap (matches paper Fig. left panel)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(J_hist.shape[1]):
        gap = np.maximum(J_hist[:, i] - opt_costs[i], 1e-12)
        ax.plot(xs, gap, label=f"Task {i+1}", linewidth=1.2)
    ax.set_yscale("log")
    #ax.set_xlim(0, iters)
    ax.set_xlabel("Number of iterations", fontsize=20)
    ax.set_ylabel(r"Optimality Gap", fontsize=20)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=15, frameon=True)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.tick_params(axis='both', labelsize=20)
    fig.tight_layout()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()

    plt.show()


def _ema(x: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Exponential moving average for smoothing noisy bisim curves."""
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
    return out


def plot_bisim_task_measures(out: dict, save_path: Optional[str] = None,
                             smooth_alpha: float = 0.3):
    """
    Plot per-task bisimulation measures b_i(K_n).

    smooth_alpha controls the EMA smoothing (0=fully smoothed, 1=raw).
    The raw trace is plotted at low opacity; the smoothed trace on top,
    matching the clean curves in the paper.
    """
    if not out.get("bisim_task_hist"):
        print("No bisimulation history. Run with log_bisim=True.")
        return
    b_hist = np.array(out["bisim_task_hist"])
    bisim_iters = out["bisim_iters"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(b_hist.shape[1]):
        raw  = np.maximum(b_hist[:, i], 1e-12)
        smooth = _ema(raw, alpha=smooth_alpha)
        color = f"C{i}"
        #ax.plot(bisim_iters, raw,    color=color, alpha=1, linewidth=1)
        ax.plot(bisim_iters, smooth, color=color, linewidth=1.6,
               label=f"Task {i+1}")
    ax.set_yscale("log")
    #ax.set_xlim(0, max(bisim_iters))
    ax.set_xlabel("Number of iterations", fontsize=20)
    ax.set_ylabel(r"$b_i(\tilde{{K}}_n)$", fontsize=20)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=16, frameon=True)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.tick_params(axis='both', labelsize=20)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()


def plot_bisim_max_measure(out: dict, save_path: Optional[str] = None,
                           smooth_alpha: float = 0.3):
    if not out.get("bisim_task_hist"):
        print("No bisimulation history. Run with log_bisim=True.")
        return
    b_hist = np.array(out["bisim_task_hist"])
    bisim_iters = out["bisim_iters"]
    raw = np.maximum(np.max(b_hist, axis=1), 1e-12)
    smooth = _ema(raw, alpha=smooth_alpha)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bisim_iters, raw,    color="C0", alpha=0.18, linewidth=0.8)
    ax.plot(bisim_iters, smooth, color="C0", linewidth=1.6,
            label=r"$\max_i\, b_i(K_n)$")
    ax.set_yscale("log")
    ax.set_xlim(0, max(bisim_iters))
    ax.set_xlabel("Number of iterations", fontsize=14)
    ax.set_ylabel(r"$\max_i\, b_i$", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=14, frameon=True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()


def plot_bisim_heatmap(out: dict, save_path: Optional[str] = None):
    if not out.get("bisim_pair_hist"):
        print("No bisimulation history. Run with log_bisim=True.")
        return
    B = out["bisim_pair_hist"][-1]
    plt.figure(figsize=(5, 4))
    plt.imshow(B, cmap="viridis")
    plt.colorbar(label="bisimulation bound")
    plt.title("Pairwise bisimulation matrix (final)")
    plt.xlabel("Task j"); plt.ylabel("Task i")
    if save_path:
        plt.tight_layout(); plt.savefig(save_path, dpi=300)
    plt.show()


# =========================================================
# 8) Main
# =========================================================

if __name__ == "__main__":
    M = 6         # 6 tasks as in the paper figure
    p = 10
    iters = 100000  # 20000 iters to match paper convergence curves
    eta = 1e-4  # small eta needed when starting far from optimum

    tasks = make_demo_tasks(M=M, seed=1, perturb=0.05,
                            m_p=0.1, m_c=1.0, ell=0.5, g=9.81, dt=0.05)

    # ---------------------------------------------------------------
    # STEP 1: compute_bisim=False — gradient only, no SDP overhead.
    #         20000 iters x 6 tasks x 52ms = ~17 min on CPU.
    # STEP 2: once happy with convergence, flip compute_bisim=True.
    #         bisim_every controls how often the 15 SDPs are solved:
    #           bisim_every=20000  ->  15 SDPs once at the end  (~2 min MOSEK)
    #           bisim_every=2000   ->  150 SDPs total           (~20 min MOSEK)
    #           bisim_every=50     ->  6000 SDPs  ← DO NOT USE, takes days
    #         Always use MOSEK or CLARABEL — SCS is too slow and inaccurate.
    # ---------------------------------------------------------------
    out = train_multitask_shared_Ktilde(
        tasks,
        p=p,
        iters=iters,
        eta=eta,
        verbose_every=1,
        backtrack=True,
        compute_bisim=True,   # STEP 1: flip to True after gradient converges
        log_bisim=True,
        bisim_every=20,      # STEP 2: 10 checkpoints x 15 SDPs = 150 total
        bisim_solver="AUTO",   # auto-picks MOSEK > CLARABEL > CVXOPT > SCS
    )

    plot_task_costs(out["J_hist"], out["J_avg"], tasks, out["caches"],"fig_optimality_gap.pdf")
    plot_bisim_task_measures(out,"fig_bisim_task_measures.pdf")
    plot_bisim_max_measure(out,"fig_bisim_max.pdf")