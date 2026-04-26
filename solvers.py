"""Linear solvers and Picard-type multigrid preconditioning."""
from __future__ import annotations
import numpy as np
from discretization import (
    apply_diffusion_from_faces,
    build_boundary_data,
    build_homogeneous_boundary_flux,
    compute_face_diffusion,
    coarsen_cell_center,
)


def gmres(matvec, b, M=None, x0=None, tol=1e-8, maxiter=100, restart=30):
    """Restarted GMRES for matrix-free operators.

    matvec: callable(x) -> Ax
    M: optional right-preconditioner callable(v) -> M^{-1} v
    Returns (x, info_dict)
    """
    n = b.size
    x = np.zeros(n, dtype=float) if x0 is None else x0.copy()
    residual_history = []
    total_iters = 0

    def apply_M(v):
        return v.copy() if M is None else M(v)

    r = b - matvec(x)
    beta = np.linalg.norm(r)
    residual_history.append(beta)
    if beta < tol:
        return x, {
            "iters": 0,
            "residual_history": residual_history,
            "final_residual_norm": beta,
            "converged": True,
        }

    while total_iters < maxiter:
        V = np.zeros((n, restart + 1), dtype=float)
        Z = np.zeros((n, restart), dtype=float)
        H = np.zeros((restart + 1, restart), dtype=float)
        g = np.zeros(restart + 1, dtype=float)

        beta = np.linalg.norm(r)
        if beta == 0.0:
            return x, {
                "iters": total_iters,
                "residual_history": residual_history,
                "final_residual_norm": residual_history[-1],
                "converged": True,
            }
        V[:, 0] = r / beta
        g[0] = beta

        for j in range(restart):
            Z[:, j] = apply_M(V[:, j])
            w = matvec(Z[:, j])
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w = w - H[i, j] * V[:, i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 0 and j + 1 < restart + 1:
                V[:, j + 1] = w / H[j + 1, j]

            y, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], g[:j + 2], rcond=None)
            x_trial = x + Z[:, :j + 1] @ y
            r_trial = b - matvec(x_trial)
            res = np.linalg.norm(r_trial)
            residual_history.append(res)
            total_iters += 1
            if res < tol:
                return x_trial, {
                    "iters": total_iters,
                    "residual_history": residual_history,
                    "final_residual_norm": res,
                    "converged": True,
                }
            if total_iters >= maxiter:
                return x_trial, {
                    "iters": total_iters,
                    "residual_history": residual_history,
                    "final_residual_norm": res,
                    "converged": False,
                }

        x = x_trial
        r = r_trial

    return x, {
        "iters": total_iters,
        "residual_history": residual_history,
        "final_residual_norm": residual_history[-1],
        "converged": False,
    }


def cg(matvec, b, M=None, x0=None, tol=1e-8, maxiter=100):
    """Conjugate-gradient interface kept for later linear-solver extensions."""
    x = np.zeros_like(b) if x0 is None else x0.copy()
    apply_M = (lambda v: v) if M is None else M
    r = b - matvec(x)
    z = apply_M(r)
    p = z.copy()
    rz_old = np.dot(r, z)
    residual_history = [np.linalg.norm(r)]
    if residual_history[-1] < tol:
        return x, {
            "iters": 0,
            "residual_history": residual_history,
            "final_residual_norm": residual_history[-1],
            "converged": True,
        }
    for k in range(maxiter):
        Ap = matvec(p)
        alpha = rz_old / max(np.dot(p, Ap), 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        residual_history.append(np.linalg.norm(r))
        if residual_history[-1] < tol:
            return x, {
                "iters": k + 1,
                "residual_history": residual_history,
                "final_residual_norm": residual_history[-1],
                "converged": True,
            }
        z = apply_M(r)
        rz_new = np.dot(r, z)
        beta = rz_new / max(rz_old, 1e-30)
        p = z + beta * p
        rz_old = rz_new
    return x, {
        "iters": maxiter,
        "residual_history": residual_history,
        "final_residual_norm": residual_history[-1],
        "converged": False,
    }


def smooth_jacobi(A_mv, diag, rhs, x, omega=0.8, n_sweeps=2):
    """Smoothing stage of the V-cycle: weighted Jacobi relaxation."""
    for _ in range(n_sweeps):
        r = rhs - A_mv(x)
        x = x + omega * r / np.maximum(diag, 1e-30)
    return x

def _gauss_seidel_sweep(level: dict,
                        rhs: np.ndarray,
                        x: np.ndarray,
                        omega: float = 1.0,
                        reverse: bool = False) -> np.ndarray:
    """One lexicographic Gauss-Seidel sweep for the Picard operator.

    The Picard operator has the form

        A x = mass/dt * x - theta * div(D grad x),

    with homogeneous Robin action on left/right boundaries.

    This update uses the current newest available neighbor values, so it is a
    true pointwise Gauss-Seidel sweep rather than a Jacobi update.
    """
    mass = level["mass"]
    diag = level["diag"]
    Dx = level["Dx"]
    Dy = level["Dy"]
    theta = level["theta"]
    dx = level["dx"]
    dy = level["dy"]

    nx, ny = mass.shape
    dx2 = dx * dx
    dy2 = dy * dy

    i_range = range(nx - 1, -1, -1) if reverse else range(nx)
    j_range = range(ny - 1, -1, -1) if reverse else range(ny)

    for i in i_range:
        for j in j_range:
            offdiag = 0.0

            # x-left neighbor
            if i > 0:
                offdiag += -theta * Dx[i, j] / dx2 * x[i - 1, j]

            # x-right neighbor
            if i < nx - 1:
                offdiag += -theta * Dx[i + 1, j] / dx2 * x[i + 1, j]

            # y-bottom neighbor
            if j > 0:
                offdiag += -theta * Dy[i, j] / dy2 * x[i, j - 1]

            # y-top neighbor
            if j < ny - 1:
                offdiag += -theta * Dy[i, j + 1] / dy2 * x[i, j + 1]

            x_new = (rhs[i, j] - offdiag) / max(diag[i, j], 1e-30)

            # omega=1.0 gives standard GS.
            # omega<1.0 gives damped GS.
            x[i, j] = (1.0 - omega) * x[i, j] + omega * x_new

    return x


def smooth_gauss_seidel(level: dict,
                        rhs: np.ndarray,
                        x: np.ndarray,
                        omega: float = 1.0,
                        n_sweeps: int = 2,
                        symmetric: bool = False) -> np.ndarray:
    """Pointwise Gauss-Seidel smoother.

    symmetric=False:
        ordinary lexicographic Gauss-Seidel.

    symmetric=True:
        symmetric Gauss-Seidel, i.e. forward sweep followed by backward sweep.
        This is usually more stable as a smoother/preconditioner.
    """
    x = x.copy()

    for _ in range(n_sweeps):
        x = _gauss_seidel_sweep(level, rhs, x, omega=omega, reverse=False)
        if symmetric:
            x = _gauss_seidel_sweep(level, rhs, x, omega=omega, reverse=True)

    return x

def restrict_full_weighting(r_fine: np.ndarray) -> np.ndarray:
    """Restriction stage of the V-cycle: 2x2 averaging to the coarse grid."""
    nx, ny = r_fine.shape
    return r_fine.reshape(nx // 2, 2, ny // 2, 2).mean(axis=(1, 3))


def prolong_piecewise_constant(e_coarse: np.ndarray) -> np.ndarray:
    """Piecewise-constant prolongation from coarse cells to four fine cells."""
    return np.repeat(np.repeat(e_coarse, 2, axis=0), 2, axis=1)


def _coarsen_face_x(Dx_fine: np.ndarray,
                    dx_fine: float,
                    dy_fine: float) -> np.ndarray:
    """Coarsen x-face coefficients by finite-volume transmissibility merging.

    Dx_fine shape: (nx + 1, ny)
    Dx_coarse shape: (nx/2 + 1, ny/2)

    For an x-face, transmissibility is approximately
        T = D * area / distance = D * dy / dx.

    A coarse vertical face consists of two fine vertical face segments in
    parallel along y. We sum fine transmissibilities and convert back to
    a coarse diffusion coefficient.
    """
    nxp1, ny = Dx_fine.shape
    nx = nxp1 - 1

    if nx % 2 != 0 or ny % 2 != 0:
        raise ValueError("Fine grid must be even in both directions.")

    cx = nx // 2
    cy = ny // 2

    dx_coarse = 2.0 * dx_fine
    dy_coarse = 2.0 * dy_fine

    Dx_coarse = np.zeros((cx + 1, cy), dtype=Dx_fine.dtype)

    # Fine transmissibility per x-face segment.
    Tx_fine = Dx_fine * (dy_fine / dx_coarse)

    # Left/right boundary coarse faces.
    Tx_left = Tx_fine[0, :].reshape(cy, 2).sum(axis=1)
    Tx_right = Tx_fine[-1, :].reshape(cy, 2).sum(axis=1)

    Dx_coarse[0, :] = Tx_left * dx_coarse / dy_coarse
    Dx_coarse[-1, :] = Tx_right * dx_coarse / dy_coarse

    # Interior coarse x-faces lie at fine x-face indices 2, 4, ..., nx-2.
    if cx > 1:
        Tx_mid = Tx_fine[2:nx:2, :].reshape(cx - 1, cy, 2).sum(axis=2)
        Dx_coarse[1:cx, :] = Tx_mid * dx_coarse / dy_coarse

    return Dx_coarse


def _coarsen_face_y(Dy_fine: np.ndarray,
                    dx_fine: float,
                    dy_fine: float) -> np.ndarray:
    """Coarsen y-face coefficients by finite-volume transmissibility merging.

    Dy_fine shape: (nx, ny + 1)
    Dy_coarse shape: (nx/2, ny/2 + 1)

    For a y-face,
        T = D * dx / dy.

    A coarse horizontal face consists of two fine horizontal face segments in
    parallel along x. We sum fine transmissibilities and convert back to D.
    """
    nx, nyp1 = Dy_fine.shape
    ny = nyp1 - 1

    if nx % 2 != 0 or ny % 2 != 0:
        raise ValueError("Fine grid must be even in both directions.")

    cx = nx // 2
    cy = ny // 2

    dx_coarse = 2.0 * dx_fine
    dy_coarse = 2.0 * dy_fine

    Dy_coarse = np.zeros((cx, cy + 1), dtype=Dy_fine.dtype)

    # Fine transmissibility per y-face segment.
    Ty_fine = Dy_fine * (dx_fine / dy_coarse)

    # Bottom/top boundary coarse faces.
    Ty_bottom = Ty_fine[:, 0].reshape(cx, 2).sum(axis=1)
    Ty_top = Ty_fine[:, -1].reshape(cx, 2).sum(axis=1)

    Dy_coarse[:, 0] = Ty_bottom * dy_coarse / dx_coarse
    Dy_coarse[:, -1] = Ty_top * dy_coarse / dx_coarse

    # Interior coarse y-faces lie at fine y-face indices 2, 4, ..., ny-2.
    if cy > 1:
        Ty_mid = Ty_fine[:, 2:ny:2].reshape(cx, 2, cy - 1).sum(axis=1)
        Dy_coarse[:, 1:cy] = Ty_mid * dy_coarse / dx_coarse

    return Dy_coarse

def _grid_from_shape_and_spacing(shape: tuple[int, int], dx: float, dy: float) -> dict:
    return {"dx": dx, "dy": dy, "nx": shape[0], "ny": shape[1]}


def _apply_picard_operator(x: np.ndarray, level: dict) -> np.ndarray:
    """Apply the frozen Picard linearization on one multigrid level."""
    boundary_flux = build_homogeneous_boundary_flux(x, level["boundary_data"])
    diff = apply_diffusion_from_faces(
        x,
        _grid_from_shape_and_spacing(x.shape, level["dx"], level["dy"]),
        level["Dx"],
        level["Dy"],
        boundary_flux=boundary_flux,
    )
    return level["mass"] * x / level["dt"] - level["theta"] * diff
def smooth_level(level: dict,
                 rhs: np.ndarray,
                 x: np.ndarray,
                 n_sweeps: int = 2) -> np.ndarray:
    """Dispatch the smoothing stage according to level['smoother']."""
    smoother = level.get("smoother", "jacobi").lower()

    if smoother == "jacobi":
        A_mv = lambda u: _apply_picard_operator(u, level)
        return smooth_jacobi(
            A_mv,
            level["diag"],
            rhs,
            x,
            omega=level.get("jacobi_omega", 0.8),
            n_sweeps=n_sweeps,
        )

    if smoother in ("gs", "gauss_seidel", "lexicographic_gs"):
        return smooth_gauss_seidel(
            level,
            rhs,
            x,
            omega=level.get("gs_omega", 1.0),
            n_sweeps=n_sweeps,
            symmetric=False,
        )

    if smoother in ("sgs", "symmetric_gs", "symmetric_gauss_seidel"):
        return smooth_gauss_seidel(
            level,
            rhs,
            x,
            omega=level.get("gs_omega", 1.0),
            n_sweeps=n_sweeps,
            symmetric=True,
        )

    raise ValueError(
        f"Unknown MG smoother '{smoother}'. "
        "Use 'jacobi', 'gs', or 'sgs'."
    )

def _picard_operator_diag(level: dict) -> np.ndarray:
    """Approximate diagonal used by the Jacobi smoother."""
    mass = level["mass"]
    nx, ny = mass.shape
    dx = level["dx"]
    dy = level["dy"]
    Dx = level["Dx"]
    Dy = level["Dy"]
    theta = level["theta"]

    left_beta = level["boundary_data"]["left"]["beta"]
    right_beta = level["boundary_data"]["right"]["beta"]

    diag = mass / level["dt"]

    x_contrib = np.zeros_like(mass)
    y_contrib = np.zeros_like(mass)

    dx2 = dx * dx
    dy2 = dy * dy

    # x direction
    x_contrib[0, :] = Dx[1, :] / dx2 + left_beta / dx
    x_contrib[-1, :] = Dx[nx - 1, :] / dx2 + right_beta / dx
    if nx > 2:
        x_contrib[1:-1, :] = (Dx[1:nx - 1, :] + Dx[2:nx, :]) / dx2

    # y direction
    y_contrib[:, 0] = Dy[:, 1] / dy2
    y_contrib[:, -1] = Dy[:, ny - 1] / dy2
    if ny > 2:
        y_contrib[:, 1:-1] = (Dy[:, 1:ny - 1] + Dy[:, 2:ny]) / dy2

    diag = diag + theta * (x_contrib + y_contrib)
    return diag


def _build_mg_levels(mass: np.ndarray, D_cell: np.ndarray, dx: float, dy: float,
                     Dx: np.ndarray | None, Dy: np.ndarray | None,
                     dt: float, theta: float = 0.5, min_coarse_size: int = 4,
                     boundary_config: dict | None = None,
                     smoother: str = "jacobi") -> list[dict]:
    """Construct a simple coefficient hierarchy for Picard-type multigrid."""
    levels = []
    mass_level = mass.copy()
    D_level = D_cell.copy()
    Dx_level = None if Dx is None else Dx.copy()
    Dy_level = None if Dy is None else Dy.copy()
    dx_level = dx
    dy_level = dy

    while True:
        if Dx_level is None or Dy_level is None:
            Dx_level, Dy_level = compute_face_diffusion(D_level)
        grid_level = _grid_from_shape_and_spacing(D_level.shape, dx_level, dy_level)
        level = {
            "mass": mass_level,
            "D_cell": D_level,
            "Dx": Dx_level,
            "Dy": Dy_level,
            "dx": dx_level,
            "dy": dy_level,
            "dt": dt,
            "theta": theta,
            "boundary_data": build_boundary_data({"boundary": boundary_config}, grid_level, Dx_level),
            "smoother": smoother,
            "jacobi_omega": 0.8,
            "gs_omega": 1.0,
        }
        level["diag"] = _picard_operator_diag(level)
        levels.append(level)

        nx, ny = D_level.shape
        if nx < 2 * min_coarse_size or ny < 2 * min_coarse_size or nx % 2 or ny % 2:
            break
        mass_level = coarsen_cell_center(mass_level)
        D_level = coarsen_cell_center(D_level)
        if Dx_level is not None and Dy_level is not None:
            Dx_level = _coarsen_face_x(Dx_level, dx_level, dy_level)
            Dy_level = _coarsen_face_y(Dy_level, dx_level, dy_level)
        dx_level *= 2.0
        dy_level *= 2.0

    return levels


def _assemble_dense_operator(level: dict) -> np.ndarray:
    """Assemble the Picard operator on a small coarse grid as a dense matrix."""
    shape = level["mass"].shape
    n = shape[0] * shape[1]
    A = np.zeros((n, n), dtype=float)

    for k in range(n):
        e = np.zeros(n, dtype=float)
        e[k] = 1.0
        A[:, k] = _apply_picard_operator(e.reshape(shape), level).ravel()

    return A


def coarse_solve(level: dict, rhs: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Solve the coarsest Picard system.

    On the coarsest level, use a dense direct solve. This is closer to a
    standard V-cycle than replacing the coarse solve by a few Jacobi sweeps.
    """
    shape = rhs.shape
    n = rhs.size

    # For 4x4 or 8x8 coarse grids, direct solve is cheap enough.
    if n <= 64:
        A = _assemble_dense_operator(level)
        b = rhs.ravel()

        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)

        return x.reshape(shape)

    # Fallback if the coarsest grid is unexpectedly large.
    return smooth_level(level, rhs, x0, n_sweeps=50)


def v_cycle(levels: list[dict], level_idx: int, rhs: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """One geometric V-cycle for the frozen Picard operator A.

    This approximates A^{-1} rhs and is used as the right preconditioner
    in the Newton-Krylov method.
    """
    level = levels[level_idx]
    A_mv = lambda u: _apply_picard_operator(u, level)

    # 1. Pre-smoothing
    x = smooth_level(level, rhs, x0, n_sweeps=2)

    # 2. Coarsest-grid solve
    if level_idx == len(levels) - 1 or min(rhs.shape) <= 4:
        return coarse_solve(level, rhs, x)

    # 3. Residual restriction
    residual = rhs - A_mv(x)
    residual_coarse = restrict_full_weighting(residual)

    # 4. Coarse-grid error equation
    error_coarse_0 = np.zeros_like(residual_coarse)
    error_coarse = v_cycle(levels, level_idx + 1, residual_coarse, error_coarse_0)

    # 5. Prolongation correction
    x = x + prolong_piecewise_constant(error_coarse)

    # 6. Post-smoothing
    x = smooth_level(level, rhs, x, n_sweeps=2)

    return x

class PicardMGPreconditioner:
    """Picard-type multigrid preconditioner for Newton-Krylov."""
    def __init__(self, mass: np.ndarray, D_cell: np.ndarray, dx: float, dy: float,
                 dt: float, theta: float = 0.5, boundary_config: dict | None = None,
                 Dx: np.ndarray | None = None, Dy: np.ndarray | None = None,
                 smoother: str = "jacobi"):
        self.shape = mass.shape
        self.smoother = smoother

        if boundary_config is None:
            raise ValueError("PicardMGPreconditioner requires boundary_config.")

        self.levels = _build_mg_levels(
            mass,
            D_cell,
            dx,
            dy,
            Dx,
            Dy,
            dt,
            theta=theta,
            boundary_config=boundary_config,
            smoother=smoother,
        )

    def apply(self, v: np.ndarray) -> np.ndarray:
        """Return a one-V-cycle approximation to the inverse Picard operator."""
        rhs = v.reshape(self.shape)
        x0 = np.zeros_like(rhs)
        out = v_cycle(self.levels, 0, rhs, x0)
        return out.ravel()
