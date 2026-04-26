"""Spatial discretization helpers."""
from __future__ import annotations
import numpy as np
from physics import diffusion_coefficient, raw_diffusion_coefficient, wilson_limiter


def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Harmonic mean used to place discontinuous diffusion on cell faces."""
    return 2.0 * a * b / np.maximum(a + b, 1e-30)


def compute_gradients(E: np.ndarray, grid: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compute cell-centered gradients with central differences in the interior."""
    dx = grid["dx"]
    dy = grid["dy"]
    gradx = np.zeros_like(E)
    grady = np.zeros_like(E)

    gradx[1:-1, :] = (E[2:, :] - E[:-2, :]) / (2.0 * dx)
    grady[:, 1:-1] = (E[:, 2:] - E[:, :-2]) / (2.0 * dy)

    gradx[0, :] = (E[1, :] - E[0, :]) / dx
    gradx[-1, :] = (E[-1, :] - E[-2, :]) / dx
    grady[:, 0] = (E[:, 1] - E[:, 0]) / dy
    grady[:, -1] = (E[:, -1] - E[:, -2]) / dy
    return gradx, grady


def compute_face_diffusion(D_cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return face diffusion coefficients.

    Dx has shape (nx+1, ny), Dy has shape (nx, ny+1).
    """
    nx, ny = D_cell.shape
    Dx = np.zeros((nx + 1, ny), dtype=float)
    Dy = np.zeros((nx, ny + 1), dtype=float)

    Dx[1:nx, :] = harmonic_mean(D_cell[:-1, :], D_cell[1:, :])
    Dx[0, :] = D_cell[0, :]
    Dx[nx, :] = D_cell[-1, :]

    Dy[:, 1:ny] = harmonic_mean(D_cell[:, :-1], D_cell[:, 1:])
    Dy[:, 0] = D_cell[:, 0]
    Dy[:, ny] = D_cell[:, -1]
    return Dx, Dy


def build_raw_face_diffusion(D_cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build unlimited face diffusion coefficients from cell-centered data."""
    return compute_face_diffusion(D_cell)


def apply_wilson_limiter_on_faces(E: np.ndarray,
                                  Dx_raw: np.ndarray,
                                  Dy_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply Wilson's limiter directly on faces for model M3.

    M3 cannot reuse the old cell-centered limiter path because the paper's
    limiter is naturally expressed with adjacent cell values on each face.
    The limited coefficients are therefore formed separately on x- and y-faces.
    """
    nx, ny = E.shape
    Dx = Dx_raw.copy()
    Dy = Dy_raw.copy()

    if nx > 1:
        Dx[1:nx, :] = wilson_limiter(E[:-1, :], E[1:, :], Dx_raw[1:nx, :])
    if ny > 1:
        Dy[:, 1:ny] = wilson_limiter(E[:, :-1], E[:, 1:], Dy_raw[:, 1:ny])

    return Dx, Dy


def _broadcast_boundary_value(value: float | np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"Boundary data must have shape ({n},), got {arr.shape}.")
    return arr.copy()


def _milne_robin_coefficients(D_face: np.ndarray, dx: float,
                              F_inc: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return affine boundary-flux coefficients for the Milne / Robin condition.

    Using a ghost-cell elimination with
        F_inc = 1/2 * D * dE/dn + 1/4 * E_boundary
    the x-face flux can be written as
        flux = alpha - beta * E_cell   (left boundary)
        flux = beta * E_cell - alpha   (right boundary)
    where alpha and beta depend on the frozen face diffusion coefficient.
    """
    F_inc_arr = _broadcast_boundary_value(F_inc, D_face.size)
    denom = 4.0 * np.maximum(D_face, 1e-30) + dx
    alpha = 8.0 * D_face * F_inc_arr / denom
    beta = 2.0 * D_face / denom
    return alpha, beta


def build_boundary_data(problem: dict, grid: dict, Dx: np.ndarray) -> dict:
    """Build reusable discrete boundary data for Robin and symmetric faces."""
    bc = problem["boundary"]

    if bc["left"]["type"] != "milne_robin" or bc["right"]["type"] != "milne_robin":
        raise ValueError("Left/right boundaries must be Milne / Robin boundaries.")
    if bc["bottom"]["type"] != "symmetric" or bc["top"]["type"] != "symmetric":
        raise ValueError("Top/bottom boundaries must be symmetric boundaries.")

    alpha_left, beta_left = _milne_robin_coefficients(Dx[0, :], grid["dx"], bc["left"]["F_inc"])
    alpha_right, beta_right = _milne_robin_coefficients(Dx[-1, :], grid["dx"], bc["right"]["F_inc"])

    return {
        "left": {"type": "milne_robin", "alpha": alpha_left, "beta": beta_left},
        "right": {"type": "milne_robin", "alpha": alpha_right, "beta": beta_right},
        "bottom": {"type": "symmetric"},
        "top": {"type": "symmetric"},
    }


def build_full_boundary_flux(E: np.ndarray, boundary_data: dict) -> dict:
    """Evaluate the full Robin boundary contribution for q = -D grad(E).

    The face quantities returned here are physical diffusive fluxes
        q_x = -D * dE/dx
    evaluated on the boundary faces after ghost-cell elimination of the Milne /
    Robin condition. These q-fluxes are then converted back to the paper's
    operator div(D grad(E)) through
        div(D grad(E)) = -div(q).

    This is the boundary contribution used in the nonlinear residual.
    """
    left = boundary_data["left"]
    right = boundary_data["right"]
    return {
        "left_flux": left["alpha"] - left["beta"] * E[0, :],
        "right_flux": right["beta"] * E[-1, :] - right["alpha"],
        "bottom_symmetric": True,
        "top_symmetric": True,
    }


def build_homogeneous_boundary_flux(E: np.ndarray, boundary_data: dict) -> dict:
    """Evaluate only the homogeneous linear Robin action for q = -D grad(E).

    The affine constants alpha belong in the residual. Linearized operators acting
    on corrections must keep only the beta-dependent homogeneous part so that the
    resulting operator is linear and satisfies A(0) = 0.
    """
    left = boundary_data["left"]
    right = boundary_data["right"]
    return {
        "left_flux": -left["beta"] * E[0, :],
        "right_flux": right["beta"] * E[-1, :] ,
        "bottom_symmetric": True,
        "top_symmetric": True,
    }


def apply_diffusion_from_faces(E: np.ndarray,
                               grid: dict,
                               Dx: np.ndarray,
                               Dy: np.ndarray,
                               boundary_flux: dict | None = None) -> np.ndarray:
    """Apply div(D grad(E)) using vectorized face fluxes."""
    dx = grid["dx"]
    dy = grid["dy"]
    nx, ny = E.shape

    qx = np.zeros((nx + 1, ny), dtype=E.dtype)
    qy = np.zeros((nx, ny + 1), dtype=E.dtype)

    if boundary_flux is not None:
        qx[0, :] = boundary_flux["left_flux"]
        qx[-1, :] = boundary_flux["right_flux"]

    qx[1:nx, :] = -Dx[1:nx, :] * (E[1:nx, :] - E[:-1, :]) / dx

    qy[:, 0] = 0.0
    qy[:, -1] = 0.0
    qy[:, 1:ny] = -Dy[:, 1:ny] * (E[:, 1:ny] - E[:, :-1]) / dy

    return -((qx[1:, :] - qx[:-1, :]) / dx +
             (qy[:, 1:] - qy[:, :-1]) / dy)

def diffusion_operator_split(E_for_grad: np.ndarray,
                             E_for_coeff: np.ndarray,
                             problem: dict,
                             model: str) -> np.ndarray:
    """Apply div(D grad(E)) with D and grad(E) evaluated at different states.

    E_for_coeff:
        state used to compute the temperature/material diffusion coefficient.

    E_for_grad:
        state used in the gradient and boundary flux evaluation.

    This is needed for the paper's midpoint form:
        D((T_old + T_new)/2) * grad((E_old + E_new)/2).
    """
    grid = problem["grid"]
    D_cell, Dx, Dy = build_frozen_diffusion(
        E_for_coeff,
        problem,
        model,
        E_for_limiter=E_for_grad,
    )
    boundary_data = build_boundary_data(problem, grid, Dx)
    boundary_flux = build_full_boundary_flux(E_for_grad, boundary_data)

    return apply_diffusion_from_faces(
        E_for_grad,
        grid,
        Dx,
        Dy,
        boundary_flux=boundary_flux,
    )
def build_frozen_diffusion(E: np.ndarray,
                           problem: dict,
                           model: str,
                           E_for_limiter: np.ndarray | None = None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Freeze the nonlinear diffusion data at the supplied state.

    E is used to build the raw temperature/material diffusion coefficient.
    For M3, E_for_limiter can be supplied separately to evaluate Wilson's
    face limiter on the energy state used by the gradient term.
    """
    Z = problem["Z"]

    if model != "M3":
        D_cell = diffusion_coefficient(E, Z, model)
        Dx, Dy = compute_face_diffusion(D_cell)
        return D_cell, Dx, Dy

    D_raw_cell = raw_diffusion_coefficient(E, Z, model)
    Dx_raw, Dy_raw = build_raw_face_diffusion(D_raw_cell)

    E_limiter = E if E_for_limiter is None else E_for_limiter
    Dx, Dy = apply_wilson_limiter_on_faces(E_limiter, Dx_raw, Dy_raw)

    return D_raw_cell, Dx, Dy


def coarsen_cell_center(a: np.ndarray) -> np.ndarray:
    """Restrict a cell-centered field by 2x2 averaging."""
    nx, ny = a.shape
    return a.reshape(nx // 2, 2, ny // 2, 2).mean(axis=(1, 3))


def diffusion_operator(E: np.ndarray, problem: dict, model: str) -> np.ndarray:
    """Apply the nonlinear diffusion operator using the same state for D and grad(E)."""
    return diffusion_operator_split(
        E_for_grad=E,
        E_for_coeff=E,
        problem=problem,
        model=model,
    )