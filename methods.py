"""Nonlinear/time-step methods: Picard and NK2."""

from __future__ import annotations
import numpy as np
from physics import mass_coefficient, storage_quantity, energy_to_temperature
from discretization import (
    apply_diffusion_from_faces,
    build_boundary_data,
    build_homogeneous_boundary_flux,
    build_frozen_diffusion,
    diffusion_operator,
    diffusion_operator_split,
)
from solvers import gmres, PicardMGPreconditioner


def residual_nk2(
    E_new: np.ndarray, E_old: np.ndarray, dt: float, problem: dict, model: str
) -> np.ndarray:
    """Implicit-midpoint residual for a single nonlinear time step.

    Paper-consistent midpoint diffusion:
        [Q(E_new) - Q(E_old)] / dt
        - div( D((T_old + T_new)/2) * grad((E_old + E_new)/2) ) = 0
    """
    E_for_grad, E_for_coeff = build_midpoint_states(E_new, E_old)

    time_term = (storage_quantity(E_new, model) - storage_quantity(E_old, model)) / dt
    diff_term = diffusion_operator_split(
        E_for_grad=E_for_grad,
        E_for_coeff=E_for_coeff,
        problem=problem,
        model=model,
    )
    return time_term - diff_term


def residual_picard_midpoint(
    E_new: np.ndarray, E_old: np.ndarray, dt: float, problem: dict, model: str
) -> np.ndarray:
    """Residual monitored by the Picard midpoint iteration.

    Picard and NK2 target the same midpoint nonlinear equation. The distinction is
    not the residual itself, but the outer iteration:
    - Picard: freeze coefficients at the current iterate and solve an approximate
      linearized correction equation
    - NK2: apply Newton-Krylov to the same residual using Jacobian-free matvecs
    """
    return residual_nk2(E_new, E_old, dt, problem, model)


def jacobian_free_matvec(
    v: np.ndarray,
    E: np.ndarray,
    residual_fn,
    rho: float = 1e-8,
    eps_mode: str = "paper",
    F_base: np.ndarray | None = None,
) -> np.ndarray:
    """Approximate J(E) v with a finite-difference directional derivative."""
    v_norm = np.linalg.norm(v)
    if v_norm == 0.0:
        return np.zeros_like(v)

    if eps_mode == "paper":
        eps = rho * (1.0 + np.linalg.norm(E.ravel()))
    elif eps_mode == "normalized":
        eps = rho * (1.0 + np.linalg.norm(E.ravel())) / v_norm
    else:
        raise ValueError(f"Unknown eps_mode: {eps_mode}")

    if F_base is None:
        F_base = residual_fn(E)

    return (residual_fn(E + eps * v.reshape(E.shape)) - F_base).ravel() / eps


def build_midpoint_states(
    E_new: np.ndarray, E_old: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return states needed by the paper's implicit midpoint diffusion term.

    Paper form:
        D((T_old + T_new) / 2) * grad((E_old + E_new) / 2)

    Therefore:
    - E_for_grad  = 0.5 * (E_old + E_new)
    - E_for_coeff = T_mid^4, where T_mid = 0.5 * (T_old + T_new)
    """
    E_for_grad = 0.5 * (E_old + E_new)

    T_old = energy_to_temperature(E_old)
    T_new = energy_to_temperature(E_new)
    T_mid = 0.5 * (T_old + T_new)

    E_for_coeff = np.maximum(T_mid, 1e-30) ** 4
    return E_for_grad, E_for_coeff


def _flatten(a: np.ndarray) -> np.ndarray:
    return a.ravel().copy()


def _unflatten(v: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return v.reshape(shape).copy()


def _damped_update(
    E: np.ndarray,
    dE: np.ndarray,
    damping_norm: str = "l2",
    enforce_positive: bool = True,
) -> tuple[np.ndarray, float]:
    rel_field = np.abs(dE) / np.maximum(np.abs(E), 1e-30)

    if damping_norm == "l2":
        rel = np.linalg.norm(rel_field.ravel())
    elif damping_norm == "linf":
        rel = np.max(rel_field)
    elif damping_norm == "rms":
        rel = np.sqrt(np.mean(rel_field.ravel() ** 2))
    else:
        raise ValueError(f"Unknown damping_norm: {damping_norm}")

    xi = min(1.0, 1.0 / max(rel, 1e-30))
    E_next = E + xi * dE

    if enforce_positive:
        E_next = np.maximum(E_next, 1e-12)

    return E_next, xi


def _build_step_stats(
    nonlinear_iters: int,
    linear_iters_total: int,
    converged: bool,
    final_residual_norm: float,
) -> dict:
    """Return a consistent step-statistics payload.

    The duplicate key aliases keep the current scaffold compatible with earlier
    callers while exposing the explicit names requested for this stage.
    """
    return {
        "nonlinear_iters": nonlinear_iters,
        "linear_iters_total": linear_iters_total,
        "final_residual_norm": final_residual_norm,
        "converged": converged,
        "linear_iters": linear_iters_total,
        "residual_norm": final_residual_norm,
    }


def build_picard_linearization(
    E_guess: np.ndarray,
    E_old: np.ndarray,
    dt: float,
    problem: dict,
    model: str,
    mg_smoother: str = "jacobi",
) -> tuple[callable, PicardMGPreconditioner]:
    """Build the midpoint Picard-linearized correction operator."""
    E_for_grad, E_for_coeff = build_midpoint_states(E_guess, E_old)

    # 时间项 Q(E_new)-Q(E_old) 对未知量 E_new 线性化，
    # 因此 M2 应该在 E_guess 处取 dQ/dE，而不是在 midpoint 处取。
    time_jacobian = mass_coefficient(E_guess, model)

    D_cell, Dx, Dy = build_frozen_diffusion(
        E_for_coeff,
        problem,
        model,
        E_for_limiter=E_for_grad,
    )

    grid = problem["grid"]
    boundary_data = build_boundary_data(problem, grid, Dx)

    def A_mv(v_flat: np.ndarray) -> np.ndarray:
        v = _unflatten(v_flat, E_guess.shape)
        boundary_flux = build_homogeneous_boundary_flux(v, boundary_data)
        diff = apply_diffusion_from_faces(v, grid, Dx, Dy, boundary_flux=boundary_flux)
        out = time_jacobian * v / dt - 0.5 * diff
        return _flatten(out)

    precond = PicardMGPreconditioner(
        mass=time_jacobian,
        D_cell=D_cell,
        Dx=Dx,
        Dy=Dy,
        dx=grid["dx"],
        dy=grid["dy"],
        dt=dt,
        theta=0.5,
        boundary_config=problem["boundary"],
        smoother=mg_smoother,
    )
    return A_mv, precond


def picard_step(
    E_old: np.ndarray, dt: float, problem: dict, run_cfg, solver_cfg
) -> tuple[np.ndarray, dict]:
    """Advance one time step with a Picard outer iteration.

    On each nonlinear iteration:
    - evaluate the midpoint residual at the current iterate
    - freeze the nonlinear coefficients at that midpoint
    - solve the Picard linearized correction system with GMRES
    - apply a damped update to obtain the next nonlinear iterate
    """
    E = E_old.copy()
    linear_iters_total = 0
    converged = False
    final_residual_norm = np.inf

    for k in range(solver_cfg.max_nonlinear_iters):
        R = residual_picard_midpoint(E, E_old, dt, problem, run_cfg.model)
        final_residual_norm = np.linalg.norm(R.ravel())
        if final_residual_norm < solver_cfg.nonlinear_tol:
            converged = True
            break
        A_mv, precond = build_picard_linearization(
            E,
            E_old,
            dt,
            problem,
            run_cfg.model,
            mg_smoother=getattr(solver_cfg, "mg_smoother", "jacobi"),
        )
        rhs = -R.ravel()
        x, info = gmres(
            A_mv,
            rhs,
            M=precond.apply if solver_cfg.use_multigrid_preconditioner else None,
            tol=solver_cfg.linear_tol_factor * final_residual_norm,
            maxiter=solver_cfg.max_linear_iters,
            restart=solver_cfg.gmres_restart,
        )
        linear_iters_total += info["iters"]
        dE = x.reshape(E.shape)
        E, _ = _damped_update(
            E,
            dE,
            damping_norm=getattr(solver_cfg, "damping_norm", "l2"),
            enforce_positive=getattr(solver_cfg, "enforce_positive_update", True),
        )

        # Re-evaluate the residual *after* the nonlinear update. Without this
        # post-update check, a nearly converged final iterate can still be marked
        # as a failed nonlinear step simply because it used the last available
        # iteration to cross the tolerance.
        R = residual_picard_midpoint(E, E_old, dt, problem, run_cfg.model)
        final_residual_norm = np.linalg.norm(R.ravel())
        if final_residual_norm < solver_cfg.nonlinear_tol:
            converged = True
            break

    return E, _build_step_stats(
        nonlinear_iters=k + 1,
        linear_iters_total=linear_iters_total,
        converged=converged,
        final_residual_norm=final_residual_norm,
    )


def nk2_step(
    E_old: np.ndarray, dt: float, problem: dict, run_cfg, solver_cfg
) -> tuple[np.ndarray, dict]:
    """Advance one time step with Newton-Krylov on the midpoint residual.

    The outer iteration is Newton-like:
    - define F(E_new) from the implicit midpoint discretization
    - approximate J(E) v with a Jacobian-free directional derivative
    - solve the linearized Newton correction equation with GMRES
    - optionally apply a Picard-based multigrid preconditioner
    """
    E = E_old.copy()
    linear_iters_total = 0
    converged = False
    final_residual_norm = np.inf

    for k in range(solver_cfg.max_nonlinear_iters):

        def F(E_state):
            return residual_nk2(E_state, E_old, dt, problem, run_cfg.model)

        R = F(E)
        final_residual_norm = np.linalg.norm(R.ravel())
        if final_residual_norm < solver_cfg.nonlinear_tol:
            converged = True
            break

        def J_mv(v_flat):
            return jacobian_free_matvec(
                v_flat,
                E,
                residual_fn=F,
                rho=solver_cfg.rho_jfnk,
                eps_mode=getattr(solver_cfg, "jfnk_eps_mode", "paper"),
                F_base=R,
            )

        _, precond = build_picard_linearization(
            E,
            E_old,
            dt,
            problem,
            run_cfg.model,
            mg_smoother=getattr(solver_cfg, "mg_smoother", "jacobi"),
        )
        M = precond.apply if solver_cfg.use_multigrid_preconditioner else None

        rhs = -R.ravel()
        dE_flat, info = gmres(
            J_mv,
            rhs,
            M=M,
            tol=solver_cfg.linear_tol_factor * final_residual_norm,
            maxiter=solver_cfg.max_linear_iters,
            restart=solver_cfg.gmres_restart,
        )
        linear_iters_total += info["iters"]
        dE = dE_flat.reshape(E.shape)
        E, _ = _damped_update(
            E,
            dE,
            damping_norm=getattr(solver_cfg, "damping_norm", "l2"),
            enforce_positive=getattr(solver_cfg, "enforce_positive_update", True),
        )

        # Re-evaluate the residual after the damped Newton update for the same
        # reason as in the Picard step: convergence can occur on the final
        # available nonlinear iteration only after the update is applied.
        R = F(E)
        final_residual_norm = np.linalg.norm(R.ravel())
        if final_residual_norm < solver_cfg.nonlinear_tol:
            converged = True
            break

    return E, _build_step_stats(
        nonlinear_iters=k + 1,
        linear_iters_total=linear_iters_total,
        converged=converged,
        final_residual_norm=final_residual_norm,
    )


def solve_one_step(
    E_old: np.ndarray, dt: float, problem: dict, run_cfg, solver_cfg
) -> tuple[np.ndarray, dict]:
    method = solver_cfg.method.lower()
    if method == "picard":
        return picard_step(E_old, dt, problem, run_cfg, solver_cfg)
    if method == "nk2":
        return nk2_step(E_old, dt, problem, run_cfg, solver_cfg)
    raise ValueError(f"Unsupported method: {solver_cfg.method}")
