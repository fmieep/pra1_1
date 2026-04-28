from __future__ import annotations

import numpy as np

from config import SolverConfig, default_run_config
from driver import compute_eta, update_dt
from methods import (
    build_midpoint_states,
    build_picard_linearization,
    jacobian_free_matvec,
    nk2_step,
    residual_nk2,
)
from problem import build_problem


def test_midpoint_states_follow_temperature_midpoint_rule():
    E_old = np.ones((2, 2))
    E_new = np.ones((2, 2)) * 16.0

    E_for_grad, E_for_coeff = build_midpoint_states(E_new, E_old)

    np.testing.assert_allclose(E_for_grad, 8.5)
    np.testing.assert_allclose(E_for_coeff, 1.5**4)


def test_jacobian_free_matvec_matches_quadratic_directional_derivative():
    E = np.array([[1.0, 2.0], [3.0, 4.0]])
    v = np.array([0.5, -0.25, 0.1, -0.2])

    def residual_fn(X):
        return X**2

    approx = jacobian_free_matvec(
        v,
        E,
        residual_fn=residual_fn,
        rho=1e-7,
        eps_mode="normalized",
    )
    exact = (2.0 * E.ravel()) * v
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert rel < 1e-5


def test_picard_linearization_is_linear_and_shape_preserving():
    run_cfg = default_run_config("M1", 8, 8, 0.10)
    problem = build_problem(run_cfg)
    E_old = problem["E0"]
    E_guess = E_old * 1.1
    A_mv, precond = build_picard_linearization(
        E_guess,
        E_old,
        dt=1e-4,
        problem=problem,
        model="M1",
    )

    rng = np.random.default_rng(4)
    x = rng.normal(size=E_old.size)
    y = rng.normal(size=E_old.size)
    np.testing.assert_allclose(A_mv(x + y), A_mv(x) + A_mv(y), rtol=1e-10, atol=1e-10)
    assert precond.apply(x).shape == x.shape


def test_nk2_residual_and_step_converge_on_tiny_grid():
    run_cfg = default_run_config("M1", 4, 4, 0.10)
    problem = build_problem(run_cfg)
    E_old = problem["E0"]
    solver_cfg = SolverConfig(
        method="nk2",
        nonlinear_tol=1e-7,
        linear_tol_factor=1e-2,
        max_nonlinear_iters=20,
        max_linear_iters=40,
        gmres_restart=40,
        damping_norm="linf",
    )

    E_new, stats = nk2_step(E_old, 1e-6, problem, run_cfg, solver_cfg)
    residual = np.linalg.norm(residual_nk2(E_new, E_old, 1e-6, problem, "M1").ravel())

    assert stats["converged"]
    assert residual < solver_cfg.nonlinear_tol
    assert np.all(E_new > 0.0)


def test_eta_and_dt_update_rules():
    E_old = np.ones((2, 2))
    E_new = np.array([[1.0, 1.2], [0.9, 1.0]])
    eta = compute_eta(E_new, E_old, e_floor=1.0)

    assert eta > 0.0
    assert update_dt(1.0, eta_measured=0.05, eta_target=0.10, growth_limit=1.1) == 1.1
    assert update_dt(1.0, eta_measured=0.20, eta_target=0.10, growth_limit=1.1) == 0.5
