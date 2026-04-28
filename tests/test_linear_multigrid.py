from __future__ import annotations

import numpy as np

from discretization import build_boundary_data, compute_face_diffusion
from problem import build_boundary_config, build_grid
from solvers import PicardMGPreconditioner, _apply_picard_operator, cg, gmres


def test_gmres_and_cg_solve_small_spd_system():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    exact = np.linalg.solve(A, b)

    x_gmres, info_gmres = gmres(lambda x: A @ x, b, tol=1e-11, maxiter=10, restart=10)
    x_cg, info_cg = cg(lambda x: A @ x, b, tol=1e-11, maxiter=10)

    assert info_gmres["converged"]
    assert info_cg["converged"]
    np.testing.assert_allclose(x_gmres, exact, atol=1e-10)
    np.testing.assert_allclose(x_cg, exact, atol=1e-10)


def _constant_diffusion_level(n: int = 16, dt: float = 0.05):
    grid = build_grid(n, n)
    D = np.ones((n, n))
    Dx, Dy = compute_face_diffusion(D)
    boundary = build_boundary_config()
    boundary_data = build_boundary_data({"boundary": boundary}, grid, Dx)
    return {
        "mass": np.ones_like(D),
        "D_cell": D,
        "Dx": Dx,
        "Dy": Dy,
        "dx": grid["dx"],
        "dy": grid["dy"],
        "dt": dt,
        "theta": 0.5,
        "boundary_data": boundary_data,
        "smoother": "jacobi",
        "jacobi_omega": 0.8,
        "gs_omega": 1.0,
        "pre_smooths": 3,
        "post_smooths": 3,
    }


def test_picard_operator_is_linear():
    level = _constant_diffusion_level(8)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 8))
    y = rng.normal(size=(8, 8))
    lhs = _apply_picard_operator(2.0 * x - 0.5 * y, level)
    rhs = 2.0 * _apply_picard_operator(x, level) - 0.5 * _apply_picard_operator(y, level)
    np.testing.assert_allclose(lhs, rhs, atol=1e-11)


def test_multigrid_vcycle_reduces_picard_residual():
    n = 16
    grid = build_grid(n, n)
    D = np.ones((n, n))
    Dx, Dy = compute_face_diffusion(D)
    boundary = build_boundary_config()

    precond = PicardMGPreconditioner(
        mass=np.ones_like(D),
        D_cell=D,
        Dx=Dx,
        Dy=Dy,
        dx=grid["dx"],
        dy=grid["dy"],
        dt=0.05,
        theta=0.5,
        boundary_config=boundary,
        smoother="jacobi",
        pre_smooths=3,
        post_smooths=3,
    )

    level = precond.levels[0]
    rng = np.random.default_rng(2)
    rhs = rng.normal(size=(n, n))
    z = precond.apply(rhs.ravel()).reshape(rhs.shape)

    initial = np.linalg.norm(rhs.ravel())
    reduced = np.linalg.norm((rhs - _apply_picard_operator(z, level)).ravel())
    assert reduced < 0.85 * initial


def test_multigrid_preconditioning_reduces_gmres_iterations():
    n = 16
    grid = build_grid(n, n)
    D = np.ones((n, n))
    Dx, Dy = compute_face_diffusion(D)
    boundary = build_boundary_config()
    precond = PicardMGPreconditioner(
        mass=np.ones_like(D),
        D_cell=D,
        Dx=Dx,
        Dy=Dy,
        dx=grid["dx"],
        dy=grid["dy"],
        dt=0.05,
        theta=0.5,
        boundary_config=boundary,
    )
    level = precond.levels[0]
    rng = np.random.default_rng(3)
    rhs = rng.normal(size=(n, n)).ravel()
    matvec = lambda v: _apply_picard_operator(v.reshape(n, n), level).ravel()

    _, plain = gmres(matvec, rhs, tol=1e-8, maxiter=80, restart=80)
    _, mg = gmres(matvec, rhs, M=precond.apply, tol=1e-8, maxiter=80, restart=80)

    assert mg["iters"] < plain["iters"]
