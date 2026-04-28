from __future__ import annotations

import numpy as np

from discretization import (
    apply_diffusion_dirichlet_from_faces,
    apply_wilson_limiter_on_faces,
    compute_face_diffusion,
    harmonic_mean,
)
from problem import build_grid


def test_harmonic_face_diffusion_for_jump_coefficients():
    D = np.array([[1.0, 4.0], [3.0, 12.0]])
    Dx, Dy = compute_face_diffusion(D)

    np.testing.assert_allclose(Dx[1, :], harmonic_mean(D[0, :], D[1, :]))
    np.testing.assert_allclose(Dy[:, 1], harmonic_mean(D[:, 0], D[:, 1]))
    np.testing.assert_allclose(Dx[0, :], D[0, :])
    np.testing.assert_allclose(Dy[:, -1], D[:, -1])


def test_dirichlet_finite_volume_manufactured_solution_converges():
    errors = []
    for n in (16, 32):
        grid = build_grid(n, n)
        X = grid["Xc"]
        Y = grid["Yc"]
        u = np.sin(np.pi * X) * np.sin(np.pi * Y)
        exact = -2.0 * np.pi**2 * u

        D = np.ones_like(u)
        Dx, Dy = compute_face_diffusion(D)
        numerical = apply_diffusion_dirichlet_from_faces(
            u,
            grid,
            Dx,
            Dy,
            boundary_values={
                "left": 0.0,
                "right": 0.0,
                "bottom": 0.0,
                "top": 0.0,
            },
        )
        errors.append(np.linalg.norm((numerical - exact).ravel()) / np.sqrt(u.size))

    assert errors[1] < 0.35 * errors[0]


def test_wilson_limiter_on_faces_reduces_raw_coefficients():
    E = np.array([[1.0, 2.0], [4.0, 8.0]])
    D = np.ones_like(E) * 3.0
    Dx_raw, Dy_raw = compute_face_diffusion(D)
    Dx, Dy = apply_wilson_limiter_on_faces(E, Dx_raw, Dy_raw, build_grid(2, 2))

    assert np.all(Dx <= Dx_raw + 1e-14)
    assert np.all(Dy <= Dy_raw + 1e-14)
    assert np.any(Dx[1:-1, :] < Dx_raw[1:-1, :])
