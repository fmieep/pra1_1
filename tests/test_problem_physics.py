from __future__ import annotations

import numpy as np

from config import default_run_config
from physics import (
    diffusion_coefficient,
    energy_to_temperature,
    frozen_storage_coefficient,
    mass_coefficient,
    storage_quantity,
    wilson_limiter,
)
from problem import build_grid, build_material_map, build_problem


def test_grid_material_and_boundary_setup():
    cfg = default_run_config("M1", 16, 16, 0.10)
    problem = build_problem(cfg)

    assert problem["E0"].shape == (16, 16)
    assert problem["grid"]["dx"] == 1.0 / 16
    assert problem["boundary"]["left"]["type"] == "milne_robin"
    assert problem["boundary"]["top"]["type"] == "symmetric"

    grid = build_grid(16, 16)
    Z = build_material_map(grid)
    assert set(np.unique(Z)).issuperset({10.0, 20.0, 50.0, 100.0})


def test_physics_storage_and_mass_models():
    E = np.array([1.0, 16.0])

    np.testing.assert_allclose(energy_to_temperature(E), np.array([1.0, 2.0]))
    np.testing.assert_allclose(storage_quantity(E, "M1"), E)
    np.testing.assert_allclose(storage_quantity(E, "M2"), np.array([1.0, 2.0]))
    np.testing.assert_allclose(mass_coefficient(E, "M1"), np.ones_like(E))
    np.testing.assert_allclose(mass_coefficient(E, "M2"), 0.25 * E ** (-0.75))
    np.testing.assert_allclose(frozen_storage_coefficient(E, "M2"), E ** (-0.75))


def test_diffusion_models_and_wilson_limiter_are_monotone():
    E = np.array([[1.0, 16.0]])
    Z = np.ones_like(E) * 10.0

    D1 = diffusion_coefficient(E, Z, "M1")
    D2 = diffusion_coefficient(E, Z, "M2")
    np.testing.assert_allclose(D1 / (Z ** -3), E ** 0.25)
    np.testing.assert_allclose(D2 / (Z ** -3), E ** 0.75)

    limited = wilson_limiter(
        np.array([1.0]),
        np.array([4.0]),
        np.array([2.0]),
        distance=0.5,
    )
    assert 0.0 < limited[0] < 2.0
