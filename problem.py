"""Problem definition for the 2D multimaterial equilibrium radiation diffusion setup."""
from __future__ import annotations
import numpy as np
from config import RunConfig


def build_grid(nx: int, ny: int, xlim=(0.0, 1.0), ylim=(0.0, 1.0)) -> dict:
    """Build a uniform cell-centered grid on [0, 1] x [0, 1]."""
    x0, x1 = xlim
    y0, y1 = ylim
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    xc = x0 + (np.arange(nx) + 0.5) * dx
    yc = y0 + (np.arange(ny) + 0.5) * dy
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
    return {
        "nx": nx,
        "ny": ny,
        "xlim": xlim,
        "ylim": ylim,
        "dx": dx,
        "dy": dy,
        "xc": xc,
        "yc": yc,
        "Xc": Xc,
        "Yc": Yc,
    }


def build_material_map(grid: dict) -> np.ndarray:
    """Build the paper's multimaterial geometry in atomic number Z.

    Paper geometry:
    - bottom-left rectangle: Z = 20
    - bottom-right rectangle: Z = 100
    - upper-right circle: Z = 50
    - background: Z = 10
    """
    Xc = grid["Xc"]
    Yc = grid["Yc"]
    Z = np.full_like(Xc, 10.0, dtype=float)

    Z[(Xc <= 0.5) & (Yc <= 0.5)] = 20.0
    Z[(Xc >= 0.75) & (Yc <= 0.25)] = 100.0
    Z[((Xc - 0.75) ** 2 + (Yc - 0.75) ** 2) <= 0.15 ** 2] = 50.0
    return Z


def build_boundary_config() -> dict:
    """Return the boundary data used in the paper test problem.

    The paper applies Milne / Robin conditions on the left/right boundaries and
    symmetry conditions on the top/bottom boundaries. The precise discrete
    boundary operator is implemented in the discretization layer.
    """
    return {
        "left": {"type": "milne_robin", "F_inc": 2.5e3},
        "right": {"type": "milne_robin", "F_inc": 0.25},
        "bottom": {"type": "symmetric"},
        "top": {"type": "symmetric"},
    }


def build_initial_energy(grid: dict) -> np.ndarray:
    """Initial condition from the paper: E = 1 everywhere."""
    return np.ones((grid["nx"], grid["ny"]), dtype=float)


def build_problem(run_cfg: RunConfig) -> dict:
    """Bundle the grid, material map, boundary data, and initial state."""
    grid = build_grid(run_cfg.nx, run_cfg.ny)
    return {
        "grid": grid,
        "Z": build_material_map(grid),
        "boundary": build_boundary_config(),
        "E0": build_initial_energy(grid),
    }
