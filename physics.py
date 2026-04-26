"""Physics/model-layer helpers for M1/M2/M3."""
from __future__ import annotations
import numpy as np


def energy_to_temperature(E: np.ndarray) -> np.ndarray:
    """Convert radiation energy to temperature using E = T^4 in the paper units."""
    return np.maximum(E, 1e-30) ** 0.25


def model_alpha(model: str) -> float:
    """Return the paper's alpha parameter for each physical model.

    - M1: energy model, alpha = 1
    - M2: temperature-dominated model, alpha = 0
    - M3: energy model with flux limiting, alpha = 1
    """
    if model == "M2":
        return 0.0
    if model in ("M1", "M3"):
        return 1.0
    raise ValueError(f"Unknown model: {model}")


def storage_quantity(E: np.ndarray, model: str) -> np.ndarray:
    """Return the quantity acted on by the time derivative in the paper model.

    The paper writes the transient term in the form
        d/dt [ (alpha + (1-alpha) E^(-3/4)) E ].

    In simplified form this becomes
        alpha * E + (1-alpha) * E^(1/4).

    Therefore:
    - M1/M3: storage quantity is E
    - M2: storage quantity is E^(1/4), i.e. temperature in the paper units
    """
    alpha = model_alpha(model)
    E_safe = np.maximum(E, 1e-30)
    return alpha * E + (1.0 - alpha) * E_safe ** 0.25


def mass_coefficient(E: np.ndarray, model: str) -> np.ndarray:
    """Return d(storage_quantity)/dE for the frozen time linearization.

    This coefficient is not used to define the nonlinear residual directly.
    Instead it is the derivative of the paper's storage quantity with respect to
    E and is used in the Picard linearization and in the Picard-type MG
    preconditioner.

    Therefore:
    - M1/M3: dQ/dE = 1
    - M2: dQ/dE = 1/4 * E^(-3/4)
    """
    alpha = model_alpha(model)
    if alpha == 1.0:
        return np.ones_like(E)
    return alpha + 0.25 * (1.0 - alpha) * np.maximum(E, 1e-30) ** (-0.75)


def material_factor(Z: np.ndarray) -> np.ndarray:
    """Material dependence used in the paper examples: D ~ Z^(-3)."""
    return np.maximum(Z, 1e-30) ** (-3.0)


def raw_diffusion_coefficient(E: np.ndarray, Z: np.ndarray, model: str) -> np.ndarray:
    """Return the unfimited diffusion coefficient before flux limiting.

    - M1: D(E) = T
    - M2: D(E) = T^3
    - M3: D(E) = T^3, then later flux limited
    """
    T = energy_to_temperature(E)
    if model == "M1":
        D = T
    elif model in ("M2", "M3"):
        D = T ** 3
    else:
        raise ValueError(f"Unknown model: {model}")
    return D * material_factor(Z)


def wilson_limiter(E_left: np.ndarray, E_right: np.ndarray,
                   D_face_raw: np.ndarray) -> np.ndarray:
    """Apply Wilson's limiter on a face using adjacent cell values.

    For M3, the limiter should be built on faces rather than from a cell-centered
    gradient magnitude. A face-based form closer to the paper is
        D_L(face) = 1 / (1 / D(face) + |E_R - E_L| / E_avg),
    with E_avg = 0.5 * (E_L + E_R).
    """
    E_avg = 0.5 * (np.maximum(E_left, 1e-30) + np.maximum(E_right, 1e-30))
    jump_ratio = np.abs(E_right - E_left) / np.maximum(E_avg, 1e-30)
    return 1.0 / (1.0 / np.maximum(D_face_raw, 1e-30) + jump_ratio)


def diffusion_coefficient(E: np.ndarray, Z: np.ndarray, model: str,
                          gradx: np.ndarray | None = None,
                          grady: np.ndarray | None = None) -> np.ndarray:
    """Return the cell-centered diffusion coefficient.

    For M1/M2 this is the coefficient used directly to build face diffusion.
    For M3 this returns the *unlimited* cell-centered coefficient; the Wilson
    limiter is applied later on faces in the discretization layer because a
    cell-centered limiter is not faithful to the paper's face-based form.
    """
    _ = gradx, grady
    return raw_diffusion_coefficient(E, Z, model)
