"""Time-integration driver with eta-based step control."""
from __future__ import annotations
import numpy as np
from methods import solve_one_step


def compute_eta(E_new: np.ndarray, E_old: np.ndarray, e_floor: float = 1.0) -> float:
    """Measure the maximum relative change over one time step."""
    ratio = np.abs(E_new - E_old) / (np.maximum(E_new, 0.0) + e_floor)
    return float(np.max(ratio))


def update_dt(dt_old: float, eta_measured: float, eta_target: float,
              growth_limit: float = 1.1) -> float:
    """Update the time step using the measured eta and a growth limiter."""
    if eta_measured <= 1e-30:
        return dt_old * growth_limit
    dt_new = dt_old * (eta_target / eta_measured)
    return max(min(dt_new, dt_old * growth_limit), 1e-30)


def run_simulation(problem: dict, run_cfg, solver_cfg) -> dict:
    t = 0.0
    dt = run_cfg.dt_init
    E = problem["E0"].copy()

    time_history = []
    dt_history = []
    linear_iters_history = []
    nonlinear_iters_history = []
    eta_history = []
    residual_history = []
    failed_step = None
    failure_reason = None
    converged_all = True

    max_step_retries = getattr(run_cfg, "max_step_retries", 0)

    while t < run_cfg.t_end:
        if t + dt > run_cfg.t_end:
            dt = run_cfg.t_end - t

        step_success = False
        dt_try = dt
        last_stats = None
        last_E_new = None

        for retry in range(max_step_retries + 1):
            E_new, stats = solve_one_step(E, dt_try, problem, run_cfg, solver_cfg)
            last_stats = stats
            last_E_new = E_new

            if stats["converged"]:
                step_success = True
                break

            if retry < max_step_retries:
                dt_try *= 0.5
            else:
                break

            if dt_try < 1e-16:
                break

        eta = compute_eta(last_E_new, E, e_floor=run_cfg.e_floor)

        time_history.append(t + dt_try)
        dt_history.append(dt_try)
        linear_iters_history.append(last_stats["linear_iters_total"])
        nonlinear_iters_history.append(last_stats["nonlinear_iters"])
        eta_history.append(eta)
        residual_history.append(last_stats["final_residual_norm"])

        if not step_success:
            converged_all = False
            failed_step = len(time_history)
            failure_reason = (
                f"{solver_cfg.method} failed after retries at "
                f"t={t + dt_try:.6e}, dt={dt_try:.6e}, "
                f"residual={last_stats['final_residual_norm']:.6e}"
            )
            break

        E = last_E_new
        t += dt_try
        dt = update_dt(dt_try, eta, run_cfg.eta_target, run_cfg.dt_growth_limit)

    return {
        "E_final": E,
        "time_history": time_history,
        "dt_history": dt_history,
        "linear_iters_history": linear_iters_history,
        "nonlinear_iters_history": nonlinear_iters_history,
        "eta_history": eta_history,
        "residual_history": residual_history,
        "converged": converged_all,
        "failed_step": failed_step,
        "failure_reason": failure_reason,
        "t_final": t,
    }