from __future__ import annotations

import numpy as np

from config import SolverConfig, default_run_config
from driver import run_simulation
from experiments import average_iterations, fit_scaling_exponent
from problem import build_problem


def test_iteration_summary_helpers():
    assert average_iterations([1, 2, 3]) == 2.0
    assert np.isnan(average_iterations([]))
    exponent = fit_scaling_exponent([8, 16, 32], [2.0, 4.0, 8.0])
    assert 0.45 < exponent < 0.55


def test_short_driver_run_records_histories():
    run_cfg = default_run_config("M1", 4, 4, 0.10)
    run_cfg.t_end = 2e-6
    run_cfg.dt_init = 1e-6
    run_cfg.progress_interval = 0
    run_cfg.checkpoint_path = None
    solver_cfg = SolverConfig(
        method="nk2",
        nonlinear_tol=1e-7,
        linear_tol_factor=1e-2,
        max_nonlinear_iters=20,
        max_linear_iters=40,
        gmres_restart=40,
        damping_norm="linf",
    )
    problem = build_problem(run_cfg)
    result = run_simulation(problem, run_cfg, solver_cfg)

    assert result["converged"]
    assert len(result["time_history"]) >= 1
    assert len(result["linear_iters_history"]) == len(result["time_history"])
    assert len(result["nonlinear_iters_history"]) == len(result["time_history"])
    assert len(result["linear_iters_per_nonlinear_history"]) >= len(result["time_history"])
    assert result["t_final"] == run_cfg.t_end
