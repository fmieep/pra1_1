"""Batch experiments for table-style summaries."""
from __future__ import annotations
import numpy as np
from config import SolverConfig, default_run_config
from problem import build_problem
from driver import run_simulation


def average_iterations(history: list[int]) -> float:
    """Return the mean iteration count over all completed time steps."""
    if len(history) == 0:
        return float("nan")
    return float(np.mean(history))


def fit_scaling_exponent(grid_sizes: list[int], values: list[float]) -> float:
    """Fit y ~ C * N^s with N = nx * ny using valid positive entries only."""
    N = np.array([g * g for g in grid_sizes], dtype=float)
    y = np.array(values, dtype=float)
    mask = np.isfinite(y) & (y > 0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    coeffs = np.polyfit(np.log(N[mask]), np.log(y[mask]), 1)
    return float(coeffs[0])


def run_case(method: str, model: str, eta: float, nx: int, ny: int) -> dict:
    """Run one (method, model, eta, grid) combination and summarize it."""
    run_cfg = default_run_config(model=model, nx=nx, ny=ny, eta_target=eta)
    problem = build_problem(run_cfg)
    solver_cfg = SolverConfig(method=method)
    result = run_simulation(problem, run_cfg, solver_cfg)
    if result["converged"]:
        avg_linear = average_iterations(result["linear_iters_history"])
        avg_nonlinear = average_iterations(result["nonlinear_iters_history"])
    else:
        # Failed cases should remain explicit failures rather than being turned
        # into partial-step averages. This matches the paper's table logic.
        avg_linear = float("nan")
        avg_nonlinear = float("nan")
    return {
        "method": method,
        "model": model,
        "eta": eta,
        "grid": (nx, ny),
        "avg_linear_iters": avg_linear,
        "avg_nonlinear_iters": avg_nonlinear,
        "converged": result["converged"],
        "num_steps": len(result["time_history"]),
        "t_final": result["t_final"],
        "failed_step": result["failed_step"],
        "failure_reason": result["failure_reason"],
        "raw": result,
    }


def run_table_xiii_to_xvi() -> dict:
    """Batch the table-style experiments for Picard and NK2.

    The traversal covers:
    - method in {picard, nk2}
    - model in {M1, M2, M3}
    - eta in {0.10, 0.50}
    - grid in {32, 64, 128, 256}
    """
    grids = [32, 64, 128, 256]
    methods = ["picard", "nk2"]
    models = ["M1", "M2", "M3"]
    etas = [0.10, 0.50]

    results = {}
    for method in methods:
        for eta in etas:
            key = (method, eta)
            results[key] = {}
            for model in models:
                rows = []
                linear_vals = []
                nonlinear_vals = []
                failures = []
                for g in grids:
                    case = run_case(method, model, eta, g, g)
                    rows.append(case)
                    linear_vals.append(case["avg_linear_iters"])
                    nonlinear_vals.append(case["avg_nonlinear_iters"])
                    if not case["converged"]:
                        failures.append({
                            "grid": case["grid"],
                            "failed_step": case["failed_step"],
                            "failure_reason": case["failure_reason"],
                        })
                results[key][model] = {
                    "rows": rows,
                    "linear_scaling_exponent": fit_scaling_exponent(grids, linear_vals),
                    "nonlinear_scaling_exponent": fit_scaling_exponent(grids, nonlinear_vals),
                    "failures": failures,
                }
    return results


def print_table_style(results: dict) -> None:
    """Print a compact terminal summary in a paper-inspired table style."""
    for (method, eta), by_model in results.items():
        print(f"\n=== method={method}, eta={eta:.2f} ===")

        print("  Average linear iterations")
        for model, payload in by_model.items():
            print(f"    {model}")
            for row in payload["rows"]:
                nx, ny = row["grid"]
                avg_linear = row["avg_linear_iters"]
                avg_linear_str = "FAIL" if not np.isfinite(avg_linear) else f"{avg_linear:.4g}"
                status = "ok" if row["converged"] else "failed"
                print(f"      grid={nx}x{ny:<3} avg_linear={avg_linear_str:<8} status={status}")
            print(f"      scaling exponent = {payload['linear_scaling_exponent']:.4g}")

        print("  Average nonlinear iterations")
        for model, payload in by_model.items():
            print(f"    {model}")
            for row in payload["rows"]:
                nx, ny = row["grid"]
                avg_nonlinear = row["avg_nonlinear_iters"]
                avg_nonlinear_str = "FAIL" if not np.isfinite(avg_nonlinear) else f"{avg_nonlinear:.4g}"
                status = "ok" if row["converged"] else "failed"
                print(f"      grid={nx}x{ny:<3} avg_nonlinear={avg_nonlinear_str:<8} status={status}")
            print(f"      scaling exponent = {payload['nonlinear_scaling_exponent']:.4g}")

        failure_rows = []
        for model, payload in by_model.items():
            for failure in payload["failures"]:
                failure_rows.append((model, failure))
        if failure_rows:
            print("  Failures")
            for model, failure in failure_rows:
                gx, gy = failure["grid"]
                reason = failure["failure_reason"] or "unknown failure"
                print(f"    {model} grid={gx}x{gy}: {reason}")


if __name__ == "__main__":
    results = run_table_xiii_to_xvi()
    print_table_style(results)
