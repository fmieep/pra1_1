"""Run focused NK2 diagnostics against the paper iteration tables.

This script is intentionally separate from test.py so the manual experiment
entry point stays easy to edit.  The default "quick" run is small enough to use
for diagnosis: it covers 32x32 and 64x64 grids and a few solver variants.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass

import numpy as np

from config import SolverConfig, default_run_config
from driver import run_simulation
from problem import build_problem


PAPER_NK2 = {
    (0.10, 32, "M1"): (2.57, 3.42),
    (0.10, 64, "M1"): (3.02, 3.57),
    (0.10, 128, "M1"): (3.40, 3.71),
    (0.10, 256, "M1"): (3.39, 3.61),
    (0.10, 32, "M2"): (7.72, 4.53),
    (0.10, 64, "M2"): (9.01, 4.62),
    (0.10, 128, "M2"): (10.4, 4.66),
    (0.10, 256, "M2"): (11.0, 4.62),
    (0.10, 32, "M3"): (2.88, 3.32),
    (0.10, 64, "M3"): (3.97, 3.49),
    (0.10, 128, "M3"): (5.70, 3.74),
    (0.10, 256, "M3"): (7.74, 3.86),
    (0.50, 32, "M1"): (3.35, 3.42),
    (0.50, 64, "M1"): (3.96, 3.63),
    (0.50, 128, "M1"): (4.59, 3.79),
    (0.50, 256, "M1"): (5.34, 3.99),
    (0.50, 32, "M2"): (14.3, 5.29),
    (0.50, 64, "M2"): (15.2, 5.37),
    (0.50, 128, "M2"): (16.2, 5.48),
    (0.50, 256, "M2"): (17.5, 5.66),
    (0.50, 32, "M3"): (4.71, 3.66),
    (0.50, 64, "M3"): (6.61, 3.97),
    (0.50, 128, "M3"): (9.37, 4.27),
    (0.50, 256, "M3"): (12.8, 4.52),
}


@dataclass(frozen=True)
class Variant:
    name: str
    nonlinear_tol: float = 1.2e-6
    linear_tol_factor: float = 3e-3
    jfnk_eps_mode: str = "normalized"
    damping_norm: str = "linf"
    mg_smoother: str = "jacobi"
    mg_pre_smooths: int = 3
    mg_post_smooths: int = 3
    picard_mass_mode: str = "q_derivative"
    use_m3_limiter: bool = True


VARIANTS = {
    "base": Variant("base"),
    "tol_1e-2": Variant("tol_1e-2", linear_tol_factor=1e-2),
    "jacobi_5_5": Variant("jacobi_5_5", linear_tol_factor=1e-2, mg_pre_smooths=5, mg_post_smooths=5),
    "sgs_3_3": Variant("sgs_3_3", linear_tol_factor=1e-2, mg_smoother="sgs"),
    "paper_eps": Variant("paper_eps", linear_tol_factor=1e-2, jfnk_eps_mode="paper"),
    "paper_mass": Variant("paper_mass", picard_mass_mode="frozen_coeff"),
    "paperish": Variant("paperish", linear_tol_factor=1e-2, picard_mass_mode="frozen_coeff"),
    "m3_no_limiter": Variant("m3_no_limiter", use_m3_limiter=False),
    "paper_nl_tol": Variant("paper_nl_tol", nonlinear_tol=1e-6),
}


def avg(values: list[float] | list[int]) -> float:
    return float("nan") if len(values) == 0 else float(np.mean(values))


def make_solver_cfg(variant: Variant) -> SolverConfig:
    return SolverConfig(
        method="nk2",
        nonlinear_tol=variant.nonlinear_tol,
        linear_tol_factor=variant.linear_tol_factor,
        max_nonlinear_iters=60,
        max_linear_iters=100,
        gmres_restart=100,
        rho_jfnk=1e-8,
        use_multigrid_preconditioner=True,
        jfnk_eps_mode=variant.jfnk_eps_mode,
        damping_norm=variant.damping_norm,
        mg_smoother=variant.mg_smoother,
        mg_pre_smooths=variant.mg_pre_smooths,
        mg_post_smooths=variant.mg_post_smooths,
        picard_mass_mode=variant.picard_mass_mode,
    )


def run_case(variant: Variant, model: str, eta: float, grid_size: int) -> dict:
    run_cfg = default_run_config(model=model, nx=grid_size, ny=grid_size, eta_target=eta)
    run_cfg.progress_interval = 0
    run_cfg.checkpoint_path = None
    run_cfg.checkpoint_interval = 0
    problem = build_problem(run_cfg)
    problem["use_m3_limiter"] = variant.use_m3_limiter
    solver_cfg = make_solver_cfg(variant)

    start = time.perf_counter()
    result = run_simulation(problem, run_cfg, solver_cfg)
    elapsed = time.perf_counter() - start

    paper_linear, paper_nonlinear = PAPER_NK2.get((eta, grid_size, model), (float("nan"), float("nan")))
    avg_linear = avg(result["linear_iters_history"]) if result["converged"] else float("nan")
    avg_nonlinear = avg(result["nonlinear_iters_history"]) if result["converged"] else float("nan")
    avg_inner = avg(result.get("linear_iters_per_nonlinear_history", []))

    return {
        "variant": variant.name,
        "model": model,
        "eta": eta,
        "grid": grid_size,
        "converged": result["converged"],
        "steps": len(result["time_history"]),
        "avg_linear": avg_linear,
        "paper_linear": paper_linear,
        "linear_delta": avg_linear - paper_linear,
        "avg_nonlinear": avg_nonlinear,
        "paper_nonlinear": paper_nonlinear,
        "nonlinear_delta": avg_nonlinear - paper_nonlinear,
        "avg_linear_per_newton_solve": avg_inner,
        "picard_mass_mode": variant.picard_mass_mode,
        "use_m3_limiter": variant.use_m3_limiter,
        "nonlinear_tol": variant.nonlinear_tol,
        "max_linear": max(result["linear_iters_history"]) if result["linear_iters_history"] else float("nan"),
        "max_nonlinear": max(result["nonlinear_iters_history"]) if result["nonlinear_iters_history"] else float("nan"),
        "elapsed_sec": elapsed,
        "failure_reason": result["failure_reason"] or "",
    }


def experiment_plan(mode: str) -> list[tuple[Variant, str, float, int]]:
    if mode == "smoke":
        return [(VARIANTS["base"], "M1", 0.10, 32)]

    if mode == "quick":
        jobs = []
        for variant_name in ("base", "tol_1e-2"):
            for model in ("M1", "M2", "M3"):
                for eta in (0.10, 0.50):
                    for grid_size in (32, 64):
                        jobs.append((VARIANTS[variant_name], model, eta, grid_size))
        for variant_name in ("jacobi_5_5", "sgs_3_3"):
            for eta in (0.10, 0.50):
                for grid_size in (32, 64):
                    jobs.append((VARIANTS[variant_name], "M3", eta, grid_size))
        return jobs

    if mode == "eps":
        jobs = []
        for variant_name in ("tol_1e-2", "paper_eps"):
            for model in ("M1", "M2"):
                for eta in (0.10, 0.50):
                    jobs.append((VARIANTS[variant_name], model, eta, 32))
        return jobs

    if mode == "base_tol_32":
        jobs = []
        for variant_name in ("base", "tol_1e-2"):
            for model in ("M1", "M2", "M3"):
                for eta in (0.10, 0.50):
                    jobs.append((VARIANTS[variant_name], model, eta, 32))
        return jobs

    if mode == "base_tol_64":
        jobs = []
        for variant_name in ("base", "tol_1e-2"):
            for model in ("M1", "M2", "M3"):
                for eta in (0.10, 0.50):
                    jobs.append((VARIANTS[variant_name], model, eta, 64))
        return jobs

    if mode == "m3_smoother_32":
        jobs = []
        for variant_name in ("tol_1e-2", "jacobi_5_5", "sgs_3_3"):
            for eta in (0.10, 0.50):
                jobs.append((VARIANTS[variant_name], "M3", eta, 32))
        return jobs

    if mode == "m3_smoother_64":
        jobs = []
        for variant_name in ("tol_1e-2", "jacobi_5_5", "sgs_3_3"):
            for eta in (0.10, 0.50):
                jobs.append((VARIANTS[variant_name], "M3", eta, 64))
        return jobs

    if mode == "mass_32":
        jobs = []
        for variant_name in ("base", "paper_mass", "paperish"):
            for model in ("M1", "M2", "M3"):
                for eta in (0.10, 0.50):
                    jobs.append((VARIANTS[variant_name], model, eta, 32))
        return jobs

    if mode == "mass_64":
        jobs = []
        for variant_name in ("base", "paper_mass", "paperish"):
            for model in ("M1", "M2", "M3"):
                for eta in (0.10, 0.50):
                    jobs.append((VARIANTS[variant_name], model, eta, 64))
        return jobs

    if mode == "m3_limiter_32":
        jobs = []
        for variant_name in ("base", "m3_no_limiter"):
            for eta in (0.10, 0.50):
                jobs.append((VARIANTS[variant_name], "M3", eta, 32))
        return jobs

    if mode == "m3_limiter_64":
        jobs = []
        for variant_name in ("base", "m3_no_limiter"):
            for eta in (0.10, 0.50):
                jobs.append((VARIANTS[variant_name], "M3", eta, 64))
        return jobs

    if mode == "nl_tol_32":
        jobs = []
        for variant_name in ("base", "paper_nl_tol"):
            for model in ("M1", "M2", "M3"):
                for eta in (0.10, 0.50):
                    jobs.append((VARIANTS[variant_name], model, eta, 32))
        return jobs

    if mode == "nl_tol_64":
        jobs = []
        for variant_name in ("base", "paper_nl_tol"):
            for model in ("M1", "M2", "M3"):
                for eta in (0.10, 0.50):
                    jobs.append((VARIANTS[variant_name], model, eta, 64))
        return jobs

    if mode == "m1_smoother_32":
        jobs = []
        for variant_name in ("base", "jacobi_5_5", "sgs_3_3"):
            for eta in (0.10, 0.50):
                jobs.append((VARIANTS[variant_name], "M1", eta, 32))
        return jobs

    if mode == "m1_smoother_64":
        jobs = []
        for variant_name in ("base", "jacobi_5_5", "sgs_3_3"):
            for eta in (0.10, 0.50):
                jobs.append((VARIANTS[variant_name], "M1", eta, 64))
        return jobs

    raise ValueError(f"Unknown mode: {mode}")


def write_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_existing(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def job_key(variant: Variant, model: str, eta: float, grid_size: int) -> tuple[str, str, str, str]:
    return (variant.name, model, f"{eta:.2f}", str(grid_size))


def row_key(row: dict) -> tuple[str, str, str, str]:
    return (row["variant"], row["model"], f"{float(row['eta']):.2f}", str(row["grid"]))


def print_summary(rows: list[dict]) -> None:
    print(
        "variant, model, eta, grid, ok, avg_lin, paper_lin, d_lin, "
        "avg_nonlin, paper_nonlin, d_nonlin, lin/newton, steps, sec"
    )
    for row in rows:
        eta = float(row["eta"])
        grid = int(row["grid"])
        converged = row["converged"]
        if isinstance(converged, str):
            converged = converged.lower() == "true"
        print(
            f"{row['variant']}, {row['model']}, {eta:.2f}, {grid}, "
            f"{converged}, {float(row['avg_linear']):.3f}, {float(row['paper_linear']):.3f}, "
            f"{float(row['linear_delta']):.3f}, {float(row['avg_nonlinear']):.3f}, "
            f"{float(row['paper_nonlinear']):.3f}, {float(row['nonlinear_delta']):.3f}, "
            f"{float(row['avg_linear_per_newton_solve']):.3f}, {int(row['steps'])}, "
            f"{float(row['elapsed_sec']):.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "smoke",
            "quick",
            "eps",
            "base_tol_32",
            "base_tol_64",
            "m3_smoother_32",
            "m3_smoother_64",
            "mass_32",
            "mass_64",
            "m3_limiter_32",
            "m3_limiter_64",
            "nl_tol_32",
            "nl_tol_64",
            "m1_smoother_32",
            "m1_smoother_64",
        ),
        default="quick",
    )
    parser.add_argument("--out", default="output/nk2_diagnostic_compare.csv")
    args = parser.parse_args()

    rows = read_existing(args.out)
    completed = {row_key(row) for row in rows}
    jobs = experiment_plan(args.mode)
    for idx, (variant, model, eta, grid_size) in enumerate(jobs, start=1):
        if job_key(variant, model, eta, grid_size) in completed:
            print(
                f"[{idx}/{len(jobs)}] skip existing variant={variant.name} "
                f"model={model} eta={eta:.2f} grid={grid_size}",
                flush=True,
            )
            continue
        print(
            f"[{idx}/{len(jobs)}] variant={variant.name} model={model} "
            f"eta={eta:.2f} grid={grid_size}",
            flush=True,
        )
        rows.append(run_case(variant, model, eta, grid_size))
        write_csv(rows, args.out)

    print_summary(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
