"""Focused M3 scaling diagnostics for the NK2 reproduction path.

This script intentionally avoids changing the production algorithm. It runs a
small set of paper-plausible solver variants for M3 and writes a CSV summary
with iteration counts and paper-style work scaling exponents.

Usage:
    python diagnose_m3_scaling.py --quick
    python diagnose_m3_scaling.py --full
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass

import numpy as np

from config import SolverConfig, default_run_config
from driver import run_simulation
from problem import build_problem


PAPER_M3 = {
    (0.10, 32): (2.88, 3.32),
    (0.10, 64): (3.97, 3.49),
    (0.10, 128): (5.70, 3.74),
    (0.10, 256): (7.74, 3.86),
    (0.50, 32): (4.71, 3.66),
    (0.50, 64): (6.61, 3.97),
    (0.50, 128): (9.37, 4.27),
    (0.50, 256): (12.8, 4.52),
}

PAPER_EXPONENT = {
    0.10: (1.240, 1.038),
    0.50: (1.242, 1.051),
}


@dataclass(frozen=True)
class Variant:
    name: str
    note: str
    nonlinear_tol: float = 1.2e-6
    linear_tol_factor: float = 3e-3
    jfnk_eps_mode: str = "normalized"
    damping_norm: str = "linf"
    mg_smoother: str = "jacobi"
    mg_pre_smooths: int = 3
    mg_post_smooths: int = 3
    picard_mass_mode: str = "q_derivative"
    use_m3_limiter: bool = True
    m3_limiter_gradient_mode: str = "cell_norm"


VARIANTS = [
    Variant("base", "current paper-like settings"),
    Variant("normal_limiter", "diagnostic: old directional face-gradient limiter", m3_limiter_gradient_mode="normal"),
    Variant("paper_linear_tol", "paper-stated linear tolerance factor", linear_tol_factor=1e-2),
    Variant("jacobi_5_5", "more Jacobi smoothing", linear_tol_factor=1e-2, mg_pre_smooths=5, mg_post_smooths=5),
    Variant("gs_3_3", "lexicographic GS smoother", linear_tol_factor=1e-2, mg_smoother="gs"),
    Variant("sgs_3_3", "symmetric GS smoother", linear_tol_factor=1e-2, mg_smoother="sgs"),
    Variant("paper_eps", "paper JFNK epsilon with paper linear tolerance", linear_tol_factor=1e-2, jfnk_eps_mode="paper"),
    Variant("paper_mass", "frozen Picard storage coefficient", picard_mass_mode="frozen_coeff"),
    Variant("no_limiter", "diagnostic only: disables M3 limiter", use_m3_limiter=False),
    Variant("cell_norm_limiter", "diagnostic: Wilson limiter uses cell-centered |grad E| / E", m3_limiter_gradient_mode="cell_norm"),
]


def average(values: list[int] | list[float]) -> float:
    return float("nan") if len(values) == 0 else float(np.mean(values))


def paper_work_exponent(grids: list[int], values: list[float]) -> float:
    """Fit work ~ N^s using avg_iters ~ N^(s-1)."""
    valid = [(g, v) for g, v in zip(grids, values) if math.isfinite(v) and v > 0]
    if len(valid) < 2:
        return float("nan")
    n_cells = np.array([g * g for g, _ in valid], dtype=float)
    y = np.array([v for _, v in valid], dtype=float)
    slope = np.polyfit(np.log(n_cells), np.log(y), 1)[0]
    return float(1.0 + slope)


def make_solver_config(variant: Variant) -> SolverConfig:
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


def run_case(variant: Variant, eta: float, grid: int) -> dict:
    run_cfg = default_run_config("M3", grid, grid, eta)
    run_cfg.progress_interval = 0
    run_cfg.checkpoint_path = None
    run_cfg.checkpoint_interval = 0

    problem = build_problem(run_cfg)
    problem["use_m3_limiter"] = variant.use_m3_limiter
    problem["m3_limiter_gradient_mode"] = variant.m3_limiter_gradient_mode
    solver_cfg = make_solver_config(variant)

    start = time.perf_counter()
    result = run_simulation(problem, run_cfg, solver_cfg)
    elapsed = time.perf_counter() - start

    paper_linear, paper_nonlinear = PAPER_M3[(eta, grid)]
    avg_linear = average(result["linear_iters_history"]) if result["converged"] else float("nan")
    avg_nonlinear = average(result["nonlinear_iters_history"]) if result["converged"] else float("nan")
    avg_inner = average(result.get("linear_iters_per_nonlinear_history", []))

    return {
        "variant": variant.name,
        "note": variant.note,
        "eta": eta,
        "grid": grid,
        "converged": result["converged"],
        "t_final": result["t_final"],
        "steps": len(result["time_history"]),
        "avg_linear": avg_linear,
        "paper_linear": paper_linear,
        "linear_delta": avg_linear - paper_linear,
        "avg_nonlinear": avg_nonlinear,
        "paper_nonlinear": paper_nonlinear,
        "nonlinear_delta": avg_nonlinear - paper_nonlinear,
        "avg_linear_per_nonlinear": avg_inner,
        "max_linear": max(result["linear_iters_history"]) if result["linear_iters_history"] else float("nan"),
        "max_nonlinear": max(result["nonlinear_iters_history"]) if result["nonlinear_iters_history"] else float("nan"),
        "linear_tol_factor": variant.linear_tol_factor,
        "jfnk_eps_mode": variant.jfnk_eps_mode,
        "damping_norm": variant.damping_norm,
        "mg_smoother": variant.mg_smoother,
        "mg_pre_smooths": variant.mg_pre_smooths,
        "mg_post_smooths": variant.mg_post_smooths,
        "picard_mass_mode": variant.picard_mass_mode,
        "use_m3_limiter": variant.use_m3_limiter,
        "m3_limiter_gradient_mode": variant.m3_limiter_gradient_mode,
        "elapsed_sec": elapsed,
        "failure_reason": result["failure_reason"] or "",
    }


def read_existing(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict) -> tuple[str, str, str]:
    return (row["variant"], f"{float(row['eta']):.2f}", str(row["grid"]))


def job_key(variant: Variant, eta: float, grid: int) -> tuple[str, str, str]:
    return (variant.name, f"{eta:.2f}", str(grid))


def print_scaling_summary(rows: list[dict]) -> None:
    print("\nScaling summary, paper-style work exponent")
    print("variant, eta, linear_s, paper_linear_s, nonlinear_s, paper_nonlinear_s")
    for variant in sorted({row["variant"] for row in rows}):
        for eta in (0.10, 0.50):
            subset = [
                row
                for row in rows
                if row["variant"] == variant
                and abs(float(row["eta"]) - eta) < 1e-12
                and str(row["converged"]).lower() == "true"
            ]
            subset = sorted(subset, key=lambda r: int(r["grid"]))
            grids = [int(row["grid"]) for row in subset]
            linear_vals = [float(row["avg_linear"]) for row in subset]
            nonlinear_vals = [float(row["avg_nonlinear"]) for row in subset]
            linear_s = paper_work_exponent(grids, linear_vals)
            nonlinear_s = paper_work_exponent(grids, nonlinear_vals)
            paper_linear_s, paper_nonlinear_s = PAPER_EXPONENT[eta]
            print(
                f"{variant}, {eta:.2f}, {linear_s:.4f}, {paper_linear_s:.4f}, "
                f"{nonlinear_s:.4f}, {paper_nonlinear_s:.4f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/m3_scaling_diagnostic.csv")
    parser.add_argument("--quick", action="store_true", help="Run 32 and 64 grids only.")
    parser.add_argument("--full", action="store_true", help="Run 32, 64, and 128 grids.")
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()

    grids = [32, 64] if args.quick or not args.full else [32, 64, 128]
    variant_names = set(args.variants) if args.variants else {v.name for v in VARIANTS}
    variants = [v for v in VARIANTS if v.name in variant_names]

    rows = read_existing(args.out)
    completed = {row_key(row) for row in rows}
    jobs = [(variant, eta, grid) for variant in variants for eta in (0.10, 0.50) for grid in grids]

    for idx, (variant, eta, grid) in enumerate(jobs, start=1):
        if job_key(variant, eta, grid) in completed:
            print(f"[{idx}/{len(jobs)}] skip {variant.name} eta={eta:.2f} grid={grid}", flush=True)
            continue
        print(f"[{idx}/{len(jobs)}] run {variant.name} eta={eta:.2f} grid={grid}", flush=True)
        rows.append(run_case(variant, eta, grid))
        write_rows(rows, args.out)

    print_scaling_summary(rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
