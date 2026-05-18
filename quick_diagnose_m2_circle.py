"""Quickly diagnose M2 Fig. 4 circle-region contour crowding.

This script reads an existing checkpoint only.  It does not rerun the PDE and
does not modify solver code.

Usage:
    python quick_diagnose_m2_circle.py
    python quick_diagnose_m2_circle.py --grid 64
    python quick_diagnose_m2_circle.py --npz checkpoints/M2_eta0.5_grid128_methodnk2.npz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import default_run_config
from physics import energy_to_temperature
from problem import build_problem


CENTER = (0.75, 0.75)
RADIUS = 0.15
ETA = 0.50
DEFAULT_GRID = 128
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
LEVELS = np.asarray([1.86, 2.72, 3.57, 4.43, 5.29, 6.15, 7.01, 7.87, 8.72])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Checkpoint file to read. Overrides --checkpoint-dir naming.",
    )
    parser.add_argument("--grid", type=int, default=DEFAULT_GRID)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    return parser.parse_args()


def checkpoint_path(args: argparse.Namespace) -> Path:
    if args.npz is not None:
        return args.npz
    return args.checkpoint_dir / f"M2_eta0.5_grid{args.grid}_methodnk2.npz"


def region_stats(name: str, values: np.ndarray) -> tuple[str, float, float, float, int]:
    if values.size == 0:
        return name, float("nan"), float("nan"), float("nan"), 0
    return (
        name,
        float(np.min(values)),
        float(np.mean(values)),
        float(np.max(values)),
        int(values.size),
    )


def print_region_stats(T: np.ndarray, r: np.ndarray) -> dict[str, tuple[str, float, float, float, int]]:
    masks = {
        "inside_circle": r <= RADIUS,
        "near_inside": (r > 0.12) & (r <= RADIUS),
        "near_outside": (r > RADIUS) & (r <= 0.18),
        "outer_annulus": (r > 0.18) & (r <= 0.23),
    }
    stats = {name: region_stats(name, T[mask]) for name, mask in masks.items()}

    print("Region temperature statistics")
    print("region            count       T_min      T_mean       T_max")
    for name in ("inside_circle", "near_inside", "near_outside", "outer_annulus"):
        _, t_min, t_mean, t_max, count = stats[name]
        print(f"{name:<16} {count:6d} {t_min:11.6g} {t_mean:11.6g} {t_max:11.6g}")
    return stats


def diagnose_temperature_contrast(
    stats: dict[str, tuple[str, float, float, float, int]]
) -> None:
    inside_mean = stats["inside_circle"][2]
    outside_mean = stats["near_outside"][2]
    contrast = outside_mean - inside_mean

    print()
    print(f"near_outside_mean - inside_circle_mean = {contrast:.6g}")
    if math.isfinite(contrast) and contrast > 1.0:
        print("圆内部明显更冷，说明高 Z=50 材料确实强烈阻滞扩散，等温线密集主要是物理结果。")
    else:
        print("圆内部并没有明显更冷，等温线密集可能主要来自绘图、线宽、标签或网格边界。")


def contour_min_distances(X: np.ndarray, Y: np.ndarray, T: np.ndarray) -> dict[float, float]:
    fig, ax = plt.subplots()
    contour = ax.contour(X, Y, T, levels=LEVELS)
    min_distances: dict[float, float] = {}

    for level, segments in zip(contour.levels, contour.allsegs):
        best = float("nan")
        for segment in segments:
            if segment.size == 0:
                continue
            x = segment[:, 0]
            y = segment[:, 1]
            radial_distance = np.sqrt((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2)
            segment_best = float(np.min(np.abs(radial_distance - RADIUS)))
            if not math.isfinite(best) or segment_best < best:
                best = segment_best
        min_distances[float(level)] = best

    plt.close(fig)
    return min_distances


def print_contour_diagnostics(min_distances: dict[float, float]) -> None:
    print()
    print("Contour distance to circle boundary r=0.15")
    print("level      min_abs_dist")
    close_count = 0
    finite_count = 0
    for level in LEVELS:
        distance = min_distances.get(float(level), float("nan"))
        if math.isfinite(distance):
            finite_count += 1
            if distance < 0.01:
                close_count += 1
            distance_text = f"{distance:.6g}"
        else:
            distance_text = "nan"
        print(f"{level:5.2f}      {distance_text}")

    print()
    print(f"levels with min_abs_dist < 0.01: {close_count} / {finite_count}")
    if close_count >= 4:
        print("多条等温线确实贴近圆边界，黑圈不是单纯绘图造成。")
    else:
        print("只有少数等温线贴近圆边界，黑圈更多可能是显示效果。")


def main() -> int:
    args = parse_args()
    path = checkpoint_path(args)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        E = data["E"].copy()
        t_final = float(data["t"]) if "t" in data.files else float("nan")
        steps = len(data["time_history"]) if "time_history" in data.files else 0

    nx = ny = int(args.grid)
    if E.shape != (nx, ny):
        raise ValueError(
            f"Checkpoint E shape {E.shape} does not match --grid {args.grid}. "
            "Pass the matching --grid value."
        )

    run_cfg = default_run_config("M2", nx, ny, ETA)
    problem = build_problem(run_cfg)
    X = problem["grid"]["Xc"]
    Y = problem["grid"]["Yc"]
    Z = problem["Z"]
    T = energy_to_temperature(E)
    r = np.sqrt((X - CENTER[0]) ** 2 + (Y - CENTER[1]) ** 2)

    print(f"checkpoint: {path}")
    print(f"grid: {nx} x {ny}")
    print(f"t_final: {t_final:.6g}, steps: {steps}")
    print(f"T range: [{float(np.min(T)):.6g}, {float(np.max(T)):.6g}]")
    print(f"Z inside circle unique values: {np.unique(Z[r <= RADIUS])}")
    print()

    stats = print_region_stats(T, r)
    diagnose_temperature_contrast(stats)
    min_distances = contour_min_distances(X, Y, T)
    print_contour_diagnostics(min_distances)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
