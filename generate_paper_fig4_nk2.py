"""Generate a paper Fig. 4 style contour plot from the NK2 solution.

The target comparison is the paper's two-panel temperature figure:
    (a) M1, eta = 0.50, t = 5.0
    (b) M2, eta = 0.50, t = 0.005

By default the script first looks for existing NK2 checkpoints in ./checkpoints.
If a matching checkpoint is missing, it runs the current NK2 solver in memory
without saving a new checkpoint.

Usage:
    python generate_paper_fig4_nk2.py
    python generate_paper_fig4_nk2.py --grid 64
    python generate_paper_fig4_nk2.py --m1-grid 256 --m2-grid 128
    python generate_paper_fig4_nk2.py --rerun
    python generate_paper_fig4_nk2.py --no-run
"""

from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import SolverConfig, default_run_config
from driver import run_simulation
from physics import energy_to_temperature
from problem import build_problem


DEFAULT_M1_GRID = 256
DEFAULT_M2_GRID = 128
DEFAULT_ETA = 0.50
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_OUTPUT = Path("output") / "paper_fig4_nk2_contours.png"
DEFAULT_NUM_LEVELS = 11
DEFAULT_INCLUDE_LEVEL_ONE = False

# These are the contour ranges visible in the paper's Fig. 4 labels.  They are
# slightly below 10 because the plotted cell-centered maximum is below the
# asymptotic left-boundary temperature.  Keeping these fixed makes positional
# differences easier to compare against the published figure.
PAPER_FIG4_LEVEL_LIMITS = {
    "M1": (1.0, 9.92),
    "M2": (1.0, 9.58),
}
PAPER_FIG4_EXACT_LEVELS = {
    "M1": np.asarray([1.00, 1.89, 2.78, 3.68, 4.57, 5.46, 6.35, 7.25, 8.14, 9.03]),
    "M2": np.asarray([1.00, 1.86, 2.72, 3.57, 4.43, 5.29, 6.15, 7.01, 7.87, 8.72, 9.58]),
}
PAPER_FIG4_VISIBLE_LEVELS = {
    # M1's published panel shows most of these labels clearly.
    "M1": PAPER_FIG4_EXACT_LEVELS["M1"],
    # In M2, low-temperature contours collapse tightly around the cold material
    # inclusions.  The published right panel mainly shows these visible levels.
    "M2": np.asarray([5.29, 6.15, 7.01, 7.87]),
}


@dataclass(frozen=True)
class FigureCase:
    panel: str
    model: str
    eta: float
    t_end: float
    default_grid: int


FIGURE_CASES = (
    FigureCase("a", "M1", DEFAULT_ETA, 5.0, DEFAULT_M1_GRID),
    FigureCase("b", "M2", DEFAULT_ETA, 0.005, DEFAULT_M2_GRID),
)


def make_solver_config() -> SolverConfig:
    """Use the current stable NK2 settings from the project scripts."""
    return SolverConfig(
        method="nk2",
        nonlinear_tol=2e-6,
        linear_tol_factor=3e-3,
        max_nonlinear_iters=60,
        max_linear_iters=100,
        gmres_restart=100,
        rho_jfnk=1e-8,
        use_multigrid_preconditioner=True,
        mg_smoother="jacobi",
        jfnk_eps_mode="normalized",
        damping_norm="linf",
        mg_pre_smooths=3,
        mg_post_smooths=3,
        picard_mass_mode="q_derivative",
    )


def checkpoint_matches(path: Path, case: FigureCase, grid: int) -> bool:
    pattern = re.compile(
        r"^(M[123])_eta([0-9.]+)_grid(\d+)_method(nk2|picard)\.npz$",
        re.IGNORECASE,
    )
    match = pattern.match(path.name)
    if not match:
        return False

    model = match.group(1).upper()
    eta = round(float(match.group(2)), 2)
    grid_size = int(match.group(3))
    method = match.group(4).lower()
    return (
        model == case.model
        and eta == round(case.eta, 2)
        and grid_size == grid
        and method == "nk2"
    )


def find_checkpoint(checkpoint_dir: Path, case: FigureCase, grid: int) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    candidates = [
        path
        for path in sorted(checkpoint_dir.glob("*.npz"))
        if checkpoint_matches(path, case, grid)
    ]
    return candidates[-1] if candidates else None


def load_checkpoint(path: Path, expected_t_end: float) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=True) as data:
        E = data["E"].copy()
        t_final = float(data["t"])
        dt = float(data["dt"]) if "dt" in data.files else float("nan")
        time_history = (
            np.asarray(data["time_history"], dtype=float)
            if "time_history" in data.files
            else np.asarray([], dtype=float)
        )
        linear_iters = (
            np.asarray(data["linear_iters_history"], dtype=float)
            if "linear_iters_history" in data.files
            else np.asarray([], dtype=float)
        )
        nonlinear_iters = (
            np.asarray(data["nonlinear_iters_history"], dtype=float)
            if "nonlinear_iters_history" in data.files
            else np.asarray([], dtype=float)
        )

    complete = abs(t_final - expected_t_end) <= max(1e-12, 1e-9 * expected_t_end)
    meta = {
        "source": str(path),
        "t_final": t_final,
        "dt": dt,
        "steps": int(time_history.size),
        "avg_linear": safe_mean(linear_iters),
        "avg_nonlinear": safe_mean(nonlinear_iters),
        "complete": complete,
    }
    return E, meta


def safe_mean(values: np.ndarray) -> float:
    return float("nan") if values.size == 0 else float(np.mean(values))


def run_case(case: FigureCase, grid: int) -> tuple[np.ndarray, dict]:
    run_cfg = default_run_config(case.model, grid, grid, case.eta)
    run_cfg.t_end = case.t_end
    run_cfg.progress_interval = 1000
    run_cfg.checkpoint_path = None
    run_cfg.checkpoint_interval = 0
    run_cfg.resume_from_checkpoint = False
    run_cfg.max_step_retries = 8
    run_cfg.dt_growth_limit = 1.1
    run_cfg.e_floor = 1.0

    problem = build_problem(run_cfg)
    solver_cfg = make_solver_config()

    start = time.perf_counter()
    result = run_simulation(problem, run_cfg, solver_cfg)
    elapsed = time.perf_counter() - start

    meta = {
        "source": "fresh-run",
        "t_final": float(result["t_final"]),
        "steps": len(result["time_history"]),
        "avg_linear": safe_mean(np.asarray(result["linear_iters_history"], dtype=float)),
        "avg_nonlinear": safe_mean(np.asarray(result["nonlinear_iters_history"], dtype=float)),
        "complete": bool(result["converged"]),
        "elapsed_sec": elapsed,
        "failure_reason": result["failure_reason"] or "",
    }
    return result["E_final"], meta


def load_or_run_case(
    case: FigureCase,
    grid: int,
    checkpoint_dir: Path,
    rerun: bool,
    no_run: bool,
) -> tuple[np.ndarray, dict]:
    checkpoint = None if rerun else find_checkpoint(checkpoint_dir, case, grid)
    if checkpoint is not None:
        E, meta = load_checkpoint(checkpoint, case.t_end)
        if meta["complete"]:
            return E, meta
        print(
            f"[warning] checkpoint is incomplete for {case.model}: "
            f"t={meta['t_final']:.6g}, expected={case.t_end:.6g}"
        )
        if no_run:
            return E, meta

    if no_run:
        raise FileNotFoundError(
            f"No complete checkpoint for {case.model}, eta={case.eta:g}, grid={grid}."
        )

    print(f"[run] {case.model}, eta={case.eta:g}, grid={grid}x{grid}, t_end={case.t_end:g}")
    return run_case(case, grid)


def contour_levels_for_case(
    case: FigureCase,
    T: np.ndarray,
    level_mode: str,
    levels_override: np.ndarray | None,
    num_levels: int,
    include_level_one: bool,
) -> np.ndarray:
    """Return contour levels for one panel.

    level_mode="paper-exact" uses the printed contour labels visible in the
    published figure.  level_mode="paper-visible" keeps only the labels that
    are visually useful in the M2 panel.  level_mode="paper" keeps the same
    ranges but allows a denser requested number of contours.
    """
    if levels_override is not None:
        levels = levels_override
    elif level_mode == "paper-exact":
        levels = PAPER_FIG4_EXACT_LEVELS[case.model].copy()
    elif level_mode == "paper-visible":
        levels = PAPER_FIG4_VISIBLE_LEVELS[case.model].copy()
    elif level_mode == "paper":
        lo, hi = PAPER_FIG4_LEVEL_LIMITS.get(case.model, (1.0, 10.0))
        levels = np.linspace(lo, hi, num_levels)
    elif level_mode == "data":
        lo = max(1.0, float(np.nanmin(T)))
        hi = float(np.nanmax(T))
        if hi <= lo:
            hi = lo + 1.0
        levels = np.linspace(lo, hi, num_levels)
    elif level_mode == "uniform":
        levels = np.linspace(1.0, 10.0, num_levels)
    else:
        raise ValueError(
            "level_mode must be 'paper-exact', 'paper-visible', 'paper', 'data', or 'uniform'."
        )

    if not include_level_one:
        levels = levels[levels > 1.0 + 1e-12]
    if levels.size < 2:
        raise ValueError("Need at least two contour levels after filtering.")
    return levels


def plot_case(
    ax,
    case: FigureCase,
    grid_data: dict,
    E: np.ndarray,
    level_mode: str,
    levels_override: np.ndarray | None,
    num_levels: int,
    include_level_one: bool,
    show_material: bool,
) -> None:
    X = grid_data["Xc"]
    Y = grid_data["Yc"]
    T = energy_to_temperature(E)
    levels = contour_levels_for_case(
        case,
        T,
        level_mode,
        levels_override,
        num_levels,
        include_level_one,
    )

    contour = ax.contour(
        X,
        Y,
        T,
        levels=levels,
        colors="black",
        linewidths=0.7,
    )
    if level_mode == "paper-visible":
        label_levels = levels
    elif levels.size > 14:
        label_levels = levels[::2]
        if label_levels[-1] == levels[-1] and levels.size > 2:
            label_levels = label_levels[:-1]
    else:
        label_levels = levels[:-1] if levels.size > 2 else levels
    ax.clabel(contour, levels=label_levels, inline=True, fontsize=6, fmt="%.2f")

    if show_material:
        Z = grid_data["Z"]
        ax.contour(
            X,
            Y,
            Z,
            levels=[15.0, 35.0, 75.0],
            colors="0.55",
            linewidths=0.5,
            linestyles="dashed",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X", fontsize=13, fontweight="bold")
    ax.set_ylabel("Y", fontsize=13, fontweight="bold")
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([0.0, 1.0])
    minor_ticks = np.arange(0.1, 1.0, 0.1)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.tick_params(direction="in", top=True, right=True, which="both")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.text(
        0.10,
        1.02,
        case.panel,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    ax.text(
        0.5,
        -0.16,
        f"({case.panel}) {case.model}, eta = {case.eta:.2f}, t = {case.t_end:g}",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="top",
    )


def make_figure(
    case_payloads: list[tuple[FigureCase, int, np.ndarray]],
    output: Path,
    level_mode: str,
    levels_override: np.ndarray | None,
    num_levels: int,
    include_level_one: bool,
    show_material: bool,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.2,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.1), constrained_layout=False)
    for ax, (case, grid, E) in zip(axes, case_payloads):
        run_cfg = default_run_config(case.model, grid, grid, case.eta)
        problem = build_problem(run_cfg)
        grid_data = {
            "Xc": problem["grid"]["Xc"],
            "Yc": problem["grid"]["Yc"],
            "Z": problem["Z"],
        }
        plot_case(
            ax,
            case,
            grid_data,
            E,
            level_mode=level_mode,
            levels_override=levels_override,
            num_levels=num_levels,
            include_level_one=include_level_one,
            show_material=show_material,
        )

    fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.20, wspace=0.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def parse_levels(raw: str | None) -> np.ndarray:
    if raw is None:
        raise ValueError("parse_levels should only be called with a non-empty string.")
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) < 2:
        raise ValueError("--levels must contain at least two comma-separated values.")
    return np.asarray(values, dtype=float)


def grid_for_case(case: FigureCase, args: argparse.Namespace) -> int:
    """Return the grid selected for one panel."""
    if args.grid is not None:
        return int(args.grid)
    if case.model == "M1":
        return int(args.m1_grid)
    if case.model == "M2":
        return int(args.m2_grid)
    return int(case.default_grid)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid",
        type=int,
        default=None,
        help="Override both panels with one common grid size.",
    )
    parser.add_argument("--m1-grid", type=int, default=DEFAULT_M1_GRID)
    parser.add_argument("--m2-grid", type=int, default=DEFAULT_M2_GRID)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--num-levels",
        type=int,
        default=DEFAULT_NUM_LEVELS,
        help="Number of contour levels when --levels is not supplied.",
    )
    parser.add_argument(
        "--level-mode",
        choices=("paper-exact", "paper-visible", "paper", "data", "uniform"),
        default="paper",
        help=(
            "'paper-exact' uses the printed Fig. 4 labels; 'paper-visible' "
            "uses the less crowded visible labels for M2; 'paper' uses "
            "Fig. 4-like fixed ranges; 'data' spaces contours between each "
            "panel's min/max; 'uniform' uses 1..10."
        ),
    )
    parser.add_argument(
        "--levels",
        default=None,
        help="Optional comma-separated levels applied to both panels.",
    )
    parser.add_argument(
        "--include-level-one",
        action="store_true",
        default=DEFAULT_INCLUDE_LEVEL_ONE,
        help="Include the T=1.00 contour. By default it is skipped because it is the cold plateau.",
    )
    parser.add_argument("--rerun", action="store_true", help="Ignore checkpoints and rerun NK2.")
    parser.add_argument("--no-run", action="store_true", help="Fail instead of running missing cases.")
    parser.add_argument(
        "--show-material",
        action="store_true",
        help="Overlay dashed material-region contours.",
    )
    args = parser.parse_args()
    if args.num_levels < 2:
        raise ValueError("--num-levels must be at least 2.")

    levels_override = parse_levels(args.levels) if args.levels is not None else None
    payloads: list[tuple[FigureCase, int, np.ndarray]] = []
    metas: list[tuple[FigureCase, int, dict]] = []

    for case in FIGURE_CASES:
        grid = grid_for_case(case, args)
        E, meta = load_or_run_case(
            case,
            grid=grid,
            checkpoint_dir=args.checkpoint_dir,
            rerun=args.rerun,
            no_run=args.no_run,
        )
        payloads.append((case, grid, E))
        metas.append((case, grid, meta))

    make_figure(
        payloads,
        output=args.output,
        level_mode=args.level_mode,
        levels_override=levels_override,
        num_levels=args.num_levels,
        include_level_one=args.include_level_one,
        show_material=args.show_material,
    )

    print(f"[saved] {args.output}")
    print(f"[saved] {args.output.with_suffix('.pdf')}")
    print("\nCase summary")
    for case, grid, meta in metas:
        avg_linear = meta["avg_linear"]
        avg_nonlinear = meta["avg_nonlinear"]
        avg_linear_text = "nan" if not math.isfinite(avg_linear) else f"{avg_linear:.3f}"
        avg_nonlinear_text = "nan" if not math.isfinite(avg_nonlinear) else f"{avg_nonlinear:.3f}"
        print(
            f"  {case.model}: grid={grid}x{grid}, source={meta['source']}, "
            f"t={meta['t_final']:.6g}, steps={meta['steps']}, "
            f"avg_linear={avg_linear_text}, "
            f"avg_nonlinear={avg_nonlinear_text}, complete={meta['complete']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
