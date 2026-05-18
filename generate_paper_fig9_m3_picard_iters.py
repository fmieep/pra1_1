"""Generate paper-style M3 iteration-history plots.

The script does not rerun the PDE.  It reads existing checkpoints that contain
``time_history`` plus either ``nonlinear_iters_history`` or
``linear_iters_history`` and draws paper-style step curves for M3 Picard or
NK2 runs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, MultipleLocator
import numpy as np


DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("output_paper_fig9_m3_picard_iters")
DEFAULT_MODEL = "M3"
DEFAULT_METHOD = "picard"
DEFAULT_GRIDS = (32, 64, 128)
DEFAULT_ETAS = (0.10, 0.50)


@dataclass(frozen=True)
class CurveData:
    method: str
    eta: float
    grid: int
    time: np.ndarray
    iterations: np.ndarray
    checkpoint: Path


def eta_token(eta: float) -> str:
    return f"{eta:g}"


def checkpoint_path(checkpoint_dir: Path, method: str, eta: float, grid: int) -> Path:
    return checkpoint_dir / f"{DEFAULT_MODEL}_eta{eta_token(eta)}_grid{grid}_method{method}.npz"


def load_curve(
    checkpoint_dir: Path,
    method: str,
    eta: float,
    grid: int,
    metric: str,
) -> CurveData:
    path = checkpoint_path(checkpoint_dir, method, eta, grid)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}. Run the M3 {method.upper()} case first, or pass "
            "--checkpoint-dir to a directory that contains it."
        )

    history_key = f"{metric}_iters_history"
    with np.load(path) as data:
        missing = [
            key
            for key in ("time_history", history_key)
            if key not in data.files
        ]
        if missing:
            raise KeyError(f"{path} is missing arrays: {', '.join(missing)}")

        time = np.asarray(data["time_history"], dtype=float)
        iterations = np.asarray(data[history_key], dtype=float)

    if time.shape != iterations.shape:
        raise ValueError(
            f"{path} has inconsistent history lengths: "
            f"time={time.size}, {metric}={iterations.size}"
        )
    if time.size == 0:
        raise ValueError(f"{path} contains no accepted time steps.")

    return CurveData(
        method=method,
        eta=eta,
        grid=grid,
        time=time,
        iterations=iterations,
        checkpoint=path,
    )


def load_all_curves(
    checkpoint_dir: Path,
    method: str,
    etas: tuple[float, ...],
    grids: tuple[int, ...],
    metric: str,
):
    curves: dict[float, list[CurveData]] = {}
    for eta in etas:
        curves[eta] = [
            load_curve(checkpoint_dir, method, eta, grid, metric) for grid in grids
        ]
    return curves


def paper_x_ticks(eta: float) -> list[float]:
    if np.isclose(eta, 0.10):
        return [0.0, 0.01, 0.03, 0.05, 0.07, 0.09]
    if np.isclose(eta, 0.50):
        return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    return []


def paper_y_ticks(eta: float, y_max: float) -> list[float]:
    if np.isclose(eta, 0.10):
        if y_max <= 12.0:
            return [0.0, 10.0, 12.0]
        return [float(v) for v in range(0, int(np.ceil(y_max)) + 1, 5)]
    if np.isclose(eta, 0.50):
        if y_max <= 35.0:
            return [0.0, 10.0, 20.0, 30.0, 35.0]
        return [float(v) for v in range(0, int(np.ceil(y_max)) + 1, 10)]
    return []


def paper_y_limit(
    eta: float,
    panel_curves: list[CurveData],
    strict_paper_limits: bool,
) -> tuple[float, float]:
    max_iter = max(float(np.max(curve.iterations)) for curve in panel_curves)
    if np.isclose(eta, 0.10):
        paper_upper = 12.0
        if strict_paper_limits:
            return 0.0, paper_upper
        return 0.0, max(paper_upper, float(np.ceil(max_iter * 1.05)))
    if np.isclose(eta, 0.50):
        paper_upper = 35.0
        if strict_paper_limits:
            return 0.0, paper_upper
        return 0.0, max(paper_upper, float(np.ceil(max_iter * 1.05)))
    return 0.0, max(1.0, np.ceil(max_iter * 1.05))


def paper_x_limit(eta: float, panel_curves: list[CurveData]) -> tuple[float, float]:
    if np.isclose(eta, 0.10):
        return 0.0, 0.10
    if np.isclose(eta, 0.50):
        return 0.0, 0.50
    max_time = max(float(np.max(curve.time)) for curve in panel_curves)
    return 0.0, max_time


def time_formatter(value: float, _pos: int) -> str:
    if abs(value) < 5e-12:
        return "0"
    return f"{value:g}"


def style_for_grid(grid: int) -> tuple[str, float]:
    styles = {
        32: ("-", 0.95),
        64: ((0, (5.0, 5.0)), 0.95),
        128: (":", 1.05),
        256: ((0, (8.0, 3.0, 1.2, 3.0)), 0.95),
    }
    return styles.get(grid, ("-", 0.95))


def write_curve_csv(
    output_path: Path,
    curves: dict[float, list[CurveData]],
    metric: str,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["eta", "grid", "time", f"{metric}_iterations"])
        for eta in sorted(curves):
            for curve in curves[eta]:
                for t, iters in zip(curve.time, curve.iterations):
                    writer.writerow([f"{eta:g}", curve.grid, f"{t:.16g}", f"{iters:.16g}"])


def plot_curves(
    curves: dict[float, list[CurveData]],
    output_dir: Path,
    output_stem: str,
    dpi: int,
    strict_paper_limits: bool,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 1.2,
            "font.size": 11,
        }
    )

    etas = sorted(curves)
    fig, axes = plt.subplots(1, len(etas), figsize=(10.8, 4.25), squeeze=False)
    axes = axes[0]

    panel_labels = ["a", "b", "c", "d"]
    captions = []

    for idx, eta in enumerate(etas):
        ax = axes[idx]
        panel_curves = sorted(curves[eta], key=lambda c: c.grid)

        for curve in panel_curves:
            linestyle, linewidth = style_for_grid(curve.grid)
            ax.step(
                curve.time,
                curve.iterations,
                where="post",
                color="black",
                linestyle=linestyle,
                linewidth=linewidth,
                label=f"{curve.grid}x{curve.grid}",
            )

        ax.set_xlim(*paper_x_limit(eta, panel_curves))
        y_min, y_max = paper_y_limit(eta, panel_curves, strict_paper_limits)
        ax.set_ylim(y_min, y_max)
        xticks = paper_x_ticks(eta)
        yticks = paper_y_ticks(eta, y_max)
        if xticks:
            ax.xaxis.set_major_locator(FixedLocator(xticks))
        if yticks:
            ax.yaxis.set_major_locator(FixedLocator(yticks))

        ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(1))

        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.tick_params(which="major", length=6.0, width=1.1)
        ax.tick_params(which="minor", length=3.0, width=1.0)

        ax.set_xlabel("time", fontsize=16, fontweight="bold")
        ax.set_ylabel("Iterations", fontsize=14, fontweight="bold")
        ax.text(
            0.01,
            1.02,
            panel_labels[idx],
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 0.96),
            frameon=False,
            handlelength=2.2,
            handletextpad=0.7,
            borderaxespad=0.0,
            fontsize=9,
        )

        captions.append((idx, rf"({panel_labels[idx]}) $\eta$ = {eta:.2f}"))

    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.24, top=0.90, wspace=0.55)

    for idx, caption in captions:
        bbox = axes[idx].get_position()
        x = 0.5 * (bbox.x0 + bbox.x1)
        fig.text(x, 0.06, caption, ha="center", va="center", fontsize=15)

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a paper-style plot from M3 Picard checkpoint iteration "
            "histories."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default=None,
    )
    parser.add_argument(
        "--method",
        choices=("picard", "nk2"),
        default=DEFAULT_METHOD,
        help="Checkpoint method to plot.",
    )
    parser.add_argument(
        "--metric",
        choices=("nonlinear", "linear"),
        default="nonlinear",
        help="Which checkpoint iteration history to plot.",
    )
    parser.add_argument(
        "--grids",
        type=int,
        nargs="+",
        default=list(DEFAULT_GRIDS),
        help="Grid sizes to plot. Defaults to 32 64 128.",
    )
    parser.add_argument(
        "--etas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ETAS),
        help="Eta values to plot. Defaults to 0.10 0.50.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--strict-paper-limits",
        action="store_true",
        help=(
            "Use the paper panel y-limits exactly. By default the y-axis is "
            "expanded if the local checkpoint has larger iteration peaks."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_stem is None:
        args.output_stem = f"m3_{args.method}_{args.metric}_iterations_paper_style"

    etas = tuple(float(eta) for eta in args.etas)
    grids = tuple(int(grid) for grid in args.grids)
    curves = load_all_curves(args.checkpoint_dir, args.method, etas, grids, args.metric)

    csv_path = args.output_dir / f"{args.output_stem}.csv"
    write_curve_csv(csv_path, curves, args.metric)
    png_path, pdf_path = plot_curves(
        curves,
        args.output_dir,
        args.output_stem,
        args.dpi,
        args.strict_paper_limits,
    )

    print(f"saved png: {png_path}")
    print(f"saved pdf: {pdf_path}")
    print(f"saved csv: {csv_path}")
    for eta in sorted(curves):
        print(f"eta={eta:g}")
        for curve in sorted(curves[eta], key=lambda c: c.grid):
            avg = float(np.mean(curve.iterations))
            max_iter = float(np.max(curve.iterations))
            print(
                f"  {curve.grid}x{curve.grid}: steps={curve.time.size}, "
                f"avg={avg:.4g}, max={max_iter:.4g}, checkpoint={curve.checkpoint}"
            )


if __name__ == "__main__":
    main()
