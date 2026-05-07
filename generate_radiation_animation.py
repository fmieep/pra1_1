"""Generate a GIF showing the 2D radiation diffusion process.

Edit the parameter block near the top, then run:

    python generate_radiation_animation.py

The script reruns one case and records accepted time-step snapshots. Existing
checkpoint files only store the final field, so a rerun is needed for animation.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from config import SolverConfig, default_run_config
from driver import run_simulation
from physics import energy_to_temperature
from problem import build_problem


# =========================================================
# Animation switches: edit here, like test.py.
# =========================================================

# "single": run the single case below.
# "batch": run combinations from BATCH_* lists.
MODE = "batch"

METHOD = "nk2"  # "nk2" or "picard"
MODEL = "M2"  # "M1", "M2", or "M3"
ETA = 0.50  # usually 0.10 or 0.50

# Grid size. For the paper's square 2D examples, edit GRID_SIZE first.
# Set USE_SQUARE_GRID = False only if you intentionally want NX != NY.
USE_SQUARE_GRID = True
GRID_SIZE = 128
NX = 128
NY = 128

# FIELD controls what is plotted:
# - "T": temperature, T = E^(1/4), recommended and used in the paper's Fig. 4
# - "E": radiation energy
FIELD = "T"

# Batch settings. Keep the first batch small, then expand deliberately: each
# case reruns the PDE solve before making its GIF.
BATCH_METHODS = ["nk2"]
BATCH_MODELS = ["M1","M2","M3"]
BATCH_ETAS = [0.10,0.50]
BATCH_GRID_SIZES = [64,128]
BATCH_FIELDS = ["T"]

GIF_OUTPUT_DIR = "output_gif"
MP4_OUTPUT_DIR = "output_mp4"

# Output format:
# - "gif": uses Pillow, available in the current project environment
# - "mp4": requires ffmpeg to be installed and visible to matplotlib
OUTPUT_FORMAT = "mp4"
ANIMATION_FPS = 12
ANIMATION_DPI = 120
MP4_BITRATE = 1800

# Optional: if ffmpeg is installed but not on PATH, put the full path here.
# Example:
# FFMPEG_EXE = r"C:\ffmpeg\bin\ffmpeg.exe"
FFMPEG_EXE = ""

# Figure size in inches. Increase these if the colorbar/title looks cramped in
# the report, or decrease them for a smaller GIF file.
FIG_WIDTH = 6.4
FIG_HEIGHT = 5.6

# Save one frame every N accepted time steps. Smaller values show smoother
# motion but make the GIF larger and slower to create.
SNAPSHOT_EVERY_STEPS = 3
ALWAYS_SAVE_FINAL_FRAME = True

# M2 on 128x128 can fail before the paper default t_end with the current solver
# settings. When this is True, the script still saves a GIF up to the last
# accepted time so batch runs leave a useful partial animation.
SAVE_PARTIAL_GIF_ON_FAILURE = True

# Color scale:
# - "paper_temperature": fixed T range [1, 10], best for FIELD = "T"
# - "auto": use the min/max over recorded frames
# - "manual": use MANUAL_VMIN and MANUAL_VMAX
COLOR_MODE = "paper_temperature"
MANUAL_VMIN = 1.0
MANUAL_VMAX = 10.0
COLORMAP = "inferno"

# Display interpolation only changes how pixels are drawn in the GIF. It does
# not change the numerical solution.
# - "nearest": honest cell-by-cell finite-volume view
# - "bilinear": smoother report-friendly view
DISPLAY_INTERPOLATION = "nearest"

# If many snapshots are recorded, keep the GIF compact by choosing frames that
# are approximately uniform in physical time. Set to None to keep all snapshots.
MAX_GIF_FRAMES = 160

# Material contours help show how the front interacts with the multimaterial map.
SHOW_MATERIAL_CONTOURS = True

# Optional overrides. Leave as False to use paper-inspired defaults.
OVERRIDE_T_END = False
T_END = 0.5

OVERRIDE_DT_INIT = False
DT_INIT = 1e-6


# Solver parameters are copied from the current test.py defaults.
NONLINEAR_TOL = 2e-6
LINEAR_TOL_FACTOR = 3e-3
MAX_NONLINEAR_ITERS = 100
MAX_LINEAR_ITERS = 300
GMRES_RESTART = 100
RHO_JFNK = 1e-8
USE_MG_PRECONDITIONER = True
MG_SMOOTHER = "jacobi"
JFNK_EPS_MODE = "normalized"
DAMPING_NORM = "linf"
MG_PRE_SMOOTHS = 3
MG_POST_SMOOTHS = 3

DT_GROWTH_LIMIT = 1.1
E_FLOOR = 1.0
MAX_STEP_RETRIES = 12
PROGRESS_INTERVAL = 25
PICARD_MASS_MODE = "q_derivative"

# =========================================================
# Usually no need to edit below this line.
# =========================================================


@dataclass
class Snapshot:
    step: int
    t: float
    dt: float
    eta: float
    field: np.ndarray


def make_run_config():
    nx, ny = selected_grid_size()
    run_cfg = default_run_config(model=MODEL, nx=nx, ny=ny, eta_target=ETA)
    run_cfg.dt_growth_limit = DT_GROWTH_LIMIT
    run_cfg.e_floor = E_FLOOR
    run_cfg.max_step_retries = MAX_STEP_RETRIES
    run_cfg.progress_interval = PROGRESS_INTERVAL
    run_cfg.checkpoint_path = None
    run_cfg.checkpoint_interval = 0
    run_cfg.resume_from_checkpoint = False

    if OVERRIDE_T_END:
        run_cfg.t_end = T_END
    if OVERRIDE_DT_INIT:
        run_cfg.dt_init = DT_INIT

    return run_cfg


def make_solver_config():
    return SolverConfig(
        method=METHOD,
        nonlinear_tol=NONLINEAR_TOL,
        linear_tol_factor=LINEAR_TOL_FACTOR,
        max_nonlinear_iters=MAX_NONLINEAR_ITERS,
        max_linear_iters=MAX_LINEAR_ITERS,
        gmres_restart=GMRES_RESTART,
        rho_jfnk=RHO_JFNK,
        use_multigrid_preconditioner=USE_MG_PRECONDITIONER,
        mg_smoother=MG_SMOOTHER,
        jfnk_eps_mode=JFNK_EPS_MODE,
        damping_norm=DAMPING_NORM,
        mg_pre_smooths=MG_PRE_SMOOTHS,
        mg_post_smooths=MG_POST_SMOOTHS,
        picard_mass_mode=PICARD_MASS_MODE,
    )


def field_from_energy(E: np.ndarray) -> np.ndarray:
    field = FIELD.upper()
    if field == "T":
        return energy_to_temperature(E)
    if field == "E":
        return E.copy()
    raise ValueError('FIELD must be "T" or "E".')


def field_label() -> str:
    return "Temperature T = E^(1/4)" if FIELD.upper() == "T" else "Radiation energy E"


def color_limits(frames: list[Snapshot]) -> tuple[float, float]:
    if COLOR_MODE == "paper_temperature":
        if FIELD.upper() != "T":
            raise ValueError('COLOR_MODE="paper_temperature" is only valid for FIELD="T".')
        return 1.0, 10.0
    if COLOR_MODE == "manual":
        return float(MANUAL_VMIN), float(MANUAL_VMAX)
    if COLOR_MODE == "auto":
        values_min = min(float(np.min(frame.field)) for frame in frames)
        values_max = max(float(np.max(frame.field)) for frame in frames)
        if values_max <= values_min:
            values_max = values_min + 1e-12
        return values_min, values_max
    raise ValueError('COLOR_MODE must be "paper_temperature", "auto", or "manual".')


def select_gif_frames(snapshots: list[Snapshot]) -> list[Snapshot]:
    if MAX_GIF_FRAMES is None or len(snapshots) <= MAX_GIF_FRAMES:
        return snapshots

    target_times = np.linspace(snapshots[0].t, snapshots[-1].t, int(MAX_GIF_FRAMES))
    times = np.asarray([snap.t for snap in snapshots], dtype=float)
    selected_indices = np.searchsorted(times, target_times, side="left")
    selected_indices = np.clip(selected_indices, 0, len(snapshots) - 1)

    unique_indices = []
    seen = set()
    for idx in selected_indices:
        idx = int(idx)
        if idx not in seen:
            unique_indices.append(idx)
            seen.add(idx)

    if 0 not in seen:
        unique_indices.insert(0, 0)
    if len(snapshots) - 1 not in seen:
        unique_indices.append(len(snapshots) - 1)

    return [snapshots[idx] for idx in unique_indices]


def output_path() -> str:
    nx, _ = selected_grid_size()
    field = FIELD.upper()
    eta_text = f"{ETA:g}".replace(".", "p")
    output_format = OUTPUT_FORMAT.lower()
    if output_format not in ("gif", "mp4"):
        raise ValueError('OUTPUT_FORMAT must be "gif" or "mp4".')
    output_dir = GIF_OUTPUT_DIR if output_format == "gif" else MP4_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    name = f"{MODEL}_eta{eta_text}_grid{nx}_method{METHOD}_{field}.{output_format}"
    return os.path.join(output_dir, name)


def animation_writer():
    output_format = OUTPUT_FORMAT.lower()
    if output_format == "gif":
        return animation.PillowWriter(fps=ANIMATION_FPS)
    if output_format == "mp4":
        configure_ffmpeg_path()
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                'OUTPUT_FORMAT="mp4" requires ffmpeg, but matplotlib cannot find it. '
                'Install ffmpeg, set FFMPEG_EXE, or use OUTPUT_FORMAT="gif".'
            )
        return animation.FFMpegWriter(fps=ANIMATION_FPS, bitrate=MP4_BITRATE)
    raise ValueError('OUTPUT_FORMAT must be "gif" or "mp4".')


def configure_ffmpeg_path() -> None:
    if FFMPEG_EXE:
        matplotlib.rcParams["animation.ffmpeg_path"] = FFMPEG_EXE


def check_output_environment() -> None:
    output_format = OUTPUT_FORMAT.lower()
    if output_format == "gif":
        return
    if output_format == "mp4":
        configure_ffmpeg_path()
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                'Cannot create MP4 because ffmpeg is not available. '
                'Install ffmpeg, set FFMPEG_EXE, or switch OUTPUT_FORMAT back to "gif".'
            )
        return
    raise ValueError('OUTPUT_FORMAT must be "gif" or "mp4".')


def selected_grid_size() -> tuple[int, int]:
    if USE_SQUARE_GRID:
        return int(GRID_SIZE), int(GRID_SIZE)
    return int(NX), int(NY)


def run_current_case() -> dict:
    start = time.perf_counter()
    check_output_environment()
    nx, ny = selected_grid_size()
    run_cfg = make_run_config()
    solver_cfg = make_solver_config()
    problem = build_problem(run_cfg)

    snapshots: list[Snapshot] = []

    def record_snapshot(step, t, dt, eta, E, stats, accepted, initial=False):
        _ = stats, accepted
        should_save = initial or step % SNAPSHOT_EVERY_STEPS == 0
        if not should_save:
            return
        snapshots.append(
            Snapshot(
                step=int(step),
                t=float(t),
                dt=float(dt),
                eta=float(eta),
                field=field_from_energy(E),
            )
        )

    run_cfg.snapshot_callback = record_snapshot

    print(
        f"[animation] running {METHOD}, {MODEL}, eta={ETA}, grid={nx}x{ny}, "
        f"field={FIELD.upper()}",
        flush=True,
    )
    result = run_simulation(problem, run_cfg, solver_cfg)

    if not result["converged"] and not SAVE_PARTIAL_GIF_ON_FAILURE:
        raise RuntimeError(f"Simulation failed: {result['failure_reason']}")
    if not result["converged"]:
        print(
            "[animation] warning: simulation did not reach t_end; "
            "saving partial GIF up to the last accepted state.",
            flush=True,
        )
        print(f"[animation] failure reason: {result['failure_reason']}", flush=True)

    final_field = field_from_energy(result["E_final"])
    if ALWAYS_SAVE_FINAL_FRAME and (
        not snapshots or snapshots[-1].step != len(result["time_history"])
    ):
        snapshots.append(
            Snapshot(
                step=len(result["time_history"]),
                t=float(result["t_final"]),
                dt=float(result["dt_history"][-1]),
                eta=float(result["eta_history"][-1]),
                field=final_field,
            )
        )

    if len(snapshots) < 2:
        raise RuntimeError("Need at least two snapshots to build an animation.")

    snapshots_for_gif = select_gif_frames(snapshots)
    status_suffix = "" if result["converged"] else ", partial"
    vmin, vmax = color_limits(snapshots)
    grid = problem["grid"]
    extent = [grid["xlim"][0], grid["xlim"][1], grid["ylim"][0], grid["ylim"][1]]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)
    image = ax.imshow(
        snapshots[0].field.T,
        origin="lower",
        extent=extent,
        cmap=COLORMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation=DISPLAY_INTERPOLATION,
        animated=True,
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(field_label())

    if SHOW_MATERIAL_CONTOURS:
        ax.contour(
            grid["Xc"],
            grid["Yc"],
            problem["Z"],
            levels=[15.0, 35.0, 75.0],
            colors="white",
            linewidths=0.7,
            alpha=0.75,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    title = ax.set_title("")

    def update(frame_index: int):
        snap = snapshots_for_gif[frame_index]
        image.set_array(snap.field.T)
        title.set_text(
            f"{MODEL}, {METHOD.upper()}, eta={ETA:g}{status_suffix}, step={snap.step}, "
            f"t={snap.t:.4e}, dt={snap.dt:.2e}"
        )
        return image, title

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(snapshots_for_gif),
        interval=1000 / ANIMATION_FPS,
        blit=False,
    )

    animation_path = output_path()
    writer = animation_writer()
    anim.save(animation_path, writer=writer, dpi=ANIMATION_DPI)
    plt.close(fig)

    elapsed = time.perf_counter() - start
    print("[animation] saved:", animation_path)
    print(
        f"[animation] frames={len(snapshots_for_gif)} of {len(snapshots)} snapshots, "
        f"steps={len(result['time_history'])}, "
        f"t_final={result['t_final']:.6e}, elapsed={elapsed:.2f}s"
    )
    return {
        "method": METHOD,
        "model": MODEL,
        "eta": ETA,
        "grid": f"{nx}x{ny}",
        "field": FIELD.upper(),
        "path": animation_path,
        "steps": len(result["time_history"]),
        "frames": len(snapshots_for_gif),
        "t_final": result["t_final"],
        "elapsed": elapsed,
        "converged": result["converged"],
        "failure_reason": result["failure_reason"],
    }


def run_one_case(method: str, model: str, eta: float, grid_size: int, field: str) -> dict:
    global METHOD, MODEL, ETA, GRID_SIZE, FIELD, USE_SQUARE_GRID

    METHOD = method
    MODEL = model
    ETA = eta
    GRID_SIZE = grid_size
    FIELD = field
    USE_SQUARE_GRID = True

    return run_current_case()


def run_batch() -> None:
    rows = []
    batch_start = time.perf_counter()

    for method in BATCH_METHODS:
        for model in BATCH_MODELS:
            for eta in BATCH_ETAS:
                for grid_size in BATCH_GRID_SIZES:
                    for field in BATCH_FIELDS:
                        label = (
                            f"method={method}, model={model}, eta={eta}, "
                            f"grid={grid_size}x{grid_size}, field={field}"
                        )
                        print("\n" + "=" * 80)
                        print(f"[batch] start {label}")
                        print("=" * 80)
                        try:
                            row = run_one_case(method, model, eta, grid_size, field)
                            row["status"] = "OK" if row["converged"] else "PARTIAL"
                            row["error"] = "" if row["converged"] else row["failure_reason"]
                        except Exception as exc:
                            row = {
                                "method": method,
                                "model": model,
                                "eta": eta,
                                "grid": f"{grid_size}x{grid_size}",
                                "field": field.upper(),
                                "path": "",
                                "steps": 0,
                                "frames": 0,
                                "t_final": float("nan"),
                                "elapsed": 0.0,
                                "status": "FAIL",
                                "error": str(exc),
                            }
                            print(f"[batch] failed: {exc}")
                        rows.append(row)

    total_elapsed = time.perf_counter() - batch_start
    print("\n" + "#" * 80)
    print("[batch] summary")
    print("#" * 80)
    for row in rows:
        print(
            f"[{row['status']}] "
            f"method={row['method']:<6} "
            f"model={row['model']:<2} "
            f"eta={row['eta']:<4} "
            f"grid={row['grid']:<9} "
            f"field={row['field']:<2} "
            f"frames={row['frames']:<4} "
            f"steps={row['steps']:<5} "
            f"path={row['path']} "
            f"error={row['error']}"
        )
    print(f"[batch] total elapsed={total_elapsed:.2f}s")


def main() -> None:
    if MODE == "single":
        run_current_case()
    elif MODE == "batch":
        run_batch()
    else:
        raise ValueError('MODE must be "single" or "batch".')


if __name__ == "__main__":
    main()
