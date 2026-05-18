"""Generate long-time diffusion animations and truth-data snapshots.

This script is for long-time diffusion-process visualization and deep-learning
truth-data generation.  It is not intended for strict reproduction of the
paper's default final times.  For paper-default animations, keep using
generate_radiation_animation.py.

Edit the parameter block near the top, then run:

    python generate_full_diffusion_animation.py

For long-time runs, increase T_END gradually.  Large T_END values can be
expensive or may expose solver limitations for difficult nonlinear cases.
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
# Long-time full-diffusion switches: edit here.
# =========================================================

# "single": run the single case below.
# "batch": run combinations from BATCH_* lists.
MODE = "batch"

METHOD = "nk2"  # Default and recommended: "nk2". Also supports "picard".
MODEL = "M3"  # "M1", "M2", or "M3"
ETA = 0.50
GRID_SIZE = 128
FIELD = "T"  # "T" for temperature T=E^(1/4), or "E" for energy.

# Long-time setting.  This intentionally overrides paper-inspired defaults.
# Paper reproduction: use generate_radiation_animation.py.
# Long-time truth data/full diffusion process: use this script.
# Increase T_END gradually; do not jump to a very large value first.
OVERRIDE_T_END = True
T_END = 5.0

OVERRIDE_DT_INIT = False
DT_INIT = 1e-6

# Batch mode.  This is a Cartesian product; be careful mixing models with very
# different useful T_END values.
BATCH_METHODS = ["nk2"]
BATCH_MODELS = ["M2"]
BATCH_ETAS = [0.10,0.50]
BATCH_GRID_SIZES = [128]
BATCH_T_ENDS = [1.0]
BATCH_FIELDS = ["T"]

MP4_OUTPUT_DIR = "output_full_mp4"
GIF_OUTPUT_DIR = "output_full_gif"
TRUTH_OUTPUT_DIR = "output_truth_npz"

# Solver checkpoints are useful for long runs.  Final MP4/NPZ files are still
# written only after run_simulation returns, but these checkpoints let you see
# that progress is being saved and allow later continuation if needed.  If you
# resume from a checkpoint, the saved truth snapshots start at the resumed state
# rather than at t=0, so use RESUME_FROM_CHECKPOINT=False for fresh truth data.
SAVE_SOLVER_CHECKPOINTS = True
CHECKPOINT_DIR = "checkpoints_full_diffusion"
CHECKPOINT_INTERVAL = 1000
RESUME_FROM_CHECKPOINT = False

# Output format:
# - "mp4": requires ffmpeg
# - "gif": uses Pillow
OUTPUT_FORMAT = "mp4"
ANIMATION_FPS = 12
ANIMATION_DPI = 120
MP4_BITRATE = 1800

# Optional: if ffmpeg is installed but not on PATH, put the full path here.
# Example:
# FFMPEG_EXE = r"C:\ffmpeg\bin\ffmpeg.exe"
FFMPEG_EXE = ""

FIG_WIDTH = 6.6
FIG_HEIGHT = 5.8

# Improved physical-time sampling.  The PDE solver still uses adaptive time
# steps, but output frames are linearly interpolated onto uniform target times.
# - "fixed_dt_interpolated": use OUTPUT_DT, so longer T_END produces more
#   frames and comparable physical playback speed.
# - "fixed_count_interpolated": use NUM_SNAPSHOTS frames over [0, T_END], so
#   every movie has roughly the same duration.
OUTPUT_TIME_MODE = "fixed_dt_interpolated"
OUTPUT_DT = 0.01
NUM_SNAPSHOTS = 240

# Color scale:
# - "paper_temperature": fixed T range [1, 10], for FIELD="T"
# - "auto": min/max over saved frames
# - "manual": MANUAL_VMIN/MANUAL_VMAX
COLOR_MODE = "paper_temperature"
MANUAL_VMIN = 1.0
MANUAL_VMAX = 10.0
COLORMAP = "inferno"
DISPLAY_INTERPOLATION = "nearest"
SHOW_MATERIAL_CONTOURS = True

# Solver parameters aligned with the current successful test.py configuration.
NONLINEAR_TOL = 2e-6
LINEAR_TOL_FACTOR = 3e-3
MAX_NONLINEAR_ITERS = 60
MAX_LINEAR_ITERS = 100
GMRES_RESTART = MAX_LINEAR_ITERS
RHO_JFNK = 1e-8
USE_MG_PRECONDITIONER = True
MG_SMOOTHER = "jacobi"
JFNK_EPS_MODE = "normalized"
DAMPING_NORM = "linf"
MG_PRE_SMOOTHS = 3
MG_POST_SMOOTHS = 3
PICARD_MASS_MODE = "q_derivative"

DT_GROWTH_LIMIT = 1.1
E_FLOOR = 1.0
MAX_STEP_RETRIES = 8
PROGRESS_INTERVAL = 100

# =========================================================
# Usually no need to edit below this line.
# =========================================================


@dataclass
class FullSnapshot:
    step: int
    t: float
    dt: float
    eta: float
    E: np.ndarray
    T: np.ndarray


def format_float_for_filename(value: float) -> str:
    value = float(value)
    if value.is_integer():
        text = f"{value:.1f}"
    else:
        text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def make_run_config(method: str, model: str, eta: float, grid_size: int, t_end: float):
    run_cfg = default_run_config(
        model=model,
        nx=int(grid_size),
        ny=int(grid_size),
        eta_target=float(eta),
    )
    run_cfg.progress_interval = PROGRESS_INTERVAL
    if SAVE_SOLVER_CHECKPOINTS:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        eta_text = format_float_for_filename(eta)
        tend_text = format_float_for_filename(t_end)
        run_cfg.checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"{model}_eta{eta_text}_grid{grid_size}_method{method}_tend{tend_text}.npz",
        )
        run_cfg.checkpoint_interval = CHECKPOINT_INTERVAL
        run_cfg.resume_from_checkpoint = RESUME_FROM_CHECKPOINT
    else:
        run_cfg.checkpoint_path = None
        run_cfg.checkpoint_interval = 0
        run_cfg.resume_from_checkpoint = False
    run_cfg.dt_growth_limit = DT_GROWTH_LIMIT
    run_cfg.e_floor = E_FLOOR
    run_cfg.max_step_retries = MAX_STEP_RETRIES

    if OVERRIDE_T_END:
        run_cfg.t_end = float(t_end)
    if OVERRIDE_DT_INIT:
        run_cfg.dt_init = DT_INIT

    return run_cfg


def make_solver_config(method: str) -> SolverConfig:
    return SolverConfig(
        method=method,
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


def solver_parameter_payload() -> dict[str, object]:
    return {
        "solver_nonlinear_tol": NONLINEAR_TOL,
        "solver_linear_tol_factor": LINEAR_TOL_FACTOR,
        "solver_max_nonlinear_iters": MAX_NONLINEAR_ITERS,
        "solver_max_linear_iters": MAX_LINEAR_ITERS,
        "solver_gmres_restart": GMRES_RESTART,
        "solver_rho_jfnk": RHO_JFNK,
        "solver_use_mg_preconditioner": USE_MG_PRECONDITIONER,
        "solver_mg_smoother": MG_SMOOTHER,
        "solver_jfnk_eps_mode": JFNK_EPS_MODE,
        "solver_damping_norm": DAMPING_NORM,
        "solver_mg_pre_smooths": MG_PRE_SMOOTHS,
        "solver_mg_post_smooths": MG_POST_SMOOTHS,
        "solver_picard_mass_mode": PICARD_MASS_MODE,
        "run_dt_growth_limit": DT_GROWTH_LIMIT,
        "run_e_floor": E_FLOOR,
        "run_max_step_retries": MAX_STEP_RETRIES,
        "output_time_mode": OUTPUT_TIME_MODE,
        "output_dt": OUTPUT_DT,
        "output_num_snapshots": NUM_SNAPSHOTS,
    }


def snapshot_field(snapshot: FullSnapshot, field: str) -> np.ndarray:
    field_key = field.upper()
    if field_key == "T":
        return snapshot.T
    if field_key == "E":
        return snapshot.E
    raise ValueError('FIELD must be "T" or "E".')


def field_label(field: str) -> str:
    return "Temperature T = E^(1/4)" if field.upper() == "T" else "Radiation energy E"


def color_limits(snapshots: list[FullSnapshot], field: str) -> tuple[float, float]:
    field_key = field.upper()
    if COLOR_MODE == "paper_temperature":
        if field_key != "T":
            raise ValueError('COLOR_MODE="paper_temperature" requires FIELD="T".')
        return 1.0, 10.0
    if COLOR_MODE == "manual":
        return float(MANUAL_VMIN), float(MANUAL_VMAX)
    if COLOR_MODE == "auto":
        values = [snapshot_field(snap, field_key) for snap in snapshots]
        vmin = min(float(np.min(value)) for value in values)
        vmax = max(float(np.max(value)) for value in values)
        if vmax <= vmin:
            vmax = vmin + 1e-12
        return vmin, vmax
    raise ValueError('COLOR_MODE must be "paper_temperature", "auto", or "manual".')


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
                "Install ffmpeg; or set FFMPEG_EXE; or set OUTPUT_FORMAT = "
                '"gif".'
            )
        return
    raise ValueError('OUTPUT_FORMAT must be "mp4" or "gif".')


def animation_writer():
    output_format = OUTPUT_FORMAT.lower()
    if output_format == "gif":
        return animation.PillowWriter(fps=ANIMATION_FPS)
    if output_format == "mp4":
        configure_ffmpeg_path()
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                'OUTPUT_FORMAT="mp4" requires ffmpeg. Install ffmpeg; or set '
                'FFMPEG_EXE; or set OUTPUT_FORMAT = "gif".'
            )
        return animation.FFMpegWriter(fps=ANIMATION_FPS, bitrate=MP4_BITRATE)
    raise ValueError('OUTPUT_FORMAT must be "mp4" or "gif".')


def output_paths(
    method: str,
    model: str,
    eta: float,
    grid_size: int,
    field: str,
    t_end: float,
    converged: bool,
) -> tuple[str, str]:
    eta_text = format_float_for_filename(eta)
    tend_text = format_float_for_filename(t_end)
    status_suffix = "" if converged else "_partial"
    output_format = OUTPUT_FORMAT.lower()

    if output_format == "mp4":
        movie_dir = MP4_OUTPUT_DIR
    elif output_format == "gif":
        movie_dir = GIF_OUTPUT_DIR
    else:
        raise ValueError('OUTPUT_FORMAT must be "mp4" or "gif".')

    os.makedirs(movie_dir, exist_ok=True)
    os.makedirs(TRUTH_OUTPUT_DIR, exist_ok=True)

    movie_name = (
        f"{model}_eta{eta_text}_grid{grid_size}_method{method}_"
        f"{field.upper()}_tend{tend_text}{status_suffix}.{output_format}"
    )
    truth_name = (
        f"{model}_eta{eta_text}_grid{grid_size}_method{method}_"
        f"tend{tend_text}{status_suffix}_truth.npz"
    )
    return os.path.join(movie_dir, movie_name), os.path.join(TRUTH_OUTPUT_DIR, truth_name)


def append_snapshot(
    snapshots: list[FullSnapshot],
    step: int,
    t: float,
    dt: float,
    eta: float,
    E: np.ndarray,
) -> None:
    if snapshots and snapshots[-1].step == int(step) and abs(snapshots[-1].t - float(t)) < 1e-14:
        return
    E_copy = E.copy()
    snapshots.append(
        FullSnapshot(
            step=int(step),
            t=float(t),
            dt=float(dt),
            eta=float(eta),
            E=E_copy,
            T=energy_to_temperature(E_copy),
        )
    )


def build_target_times(t_end: float) -> np.ndarray:
    t_end = float(t_end)
    if t_end < 0.0:
        raise ValueError("T_END must be nonnegative.")
    if OUTPUT_TIME_MODE == "fixed_count_interpolated":
        if int(NUM_SNAPSHOTS) < 2:
            raise ValueError("NUM_SNAPSHOTS must be at least 2.")
        return np.linspace(0.0, t_end, int(NUM_SNAPSHOTS))
    if OUTPUT_TIME_MODE == "fixed_dt_interpolated":
        output_dt = float(OUTPUT_DT)
        if output_dt <= 0.0:
            raise ValueError("OUTPUT_DT must be positive.")
        target_times = list(np.arange(0.0, t_end + 0.5 * output_dt, output_dt))
        if not target_times or target_times[0] != 0.0:
            target_times.insert(0, 0.0)
        if target_times[-1] < t_end - 1e-14:
            target_times.append(t_end)
        else:
            target_times[-1] = t_end
        return np.asarray(target_times, dtype=float)
    raise ValueError(
        'OUTPUT_TIME_MODE must be "fixed_dt_interpolated" or '
        '"fixed_count_interpolated".'
    )


def make_time_sampler(t_end: float, snapshots: list[FullSnapshot]):
    target_times = build_target_times(t_end)
    next_target_index = {"value": 0}
    previous = {"step": None, "t": None, "dt": None, "eta": None, "E": None}

    def callback(step, t, dt, eta, E, stats, accepted, initial=False):
        _ = stats, accepted
        t_now = float(t)
        E_now = E.copy()
        if initial:
            append_snapshot(snapshots, step, target_times[0], dt, eta, E_now)
            previous.update(
                {
                    "step": int(step),
                    "t": t_now,
                    "dt": float(dt),
                    "eta": float(eta),
                    "E": E_now,
                }
            )
            while (
                next_target_index["value"] < len(target_times)
                and target_times[next_target_index["value"]] <= t_now + 1e-14
            ):
                next_target_index["value"] += 1
            return

        if previous["E"] is None or previous["t"] is None:
            previous.update(
                {
                    "step": int(step),
                    "t": t_now,
                    "dt": float(dt),
                    "eta": float(eta),
                    "E": E_now,
                }
            )
            return

        t_prev = float(previous["t"])
        E_prev = previous["E"]
        if t_now < t_prev:
            raise RuntimeError("Snapshot times must be monotone.")

        while next_target_index["value"] < len(target_times):
            target_t = float(target_times[next_target_index["value"]])
            if target_t > t_now + 1e-14:
                break
            if target_t + 1e-14 < t_prev:
                next_target_index["value"] += 1
                continue

            if abs(t_now - t_prev) <= 1e-30:
                alpha = 1.0
            else:
                alpha = (target_t - t_prev) / (t_now - t_prev)
                alpha = min(1.0, max(0.0, alpha))
            E_interp = (1.0 - alpha) * E_prev + alpha * E_now
            append_snapshot(snapshots, step, target_t, dt, eta, E_interp)
            next_target_index["value"] += 1

        previous.update(
            {
                "step": int(step),
                "t": t_now,
                "dt": float(dt),
                "eta": float(eta),
                "E": E_now,
            }
        )

    return callback


def ensure_final_snapshot(snapshots: list[FullSnapshot], result: dict) -> None:
    dt_history = result["dt_history"]
    eta_history = result["eta_history"]
    final_dt = float(dt_history[-1]) if dt_history else 0.0
    final_eta = float(eta_history[-1]) if eta_history else 0.0
    append_snapshot(
        snapshots,
        step=len(result["time_history"]),
        t=float(result["t_final"]),
        dt=final_dt,
        eta=final_eta,
        E=result["E_final"],
    )


def save_truth_npz(
    path: str,
    snapshots: list[FullSnapshot],
    problem: dict,
    method: str,
    model: str,
    eta: float,
    grid_size: int,
    t_end_requested: float,
    result: dict,
) -> None:
    E_snapshots = np.stack([snap.E for snap in snapshots], axis=0)
    T_snapshots = np.stack([snap.T for snap in snapshots], axis=0)
    times = np.asarray([snap.t for snap in snapshots], dtype=float)
    steps = np.asarray([snap.step for snap in snapshots], dtype=int)
    dts = np.asarray([snap.dt for snap in snapshots], dtype=float)
    etas = np.asarray([snap.eta for snap in snapshots], dtype=float)
    grid = problem["grid"]

    payload = {
        "E_snapshots": E_snapshots,
        "T_snapshots": T_snapshots,
        "times": times,
        "steps": steps,
        "dts": dts,
        "etas": etas,
        "Z": problem["Z"],
        "x": grid["xc"],
        "y": grid["yc"],
        "model": model,
        "eta_target": float(eta),
        "grid_size": int(grid_size),
        "method": method,
        "t_end_requested": float(t_end_requested),
        "t_final": float(result["t_final"]),
        "converged": bool(result["converged"]),
        "failure_reason": "" if result["failure_reason"] is None else result["failure_reason"],
    }
    payload.update(solver_parameter_payload())
    np.savez_compressed(path, **payload)


def save_animation(
    path: str,
    snapshots: list[FullSnapshot],
    problem: dict,
    method: str,
    model: str,
    eta: float,
    grid_size: int,
    field: str,
    t_end_requested: float,
    converged: bool,
) -> None:
    if not snapshots:
        raise RuntimeError("No snapshots were recorded.")

    # FuncAnimation can be fragile with a single frame.  Duplicate it for the
    # movie only; the truth npz keeps the original snapshot list.
    movie_snapshots = snapshots if len(snapshots) >= 2 else [snapshots[0], snapshots[0]]
    vmin, vmax = color_limits(movie_snapshots, field)
    grid = problem["grid"]
    extent = [grid["xlim"][0], grid["xlim"][1], grid["ylim"][0], grid["ylim"][1]]
    status_text = "" if converged else " partial"

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)
    image = ax.imshow(
        snapshot_field(movie_snapshots[0], field).T,
        origin="lower",
        extent=extent,
        cmap=COLORMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation=DISPLAY_INTERPOLATION,
        animated=True,
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(field_label(field))

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
        snap = movie_snapshots[frame_index]
        image.set_array(snapshot_field(snap, field).T)
        title.set_text(
            f"{model}, {method.upper()}, eta={eta:g}{status_text}, "
            f"step={snap.step}, t={snap.t:.4e}, dt={snap.dt:.2e}, "
            f"T_END={t_end_requested:g}"
        )
        return image, title

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(movie_snapshots),
        interval=1000 / ANIMATION_FPS,
        blit=False,
    )
    anim.save(path, writer=animation_writer(), dpi=ANIMATION_DPI)
    plt.close(fig)


def run_case(method: str, model: str, eta: float, grid_size: int, t_end: float, field: str) -> dict:
    start = time.perf_counter()
    check_output_environment()

    run_cfg = make_run_config(method, model, eta, grid_size, t_end)
    solver_cfg = make_solver_config(method)
    problem = build_problem(run_cfg)
    snapshots: list[FullSnapshot] = []
    run_cfg.snapshot_callback = make_time_sampler(t_end, snapshots)

    print(
        f"[full-animation] running method={method}, model={model}, eta={eta}, "
        f"grid={grid_size}x{grid_size}, T_END={t_end}, field={field.upper()}",
        flush=True,
    )
    result = run_simulation(problem, run_cfg, solver_cfg)
    ensure_final_snapshot(snapshots, result)

    if not result["converged"]:
        print(
            f"[full-animation] warning: partial result, requested T_END={t_end}, "
            f"actual t_final={result['t_final']:.6e}",
            flush=True,
        )
        print(f"[full-animation] failure_reason: {result['failure_reason']}", flush=True)

    movie_path, npz_path = output_paths(
        method,
        model,
        eta,
        grid_size,
        field,
        t_end,
        converged=bool(result["converged"]),
    )
    save_truth_npz(
        npz_path,
        snapshots,
        problem,
        method,
        model,
        eta,
        grid_size,
        t_end,
        result,
    )
    save_animation(
        movie_path,
        snapshots,
        problem,
        method,
        model,
        eta,
        grid_size,
        field,
        t_end,
        converged=bool(result["converged"]),
    )

    elapsed = time.perf_counter() - start
    summary = {
        "method": method,
        "model": model,
        "eta": float(eta),
        "grid": int(grid_size),
        "requested_t_end": float(t_end),
        "actual_t_final": float(result["t_final"]),
        "converged": bool(result["converged"]),
        "num_accepted_steps": len(result["time_history"]),
        "num_saved_snapshots": len(snapshots),
        "mp4_path": movie_path,
        "npz_path": npz_path,
        "elapsed_seconds": elapsed,
        "failure_reason": result["failure_reason"],
    }
    print_case_summary(summary)
    return summary


def print_case_summary(summary: dict) -> None:
    print("\n" + "=" * 72)
    print("[full-animation] case summary")
    print("=" * 72)
    print(f"method             = {summary['method']}")
    print(f"model              = {summary['model']}")
    print(f"eta                = {summary['eta']}")
    print(f"grid               = {summary['grid']} x {summary['grid']}")
    print(f"requested T_END    = {summary['requested_t_end']}")
    print(f"actual t_final     = {summary['actual_t_final']}")
    print(f"converged          = {summary['converged']}")
    print(f"accepted steps     = {summary['num_accepted_steps']}")
    print(f"saved snapshots    = {summary['num_saved_snapshots']}")
    print(f"MP4/GIF path       = {summary['mp4_path']}")
    print(f"NPZ path           = {summary['npz_path']}")
    print(f"elapsed seconds    = {summary['elapsed_seconds']:.3f}")
    print(f"failure_reason     = {summary['failure_reason']}")
    print("=" * 72)


def run_single() -> dict:
    return run_case(METHOD, MODEL, ETA, GRID_SIZE, T_END, FIELD)


def run_batch() -> None:
    rows = []
    batch_start = time.perf_counter()
    for method in BATCH_METHODS:
        for model in BATCH_MODELS:
            for eta in BATCH_ETAS:
                for grid_size in BATCH_GRID_SIZES:
                    for t_end in BATCH_T_ENDS:
                        for field in BATCH_FIELDS:
                            try:
                                rows.append(run_case(method, model, eta, grid_size, t_end, field))
                            except Exception as exc:
                                print(
                                    f"[full-animation] warning: case failed before output: "
                                    f"method={method}, model={model}, eta={eta}, "
                                    f"grid={grid_size}, T_END={t_end}, error={exc}",
                                    flush=True,
                                )
                                rows.append(
                                    {
                                        "method": method,
                                        "model": model,
                                        "eta": float(eta),
                                        "grid": int(grid_size),
                                        "requested_t_end": float(t_end),
                                        "actual_t_final": float("nan"),
                                        "converged": False,
                                        "num_accepted_steps": 0,
                                        "num_saved_snapshots": 0,
                                        "mp4_path": "",
                                        "npz_path": "",
                                        "elapsed_seconds": 0.0,
                                        "failure_reason": str(exc),
                                    }
                                )

    total_elapsed = time.perf_counter() - batch_start
    print("\n" + "#" * 96)
    print("[full-animation] batch summary")
    print("#" * 96)
    for row in rows:
        status = "OK" if row["converged"] else "PARTIAL/FAIL"
        print(
            f"[{status}] "
            f"method={row['method']:<6} "
            f"model={row['model']:<2} "
            f"eta={row['eta']:<4g} "
            f"grid={row['grid']}x{row['grid']:<4} "
            f"T_END={row['requested_t_end']:<8g} "
            f"t_final={row['actual_t_final']:<12.6g} "
            f"steps={row['num_accepted_steps']:<6} "
            f"snapshots={row['num_saved_snapshots']:<4} "
            f"movie={row['mp4_path']} "
            f"npz={row['npz_path']} "
            f"reason={row['failure_reason']}"
        )
    print(f"[full-animation] batch elapsed seconds = {total_elapsed:.3f}")


def main() -> None:
    if MODE == "single":
        run_single()
    elif MODE == "batch":
        run_batch()
    else:
        raise ValueError('MODE must be "single" or "batch".')


if __name__ == "__main__":
    main()
