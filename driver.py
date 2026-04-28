"""Time-integration driver with eta-based step control."""

from __future__ import annotations
import numpy as np
from methods import solve_one_step
import os


def compute_eta(E_new: np.ndarray, E_old: np.ndarray, e_floor: float = 1.0) -> float:
    """Measure the maximum relative change over one time step."""
    ratio = np.abs(E_new - E_old) / (np.maximum(E_new, 0.0) + e_floor)
    return float(np.max(ratio))


def update_dt(
    dt_old: float, eta_measured: float, eta_target: float, growth_limit: float = 1.1
) -> float:
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
    linear_iters_per_nonlinear_history = []
    linear_iters_by_step = []
    eta_history = []
    residual_history = []

    checkpoint_path = getattr(run_cfg, "checkpoint_path", None)
    resume_from_checkpoint = getattr(run_cfg, "resume_from_checkpoint", False)

    if resume_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = np.load(checkpoint_path, allow_pickle=True)

        t = float(ckpt["t"])
        dt = float(ckpt["dt"])
        E = ckpt["E"].copy()

        time_history = ckpt["time_history"].tolist()
        dt_history = ckpt["dt_history"].tolist()
        linear_iters_history = ckpt["linear_iters_history"].tolist()
        nonlinear_iters_history = ckpt["nonlinear_iters_history"].tolist()
        eta_history = ckpt["eta_history"].tolist()
        residual_history = ckpt["residual_history"].tolist()

        print(
            f"[resume] loaded checkpoint from {checkpoint_path}, "
            f"step={len(time_history)}, t={t:.6e}, dt={dt:.3e}",
            flush=True,
        )

    failed_step = None
    failure_reason = None
    converged_all = True

    max_step_retries = getattr(run_cfg, "max_step_retries", 0)

    def save_checkpoint(label: str = "checkpoint"):
        if not checkpoint_path:
            return

        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        np.savez_compressed(
            checkpoint_path,
            t=t,
            dt=dt,
            E=E,
            time_history=np.asarray(time_history, dtype=float),
            dt_history=np.asarray(dt_history, dtype=float),
            linear_iters_history=np.asarray(linear_iters_history, dtype=int),
            nonlinear_iters_history=np.asarray(nonlinear_iters_history, dtype=int),
            eta_history=np.asarray(eta_history, dtype=float),
            residual_history=np.asarray(residual_history, dtype=float),
        )

        print(
            f"[{label}] saved step={len(time_history)}, "
            f"t={t:.6e}, file={checkpoint_path}",
            flush=True,
        )

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
        step_linear_iters = last_stats.get("linear_iters_by_nonlinear", [])
        linear_iters_by_step.append(step_linear_iters)
        linear_iters_per_nonlinear_history.extend(step_linear_iters)
        eta_history.append(eta)
        residual_history.append(last_stats["final_residual_norm"])

        step_no = len(time_history)
        progress_interval = getattr(run_cfg, "progress_interval", 0)

        if progress_interval and step_no % progress_interval == 0:
            print(
                f"[progress] step={step_no}, "
                f"t={t + dt_try:.6e}/{run_cfg.t_end:.6e}, "
                f"dt={dt_try:.3e}, eta={eta:.3e}, "
                f"lin={last_stats['linear_iters_total']}, "
                f"nonlin={last_stats['nonlinear_iters']}, "
                f"res={last_stats['final_residual_norm']:.3e}",
                flush=True,
            )

        if not step_success:
            converged_all = False
            failed_step = len(time_history)
            failure_reason = (
                f"{solver_cfg.method} failed after retries at "
                f"t={t + dt_try:.6e}, dt={dt_try:.6e}, "
                f"residual={last_stats['final_residual_norm']:.6e}"
            )

            # 失败时也保存，方便之后检查最后状态
            save_checkpoint(label="checkpoint-failed")
            break

        E = last_E_new
        t += dt_try
        dt = update_dt(dt_try, eta, run_cfg.eta_target, run_cfg.dt_growth_limit)

        checkpoint_interval = getattr(run_cfg, "checkpoint_interval", 0)
        if checkpoint_interval and step_no % checkpoint_interval == 0:
            save_checkpoint(label="checkpoint")

    # 正常结束或中断前最终保存一次
    save_checkpoint(label="checkpoint-final")

    return {
        "E_final": E,
        "time_history": time_history,
        "dt_history": dt_history,
        "linear_iters_history": linear_iters_history,
        "nonlinear_iters_history": nonlinear_iters_history,
        "linear_iters_per_nonlinear_history": linear_iters_per_nonlinear_history,
        "linear_iters_by_step": linear_iters_by_step,
        "eta_history": eta_history,
        "residual_history": residual_history,
        "converged": converged_all,
        "failed_step": failed_step,
        "failure_reason": failure_reason,
        "t_final": t,
    }
