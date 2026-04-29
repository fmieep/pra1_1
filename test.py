from config import SolverConfig, default_run_config
from problem import build_problem
from driver import run_simulation
import numpy as np
import time
import os
import csv
from datetime import datetime
# =========================================================
# 这里是“总开关”：改这里就行
# =========================================================

MODE = "batch"  # 可选: "single" 或 "batch"

# ---------- 单个测试参数 ----------
METHOD = "nk2"  # 可选: "nk2" 或 "picard"
MODEL = "M2"  # 可选: "M1", "M2", "M3"
ETA = 0.50  # 常用: 0.10 或 0.50
NX = 128
NY = 128


# ---------- 求解器细参数 ----------
NONLINEAR_TOL = 1.2e-6
LINEAR_TOL_FACTOR = 3e-3
MAX_NONLINEAR_ITERS = 60
MAX_LINEAR_ITERS = 100
GMRES_RESTART = MAX_LINEAR_ITERS
RHO_JFNK = 1e-8
USE_MG_PRECONDITIONER = True
MG_SMOOTHER = "jacobi"  # 可选: "jacobi", "gs", "sgs"
JFNK_EPS_MODE = "normalized"
DAMPING_NORM = "linf"
MG_PRE_SMOOTHS = 3
MG_POST_SMOOTHS = 3
PROGRESS_INTERVAL = 1000
CHECKPOINT_INTERVAL = 1000
RESUME_FROM_CHECKPOINT = False
CHECKPOINT_DIR = "checkpoints"
RESULT_CSV_PATH = os.path.join("output", "test_results.csv")

# ---------- 运行细参数 ----------
# 是否覆盖 default_run_config 里的默认值
OVERRIDE_T_END = False
T_END = 0.002

OVERRIDE_DT_INIT = False
DT_INIT = 1e-6

DT_GROWTH_LIMIT = 1.1
E_FLOOR = 1.0

# ---------- 批量测试参数 ----------
BATCH_METHODS = ["nk2","picard"]
BATCH_MODELS = ["M3"]
BATCH_ETAS = [0.10,0.50]
BATCH_GRIDS = [32,64,128,256]

# =========================================================
# 下面一般不用改
# =========================================================


def safe_average(values):
    return float("nan") if len(values) == 0 else sum(values) / len(values)


def make_run_config(model, nx, ny, eta, method):
    run_cfg = default_run_config(model=model, nx=nx, ny=ny, eta_target=eta)
    run_cfg.progress_interval = PROGRESS_INTERVAL
    if OVERRIDE_T_END:
        run_cfg.t_end = T_END
    if OVERRIDE_DT_INIT:
        run_cfg.dt_init = DT_INIT
    
    run_cfg.dt_growth_limit = DT_GROWTH_LIMIT
    run_cfg.e_floor = E_FLOOR
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    run_cfg.checkpoint_interval = CHECKPOINT_INTERVAL
    run_cfg.resume_from_checkpoint = RESUME_FROM_CHECKPOINT
    run_cfg.checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"{model}_eta{eta}_grid{nx}_method{method}.npz"
    )
    return run_cfg


def make_solver_config(method):
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
    )


def print_case_summary(method, model, eta, nx, ny, result, solver_cfg):
    avg_linear = safe_average(result["linear_iters_history"])
    avg_nonlinear = safe_average(result["nonlinear_iters_history"])
    avg_lin_per_nonlin = (
        avg_linear / avg_nonlinear
        if np.isfinite(avg_linear) and np.isfinite(avg_nonlinear) and avg_nonlinear > 0
        else float("nan")
    )
    max_linear = (
        max(result["linear_iters_history"])
        if len(result["linear_iters_history"]) > 0
        else float("nan")
    )
    max_nonlinear = (
        max(result["nonlinear_iters_history"])
        if len(result["nonlinear_iters_history"]) > 0
        else float("nan")
    )
    max_eta = (
        max(result["eta_history"]) if len(result["eta_history"]) > 0 else float("nan")
    )
    last_dt = (
        result["dt_history"][-1] if len(result["dt_history"]) > 0 else float("nan")
    )
    final_res = (
        result["residual_history"][-1]
        if len(result["residual_history"]) > 0
        else float("nan")
    )

    print("=" * 60)
    print(f"method           = {method}")
    print(f"model            = {model}")
    print(f"eta_target       = {eta}")
    print(f"grid             = {nx} x {ny}")
    print(f"mg smoother      = {solver_cfg.mg_smoother}")
    print(f"mg pre smooths   = {solver_cfg.mg_pre_smooths}")
    print(f"mg post smooths  = {solver_cfg.mg_post_smooths}")
    print(f"jfnk eps mode    = {solver_cfg.jfnk_eps_mode}")
    print(f"damping norm     = {solver_cfg.damping_norm}")
    print(f"gmres restart    = {solver_cfg.gmres_restart}")
    print("-" * 60)
    print(f"converged        = {result['converged']}")
    print(f"t_final          = {result['t_final']}")
    print(f"num_steps        = {len(result['time_history'])}")
    print(f"avg linear iters = {avg_linear}")
    print(f"avg nonlinear    = {avg_nonlinear}")
    print(f"avg lin/nonlin  = {avg_lin_per_nonlin}")
    print(f"max eta measured = {max_eta}")
    print(f"last dt          = {last_dt}")
    print(f"final residual   = {final_res}")
    print(f"max linear iters = {max_linear}")
    print(f"max nonlinear    = {max_nonlinear}")
    print(f"failed_step      = {result['failed_step']}")
    print(f"failure_reason   = {result['failure_reason']}")
    print("=" * 60)
    print()


def summarize_result(result):
    avg_linear = safe_average(result["linear_iters_history"])
    avg_nonlinear = safe_average(result["nonlinear_iters_history"])
    avg_lin_per_nonlin = (
        avg_linear / avg_nonlinear
        if np.isfinite(avg_linear) and np.isfinite(avg_nonlinear) and avg_nonlinear > 0
        else float("nan")
    )
    max_linear = (
        max(result["linear_iters_history"])
        if len(result["linear_iters_history"]) > 0
        else float("nan")
    )
    max_nonlinear = (
        max(result["nonlinear_iters_history"])
        if len(result["nonlinear_iters_history"]) > 0
        else float("nan")
    )
    max_eta = (
        max(result["eta_history"]) if len(result["eta_history"]) > 0 else float("nan")
    )
    last_dt = (
        result["dt_history"][-1] if len(result["dt_history"]) > 0 else float("nan")
    )
    final_res = (
        result["residual_history"][-1]
        if len(result["residual_history"]) > 0
        else float("nan")
    )
    return {
        "num_steps": len(result["time_history"]),
        "avg_linear": avg_linear,
        "avg_nonlinear": avg_nonlinear,
        "avg_lin_per_nonlin": avg_lin_per_nonlin,
        "max_linear": max_linear,
        "max_nonlinear": max_nonlinear,
        "max_eta": max_eta,
        "last_dt": last_dt,
        "final_residual": final_res,
    }


def append_result_csv(method, model, eta, nx, ny, result, solver_cfg, run_cfg, elapsed):
    os.makedirs(os.path.dirname(RESULT_CSV_PATH), exist_ok=True)
    file_exists = os.path.exists(RESULT_CSV_PATH)
    summary = summarize_result(result)

    fieldnames = [
        "timestamp",
        "method",
        "model",
        "eta_target",
        "nx",
        "ny",
        "t_end",
        "t_final",
        "converged",
        "num_steps",
        "avg_linear",
        "avg_nonlinear",
        "avg_lin_per_nonlin",
        "max_linear",
        "max_nonlinear",
        "max_eta",
        "last_dt",
        "final_residual",
        "elapsed_seconds",
        "nonlinear_tol",
        "linear_tol_factor",
        "max_nonlinear_iters",
        "max_linear_iters",
        "gmres_restart",
        "use_mg_preconditioner",
        "mg_smoother",
        "mg_pre_smooths",
        "mg_post_smooths",
        "jfnk_eps_mode",
        "damping_norm",
        "dt_growth_limit",
        "e_floor",
        "resume_from_checkpoint",
        "checkpoint_path",
        "failed_step",
        "failure_reason",
    ]

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": method,
        "model": model,
        "eta_target": eta,
        "nx": nx,
        "ny": ny,
        "t_end": run_cfg.t_end,
        "t_final": result["t_final"],
        "converged": result["converged"],
        "num_steps": summary["num_steps"],
        "avg_linear": summary["avg_linear"],
        "avg_nonlinear": summary["avg_nonlinear"],
        "avg_lin_per_nonlin": summary["avg_lin_per_nonlin"],
        "max_linear": summary["max_linear"],
        "max_nonlinear": summary["max_nonlinear"],
        "max_eta": summary["max_eta"],
        "last_dt": summary["last_dt"],
        "final_residual": summary["final_residual"],
        "elapsed_seconds": elapsed,
        "nonlinear_tol": solver_cfg.nonlinear_tol,
        "linear_tol_factor": solver_cfg.linear_tol_factor,
        "max_nonlinear_iters": solver_cfg.max_nonlinear_iters,
        "max_linear_iters": solver_cfg.max_linear_iters,
        "gmres_restart": solver_cfg.gmres_restart,
        "use_mg_preconditioner": solver_cfg.use_multigrid_preconditioner,
        "mg_smoother": solver_cfg.mg_smoother,
        "mg_pre_smooths": solver_cfg.mg_pre_smooths,
        "mg_post_smooths": solver_cfg.mg_post_smooths,
        "jfnk_eps_mode": solver_cfg.jfnk_eps_mode,
        "damping_norm": solver_cfg.damping_norm,
        "dt_growth_limit": run_cfg.dt_growth_limit,
        "e_floor": run_cfg.e_floor,
        "resume_from_checkpoint": run_cfg.resume_from_checkpoint,
        "checkpoint_path": run_cfg.checkpoint_path,
        "failed_step": result["failed_step"],
        "failure_reason": result["failure_reason"],
    }

    with open(RESULT_CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[result-log] appended {RESULT_CSV_PATH}")


def run_one_case(method, model, eta, nx, ny):
    start = time.perf_counter()

    run_cfg = make_run_config(model=model, nx=nx, ny=ny, eta=eta, method=method)
    problem = build_problem(run_cfg)
    solver_cfg = make_solver_config(method=method)
    result = run_simulation(problem, run_cfg, solver_cfg)

    elapsed = time.perf_counter() - start
    print_case_summary(method, model, eta, nx, ny, result, solver_cfg)
    print(f"elapsed seconds  = {elapsed:.3f}")
    append_result_csv(method, model, eta, nx, ny, result, solver_cfg, run_cfg, elapsed)
    print()

    return result


def run_batch():
    all_results = []
    for method in BATCH_METHODS:
        for model in BATCH_MODELS:
            for eta in BATCH_ETAS:
                for g in BATCH_GRIDS:
                    result = run_one_case(method, model, eta, g, g)
                    summary = summarize_result(result)
                    all_results.append(
                        {
                            "method": method,
                            "model": model,
                            "eta": eta,
                            "grid": g,
                            "smoother": MG_SMOOTHER,
                            "jfnk_eps_mode": JFNK_EPS_MODE,
                            "damping_norm": DAMPING_NORM,
                            "converged": result["converged"],
                            "num_steps": summary["num_steps"],
                            "avg_linear": (
                                summary["avg_linear"]
                                if result["converged"]
                                else float("nan")
                            ),
                            "avg_nonlinear": (
                                summary["avg_nonlinear"]
                                if result["converged"]
                                else float("nan")
                            ),
                            "failure_reason": result["failure_reason"],
                        }
                    )

    print("\n" + "#" * 80)
    print("批量测试汇总")
    print("#" * 80)

    for row in all_results:
        if row["converged"]:
            status = "OK"
            avg_lin_str = f"{row['avg_linear']:<10.4f}"
            avg_nonlin_str = f"{row['avg_nonlinear']:<10.4f}"
        else:
            status = "FAIL"
            avg_lin_str = f"{'--':<10}"
            avg_nonlin_str = f"{'--':<10}"

        print(
            f"[{status}] "
            f"method={row['method']:<6} "
            f"model={row['model']:<2} "
            f"eta={row['eta']:<4} "
            f"grid={row['grid']}x{row['grid']:<3} "
            f"smoother={row['smoother']:<6} "
            f"eps={row['jfnk_eps_mode']:<10} "
            f"damp={row['damping_norm']:<5} "
            f"steps={row['num_steps']:<4} "
            f"avg_lin={avg_lin_str} "
            f"avg_nonlin={avg_nonlin_str} "
            f"reason={row['failure_reason']}"
        )


if __name__ == "__main__":
    if MODE == "single":
        run_one_case(METHOD, MODEL, ETA, NX, NY)
    elif MODE == "batch":
        run_batch()
    else:
        raise ValueError("MODE 必须是 'single' 或 'batch'")
