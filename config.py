"""Configuration objects for the core reproduction path.

This file should remain lightweight: dataclasses + helper factories only.
"""
from dataclasses import dataclass


@dataclass
class SolverConfig:
    method: str  # "picard" or "nk2"
    nonlinear_tol: float = 1e-6
    linear_tol_factor: float = 1e-2
    max_nonlinear_iters: int = 50
    max_linear_iters: int = 100
    gmres_restart: int = 30
    rho_jfnk: float = 1e-8
    use_multigrid_preconditioner: bool = True
    # 新增：JFNK 差分扰动模式
    # "paper": eps = rho * (1 + ||E||)
    # "normalized": eps = rho * (1 + ||E||) / ||v||
    jfnk_eps_mode: str = "normalized"

    # 新增：阻尼范数模式
    # "l2": 维持你原来的全局 2-范数
    # "linf": 最大相对增量，更网格无关
    # "rms": 均方根相对增量
    damping_norm: str = "l2"

    # 新增：是否强制能量为正
    enforce_positive_update: bool = True
    mg_smoother: str = "jacobi"

@dataclass
class RunConfig:
    model: str  # "M1", "M2", "M3"
    nx: int
    ny: int
    t_end: float
    eta_target: float
    dt_init: float
    dt_growth_limit: float = 1.1
    e_floor: float = 1.0
     # 新增：论文复现时建议为 0；工程稳健运行可以设为 8
    max_step_retries: int = 0

def default_run_config(model: str, nx: int, ny: int, eta_target: float) -> RunConfig:
    """Return a paper-inspired default RunConfig."""
    if model == "M1":
        t_end = 5.0
    elif model == "M2":
        t_end = 0.005
    elif model == "M3":
        # 论文图中 M3, eta=0.10 通常展示到 0.1；
        # eta=0.50 展示到 0.5。这里先按这个设置。
        t_end = 0.1 if eta_target <= 0.1000001 else 0.5
    else:
        raise ValueError(f"Unknown model: {model}")

    return RunConfig(
        model=model,
        nx=nx,
        ny=ny,
        t_end=t_end,
        eta_target=eta_target,
        dt_init=1e-6,
        max_step_retries=0,
    )
