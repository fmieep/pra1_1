"""Generate paper-style comparison reports from checkpointed experiments.

Usage:
    python compare_paper_results.py

Outputs:
    output/paper_comparison.html
    output/paper_comparison.csv
"""

from __future__ import annotations

import csv
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("output")
HTML_PATH = OUTPUT_DIR / "paper_comparison.html"
CSV_PATH = OUTPUT_DIR / "paper_comparison.csv"

MODELS = ["M1", "M2", "M3"]
ETAS = [0.10, 0.50]
NK2_GRIDS = [32, 64, 128, 256]
PICARD_GRIDS = [32, 64, 128]


PAPER_NK2_LINEAR = {
    0.10: {
        "M1": {32: 2.57, 64: 3.02, 128: 3.40, 256: 3.39},
        "M2": {32: 7.72, 64: 9.01, 128: 10.4, 256: 11.0},
        "M3": {32: 2.88, 64: 3.97, 128: 5.70, 256: 7.74},
    },
    0.50: {
        "M1": {32: 3.35, 64: 3.96, 128: 4.59, 256: 5.34},
        "M2": {32: 14.3, 64: 15.2, 128: 16.2, 256: 17.5},
        "M3": {32: 4.71, 64: 6.61, 128: 9.37, 256: 12.8},
    },
}

PAPER_NK2_NONLINEAR = {
    0.10: {
        "M1": {32: 3.42, 64: 3.57, 128: 3.71, 256: 3.61},
        "M2": {32: 4.53, 64: 4.62, 128: 4.66, 256: 4.62},
        "M3": {32: 3.32, 64: 3.49, 128: 3.74, 256: 3.86},
    },
    0.50: {
        "M1": {32: 3.42, 64: 3.63, 128: 3.79, 256: 3.99},
        "M2": {32: 5.29, 64: 5.37, 128: 5.48, 256: 5.66},
        "M3": {32: 3.66, 64: 3.97, 128: 4.27, 256: 4.52},
    },
}

PAPER_NK2_LINEAR_EXPONENT = {
    0.10: {"M1": 1.0668, "M2": 1.087, "M3": 1.240},
    0.50: {"M1": 1.111, "M2": 1.048, "M3": 1.242},
}

PAPER_NK2_NONLINEAR_EXPONENT = {
    0.10: {"M1": 1.014, "M2": 1.005, "M3": 1.038},
    0.50: {"M1": 1.036, "M2": 1.016, "M3": 1.051},
}

PAPER_PICARD_LINEAR = {
    0.10: {"M3": {32: 3.26, 64: 4.09, 128: 7.06}},
    0.50: {"M3": {32: 5.43, 64: 9.75, 128: 22.7}},
}

PAPER_PICARD_NONLINEAR = {
    0.10: {"M3": {32: 4.26, 64: 5.69, 128: 8.05}},
    0.50: {"M3": {32: 6.43, 64: 9.49, 128: 14.7}},
}

PAPER_PICARD_LINEAR_EXPONENT = {
    0.10: {"M3": 1.278},
    0.50: {"M3": 1.516},
}

PAPER_PICARD_NONLINEAR_EXPONENT = {
    0.10: {"M3": 1.230},
    0.50: {"M3": 1.298},
}


@dataclass(frozen=True)
class ComparisonRow:
    method: str
    model: str
    eta: float
    grid: int
    paper_linear: float | None
    local_linear: float | None
    linear_abs_diff: float | None
    linear_rel_diff: float | None
    paper_nonlinear: float | None
    local_nonlinear: float | None
    nonlinear_abs_diff: float | None
    nonlinear_rel_diff: float | None
    num_steps: int | None
    t_final: float | None
    expected_t_end: float
    complete: bool
    checkpoint: str | None


def _safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _eta_key(value: str) -> float:
    return round(float(value), 2)


def _expected_t_end(model: str, eta: float) -> float:
    if model == "M1":
        return 5.0
    if model == "M2":
        return 0.005
    return eta


def _load_checkpoint_results() -> dict[tuple[str, str, float, int], dict]:
    pattern = re.compile(
        r"^(M[123])_eta([0-9.]+)_grid(\d+)_method(nk2|picard)\.npz$",
        re.IGNORECASE,
    )
    results: dict[tuple[str, str, float, int], dict] = {}
    if not CHECKPOINT_DIR.exists():
        return results

    for path in sorted(CHECKPOINT_DIR.glob("*.npz")):
        match = pattern.match(path.name)
        if not match:
            continue

        model = match.group(1).upper()
        eta = _eta_key(match.group(2))
        grid = int(match.group(3))
        method = match.group(4).lower()

        data = np.load(path, allow_pickle=True)
        linear = _safe_mean(np.asarray(data["linear_iters_history"], dtype=float))
        nonlinear = _safe_mean(np.asarray(data["nonlinear_iters_history"], dtype=float))
        results[(method, model, eta, grid)] = {
            "local_linear": linear,
            "local_nonlinear": nonlinear,
            "num_steps": int(len(data["time_history"])),
            "t_final": float(data["t"]),
            "checkpoint": str(path),
        }

    return results


def _diff(local: float | None, paper: float | None) -> tuple[float | None, float | None]:
    if paper is None or local is None or not math.isfinite(local):
        return None, None
    abs_diff = local - paper
    rel_diff = abs_diff / paper * 100.0
    return abs_diff, rel_diff


def _make_row(
    local: dict[tuple[str, str, float, int], dict],
    method: str,
    model: str,
    eta: float,
    grid: int,
    paper_linear: float | None,
    paper_nonlinear: float | None,
) -> ComparisonRow:
    payload = local.get((method, model, eta, grid), {})
    local_linear = payload.get("local_linear")
    local_nonlinear = payload.get("local_nonlinear")
    t_final = payload.get("t_final")
    expected_t_end = _expected_t_end(model, eta)
    complete = (
        t_final is not None
        and abs(float(t_final) - expected_t_end) <= max(1e-12, 1e-9 * expected_t_end)
    )
    linear_abs_diff, linear_rel_diff = _diff(local_linear, paper_linear)
    nonlinear_abs_diff, nonlinear_rel_diff = _diff(local_nonlinear, paper_nonlinear)
    return ComparisonRow(
        method=method,
        model=model,
        eta=eta,
        grid=grid,
        paper_linear=paper_linear,
        local_linear=local_linear,
        linear_abs_diff=linear_abs_diff,
        linear_rel_diff=linear_rel_diff,
        paper_nonlinear=paper_nonlinear,
        local_nonlinear=local_nonlinear,
        nonlinear_abs_diff=nonlinear_abs_diff,
        nonlinear_rel_diff=nonlinear_rel_diff,
        num_steps=payload.get("num_steps"),
        t_final=t_final,
        expected_t_end=expected_t_end,
        complete=complete,
        checkpoint=payload.get("checkpoint"),
    )


def build_rows() -> list[ComparisonRow]:
    local = _load_checkpoint_results()
    rows: list[ComparisonRow] = []

    for eta in ETAS:
        for model in MODELS:
            for grid in NK2_GRIDS:
                rows.append(
                    _make_row(
                        local,
                        "nk2",
                        model,
                        eta,
                        grid,
                        PAPER_NK2_LINEAR[eta][model][grid],
                        PAPER_NK2_NONLINEAR[eta][model][grid],
                    )
                )

    for eta in ETAS:
        for grid in PICARD_GRIDS:
            rows.append(
                _make_row(
                    local,
                    "picard",
                    "M3",
                    eta,
                    grid,
                    PAPER_PICARD_LINEAR[eta]["M3"][grid],
                    PAPER_PICARD_NONLINEAR[eta]["M3"][grid],
                )
            )

    return rows


def _fit_scaling_exponent(grids: list[int], values: list[float | None]) -> float | None:
    valid = [
        (grid, value)
        for grid, value in zip(grids, values)
        if value is not None and math.isfinite(value) and value > 0
    ]
    if len(valid) < 2:
        return None
    n_cells = np.array([grid * grid for grid, _ in valid], dtype=float)
    y = np.array([value for _, value in valid], dtype=float)
    return float(1.0 + np.polyfit(np.log(n_cells), np.log(y), 1)[0])


def _fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _err_class(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "missing"
    magnitude = abs(value)
    if magnitude <= 10.0:
        return "good"
    if magnitude <= 25.0:
        return "warn"
    return "bad"


def _err_label(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "缺失"
    if abs(value) <= 10.0:
        return "接近"
    if value > 0:
        return "偏高"
    return "偏低"


def _paper_exponents(method: str, eta: float, model: str) -> tuple[float | None, float | None]:
    if method == "nk2":
        return PAPER_NK2_LINEAR_EXPONENT[eta][model], PAPER_NK2_NONLINEAR_EXPONENT[eta][model]
    if method == "picard" and model == "M3":
        return PAPER_PICARD_LINEAR_EXPONENT[eta][model], PAPER_PICARD_NONLINEAR_EXPONENT[eta][model]
    return None, None


def write_csv(rows: list[ComparisonRow], path: Path) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fieldnames = [
        "method",
        "model",
        "eta",
        "grid",
        "paper_linear",
        "local_linear",
        "linear_abs_diff",
        "linear_rel_diff_percent",
        "paper_nonlinear",
        "local_nonlinear",
        "nonlinear_abs_diff",
        "nonlinear_rel_diff_percent",
        "num_steps",
        "t_final",
        "expected_t_end",
        "complete",
        "checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.method,
                    "model": row.model,
                    "eta": row.eta,
                    "grid": row.grid,
                    "paper_linear": row.paper_linear,
                    "local_linear": row.local_linear,
                    "linear_abs_diff": row.linear_abs_diff,
                    "linear_rel_diff_percent": row.linear_rel_diff,
                    "paper_nonlinear": row.paper_nonlinear,
                    "local_nonlinear": row.local_nonlinear,
                    "nonlinear_abs_diff": row.nonlinear_abs_diff,
                    "nonlinear_rel_diff_percent": row.nonlinear_rel_diff,
                    "num_steps": row.num_steps,
                    "t_final": row.t_final,
                    "expected_t_end": row.expected_t_end,
                    "complete": row.complete,
                    "checkpoint": row.checkpoint,
                }
            )


def _summary(rows: list[ComparisonRow]) -> dict[str, float | int]:
    linear_errors = [
        abs(r.linear_rel_diff)
        for r in rows
        if r.complete and r.linear_rel_diff is not None and math.isfinite(r.linear_rel_diff)
    ]
    nonlinear_errors = [
        abs(r.nonlinear_rel_diff)
        for r in rows
        if r.complete and r.nonlinear_rel_diff is not None and math.isfinite(r.nonlinear_rel_diff)
    ]
    return {
        "cases": len(rows),
        "available": sum(r.local_linear is not None and r.local_nonlinear is not None for r in rows),
        "complete": sum(r.complete for r in rows),
        "linear_mean_abs": float(np.mean(linear_errors)) if linear_errors else float("nan"),
        "nonlinear_mean_abs": float(np.mean(nonlinear_errors)) if nonlinear_errors else float("nan"),
        "linear_within_10": sum(e <= 10.0 for e in linear_errors),
        "nonlinear_within_10": sum(e <= 10.0 for e in nonlinear_errors),
    }


def _method_title(method: str) -> str:
    return "Newton-Krylov / NK2" if method == "nk2" else "Picard"


def _comparison_table(rows: list[ComparisonRow], method: str, eta: float, model: str) -> str:
    subset = [r for r in rows if r.method == method and r.eta == eta and r.model == model]
    complete_subset = [r for r in subset if r.complete]
    local_linear_exp = _fit_scaling_exponent(
        [r.grid for r in complete_subset],
        [r.local_linear for r in complete_subset],
    )
    local_nonlinear_exp = _fit_scaling_exponent(
        [r.grid for r in complete_subset],
        [r.local_nonlinear for r in complete_subset],
    )
    paper_linear_exp, paper_nonlinear_exp = _paper_exponents(method, eta, model)
    linear_exp_abs, linear_exp_rel = _diff(local_linear_exp, paper_linear_exp)
    nonlinear_exp_abs, nonlinear_exp_rel = _diff(local_nonlinear_exp, paper_nonlinear_exp)

    body_rows = []
    for row in subset:
        linear_class = _err_class(row.linear_rel_diff)
        nonlinear_class = _err_class(row.nonlinear_rel_diff)
        body_rows.append(
            "<tr>"
            f"<td>{row.grid} x {row.grid}</td>"
            f"<td class=\"{'good' if row.complete else 'missing'}\">{_escape('完整' if row.complete else '未完成/缺失')}</td>"
            f"<td>{_fmt(row.paper_linear)}</td>"
            f"<td>{_fmt(row.local_linear)}</td>"
            f"<td class=\"{linear_class}\">{_fmt(row.linear_abs_diff, 2)}</td>"
            f"<td class=\"{linear_class}\">{_fmt(row.linear_rel_diff, 1)}%</td>"
            f"<td class=\"{linear_class}\">{_escape(_err_label(row.linear_rel_diff))}</td>"
            f"<td>{_fmt(row.paper_nonlinear)}</td>"
            f"<td>{_fmt(row.local_nonlinear)}</td>"
            f"<td class=\"{nonlinear_class}\">{_fmt(row.nonlinear_abs_diff, 2)}</td>"
            f"<td class=\"{nonlinear_class}\">{_fmt(row.nonlinear_rel_diff, 1)}%</td>"
            f"<td class=\"{nonlinear_class}\">{_escape(_err_label(row.nonlinear_rel_diff))}</td>"
            f"<td>{_fmt(row.num_steps, 0)}</td>"
            "</tr>"
        )

    body_rows.append(
        "<tr class=\"scaling-row\">"
        "<td colspan=\"2\">Scaling exponent</td>"
        f"<td>{_fmt(paper_linear_exp, 4)}</td>"
        f"<td>{_fmt(local_linear_exp, 4)}</td>"
        f"<td class=\"{_err_class(linear_exp_rel)}\">{_fmt(linear_exp_abs, 4)}</td>"
        f"<td class=\"{_err_class(linear_exp_rel)}\">{_fmt(linear_exp_rel, 1)}%</td>"
        f"<td class=\"{_err_class(linear_exp_rel)}\">{_escape(_err_label(linear_exp_rel))}</td>"
        f"<td>{_fmt(paper_nonlinear_exp, 4)}</td>"
        f"<td>{_fmt(local_nonlinear_exp, 4)}</td>"
        f"<td class=\"{_err_class(nonlinear_exp_rel)}\">{_fmt(nonlinear_exp_abs, 4)}</td>"
        f"<td class=\"{_err_class(nonlinear_exp_rel)}\">{_fmt(nonlinear_exp_rel, 1)}%</td>"
        f"<td class=\"{_err_class(nonlinear_exp_rel)}\">{_escape(_err_label(nonlinear_exp_rel))}</td>"
        "<td>-</td>"
        "</tr>"
    )

    return f"""
    <section class="paper-block">
      <h3>{_escape(_method_title(method))}：{_escape(model)}, eta = {eta:.2f}</h3>
      <table class="booktabs">
        <thead>
          <tr>
            <th rowspan="2">网格</th>
            <th rowspan="2">状态</th>
            <th colspan="5">线性迭代</th>
            <th colspan="5">非线性迭代</th>
            <th rowspan="2">时间步数</th>
          </tr>
          <tr>
            <th>论文</th>
            <th>本代码</th>
            <th>差值</th>
            <th>相对误差</th>
            <th>判断</th>
            <th>论文</th>
            <th>本代码</th>
            <th>差值</th>
            <th>相对误差</th>
            <th>判断</th>
          </tr>
        </thead>
        <tbody>
          {''.join(body_rows)}
        </tbody>
      </table>
    </section>
    """


def _m3_method_overview(rows: list[ComparisonRow]) -> str:
    body = []
    for eta in ETAS:
        for method in ("nk2", "picard"):
            subset = [
                r
                for r in rows
                if r.method == method and r.model == "M3" and r.eta == eta and r.complete
            ]
            linear_exp = _fit_scaling_exponent([r.grid for r in subset], [r.local_linear for r in subset])
            nonlinear_exp = _fit_scaling_exponent([r.grid for r in subset], [r.local_nonlinear for r in subset])
            linear_avg = float(np.mean([r.local_linear for r in subset])) if subset else float("nan")
            nonlinear_avg = float(np.mean([r.local_nonlinear for r in subset])) if subset else float("nan")
            body.append(
                "<tr>"
                f"<td>{eta:.2f}</td>"
                f"<td>{_escape(_method_title(method))}</td>"
                f"<td>{len(subset)}</td>"
                f"<td>{_fmt(linear_avg)}</td>"
                f"<td>{_fmt(nonlinear_avg)}</td>"
                f"<td>{_fmt(linear_exp, 4)}</td>"
                f"<td>{_fmt(nonlinear_exp, 4)}</td>"
                "</tr>"
            )
    return f"""
    <section class="paper-block">
      <h2>M3 方法总览</h2>
      <table class="booktabs compact">
        <thead>
          <tr>
            <th>eta</th>
            <th>方法</th>
            <th>完整算例数</th>
            <th>平均线性迭代</th>
            <th>平均非线性迭代</th>
            <th>线性 scaling</th>
            <th>非线性 scaling</th>
          </tr>
        </thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </section>
    """


def write_html(rows: list[ComparisonRow], path: Path) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = _summary(rows)
    nk2_tables = "\n".join(
        _comparison_table(rows, "nk2", eta, model) for eta in ETAS for model in MODELS
    )
    picard_tables = "\n".join(_comparison_table(rows, "picard", eta, "M3") for eta in ETAS)
    m3_overview = _m3_method_overview(rows)

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>论文迭代结果对比</title>
  <style>
    body {{
      margin: 0;
      background: #f6f7f9;
      color: #1f2937;
      font-family: Arial, "Microsoft YaHei", "PingFang SC", sans-serif;
      font-size: 14px;
      line-height: 1.55;
    }}
    main {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 34px 24px 56px;
    }}
    h1, h2, h3 {{
      color: #111827;
      line-height: 1.25;
      margin: 0;
    }}
    h1 {{ font-size: 28px; margin-bottom: 8px; }}
    h2 {{ font-size: 20px; margin: 30px 0 12px; }}
    h3 {{ font-size: 16px; margin: 22px 0 8px; }}
    p {{ margin: 8px 0; }}
    code {{ font-family: Consolas, "Courier New", monospace; font-size: 13px; }}
    .muted {{ color: #64748b; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin: 20px 0 8px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #dde3ea;
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .label {{ color: #64748b; font-size: 13px; margin-bottom: 4px; }}
    .value {{ font-size: 22px; font-weight: 700; }}
    .paper-block {{ margin-top: 18px; overflow-x: auto; }}
    table.booktabs {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border-top: 2px solid #111827;
      border-bottom: 2px solid #111827;
      margin-top: 8px;
    }}
    .booktabs th, .booktabs td {{
      padding: 8px 9px;
      text-align: center;
      vertical-align: middle;
      white-space: nowrap;
    }}
    .booktabs thead tr:first-child th {{ border-bottom: 1px solid #111827; }}
    .booktabs thead tr:last-child th {{
      border-bottom: 1px solid #9ca3af;
      color: #374151;
      font-weight: 700;
    }}
    .booktabs tbody tr + tr td {{ border-top: 1px solid #e5e7eb; }}
    .booktabs .scaling-row td {{
      border-top: 1.5px solid #111827;
      background: #f9fafb;
      font-weight: 700;
    }}
    .compact th, .compact td {{ padding: 7px 10px; }}
    .good {{ color: #166534; font-weight: 700; }}
    .warn {{ color: #a16207; font-weight: 700; }}
    .bad {{ color: #991b1b; font-weight: 700; }}
    .missing {{ color: #6b7280; font-weight: 700; }}
    .note {{
      background: #ffffff;
      border-left: 4px solid #64748b;
      padding: 10px 14px;
      margin-top: 14px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>论文迭代结果对比</h1>
    <p class="muted">结果来自 <code>checkpoints/*.npz</code>。NK2 对照论文 Tables XIII-XVI；M3 Picard 对照论文 Tables IX-XII。</p>

    <div class="summary">
      <div class="card"><div class="label">对比项数</div><div class="value">{summary['cases']}</div></div>
      <div class="card"><div class="label">有本地结果</div><div class="value">{summary['available']}</div></div>
      <div class="card"><div class="label">完整算例</div><div class="value">{summary['complete']}</div></div>
      <div class="card"><div class="label">线性平均绝对误差</div><div class="value">{_fmt(summary['linear_mean_abs'], 1)}%</div></div>
      <div class="card"><div class="label">非线性平均绝对误差</div><div class="value">{_fmt(summary['nonlinear_mean_abs'], 1)}%</div></div>
      <div class="card"><div class="label">线性误差 <= 10%</div><div class="value">{summary['linear_within_10']}</div></div>
      <div class="card"><div class="label">非线性误差 <= 10%</div><div class="value">{summary['nonlinear_within_10']}</div></div>
    </div>

    <section class="note">
      <p>判断规则：相对误差绝对值不超过 10% 记为“接近”；正值表示本代码迭代数偏高，负值表示偏低。Scaling exponent 按论文的工作量指数计算，即拟合平均迭代数关于自由度 N 的幂次后再加 1。</p>
    </section>

    {m3_overview}

    <h2>Newton-Krylov / NK2 对比</h2>
    {nk2_tables}

    <h2>M3 Picard 对比</h2>
    {picard_tables}
  </main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> int:
    rows = build_rows()
    write_csv(rows, CSV_PATH)
    write_html(rows, HTML_PATH)
    print(f"wrote {CSV_PATH}")
    print(f"wrote {HTML_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
