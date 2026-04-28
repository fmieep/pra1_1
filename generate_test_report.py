"""Run pytest and write Chinese Markdown/HTML test reports.

Usage:
    python generate_test_report.py

Outputs:
    output/test_report.md
    output/test_report.html
"""

from __future__ import annotations

import html
import os
import time
from dataclasses import dataclass

import pytest


@dataclass
class TestRecord:
    nodeid: str
    outcome: str
    duration: float
    message: str = ""


class MarkdownReportPlugin:
    def __init__(self) -> None:
        self.records: dict[str, TestRecord] = {}

    def pytest_runtest_logreport(self, report):  # noqa: D401 - pytest hook
        if report.when != "call":
            if report.failed:
                self.records[report.nodeid] = TestRecord(
                    nodeid=report.nodeid,
                    outcome="failed",
                    duration=getattr(report, "duration", 0.0),
                    message=str(report.longrepr),
                )
            return

        message = ""
        if report.failed:
            message = str(report.longrepr)
        elif report.skipped:
            message = str(report.longrepr)

        self.records[report.nodeid] = TestRecord(
            nodeid=report.nodeid,
            outcome=report.outcome,
            duration=report.duration,
            message=message,
        )


def _module_name(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def _test_name(nodeid: str) -> str:
    return nodeid.rsplit("::", 1)[-1]


MODULE_DESCRIPTIONS = {
    "tests/test_problem_physics.py": (
        "问题设置与物理模型",
        "检查网格、材料区域、边界条件、M1/M2/M3 的储能项、质量系数、扩散系数和 Wilson limiter。",
    ),
    "tests/test_discretization.py": (
        "有限体积离散",
        "检查 harmonic face diffusion、测试专用 Dirichlet 有限体积算子是否对有真解问题收敛，并检查 Wilson limiter 是否降低原始扩散系数。",
    ),
    "tests/test_linear_multigrid.py": (
        "线性求解器与多重网格",
        "检查 GMRES/CG 是否能解小型 SPD 系统；检查 Picard 线性算子的线性性；检查一次 V-cycle 是否降低残差；检查 MG 作为预条件器是否减少 GMRES 迭代。",
    ),
    "tests/test_methods_driver.py": (
        "Picard/JFNK/NK2 与时间步控制",
        "检查 midpoint 状态构造、Jacobian-free matvec、Picard 线性化、一个小网格 NK2 时间步收敛，以及 eta/dt 更新规则。",
    ),
    "tests/test_integration_experiments.py": (
        "驱动器与实验统计",
        "检查迭代平均值、标度指数辅助函数，以及一个极短时间积分是否正确记录线性/非线性迭代历史。",
    ),
}


TEST_DESCRIPTIONS = {
    "test_grid_material_and_boundary_setup": "网格、材料分区、初值和边界类型是否符合论文二维问题设置。",
    "test_physics_storage_and_mass_models": "M1/M2/M3 的储能量 Q(E) 和 Q'(E) 是否与模型定义一致。",
    "test_diffusion_models_and_wilson_limiter_are_monotone": "扩散系数随温度的幂次是否正确，Wilson limiter 是否确实限制扩散系数。",
    "test_harmonic_face_diffusion_for_jump_coefficients": "材料跳跃处 face diffusion 是否使用 harmonic mean。",
    "test_dirichlet_finite_volume_manufactured_solution_converges": "有限体积离散在有解析解的 Dirichlet 问题上是否随网格加密收敛。",
    "test_wilson_limiter_on_faces_reduces_raw_coefficients": "M3 face-based limiter 是否不会放大原始扩散系数。",
    "test_gmres_and_cg_solve_small_spd_system": "GMRES 和 CG 对小型线性系统是否给出正确解。",
    "test_picard_operator_is_linear": "冻结系数后的 Picard operator 是否满足线性性。",
    "test_multigrid_vcycle_reduces_picard_residual": "一次多重网格 V-cycle 是否能降低 Picard 线性系统残差。",
    "test_multigrid_preconditioning_reduces_gmres_iterations": "MG 预条件器是否能减少 GMRES 迭代数。",
    "test_midpoint_states_follow_temperature_midpoint_rule": "NK2 的 implicit midpoint 状态是否按论文温度中点规则构造。",
    "test_jacobian_free_matvec_matches_quadratic_directional_derivative": "JFNK 有限差分 matvec 是否能近似已知导数。",
    "test_picard_linearization_is_linear_and_shape_preserving": "Picard 线性化接口是否线性，且预条件器返回维度正确。",
    "test_nk2_residual_and_step_converge_on_tiny_grid": "小网格上单个 NK2 时间步是否能把非线性残差降到容差以下。",
    "test_eta_and_dt_update_rules": "eta 计算和时间步增长限制是否符合设计。",
    "test_iteration_summary_helpers": "实验表格中的平均迭代数和标度指数统计函数是否工作正常。",
    "test_short_driver_run_records_histories": "短时间积分是否正确记录时间、dt、线性迭代、非线性迭代和残差历史。",
}


def _outcome_zh(outcome: str) -> str:
    return {
        "passed": "通过",
        "failed": "失败",
        "skipped": "跳过",
    }.get(outcome, outcome)


def _outcome_class(outcome: str) -> str:
    return {
        "passed": "pass",
        "failed": "fail",
        "skipped": "skip",
    }.get(outcome, "other")


def _counts(records: list[TestRecord]) -> tuple[int, int, int, int]:
    total = len(records)
    passed = sum(r.outcome == "passed" for r in records)
    failed = sum(r.outcome == "failed" for r in records)
    skipped = sum(r.outcome == "skipped" for r in records)
    return total, passed, failed, skipped


def write_markdown(records: list[TestRecord], exit_code: int, elapsed: float, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    total, passed, failed, skipped = _counts(records)
    modules = sorted({_module_name(r.nodeid) for r in records})

    lines = [
        "# 中文测试报告",
        "",
        "## 总览",
        "",
        f"- 测试状态：{'全部通过' if exit_code == 0 else '存在失败'}",
        f"- 测试总数：{total}",
        f"- 通过：{passed}",
        f"- 失败：{failed}",
        f"- 跳过：{skipped}",
        f"- 总耗时：{elapsed:.2f} 秒",
        "",
        "## 模块覆盖说明",
        "",
        "| 模块 | 测试目的 | 通过 | 失败 | 跳过 | 耗时 (秒) |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for module in modules:
        rows = [r for r in records if _module_name(r.nodeid) == module]
        title, description = MODULE_DESCRIPTIONS.get(module, (module, "未登记说明。"))
        lines.append(
            f"| **{title}** (`{module}`) | {description} | "
            f"{sum(r.outcome == 'passed' for r in rows)} | "
            f"{sum(r.outcome == 'failed' for r in rows)} | "
            f"{sum(r.outcome == 'skipped' for r in rows)} | "
            f"{sum(r.duration for r in rows):.3f} |"
        )

    lines.extend(
        [
            "",
            "## 逐项测试说明",
            "",
            "| 测试项 | 所属模块 | 检查内容 | 结果 | 耗时 (秒) |",
            "|---|---|---|---|---:|",
        ]
    )
    for record in sorted(records, key=lambda r: r.nodeid):
        module = _module_name(record.nodeid)
        test_name = _test_name(record.nodeid)
        title, _ = MODULE_DESCRIPTIONS.get(module, (module, ""))
        description = TEST_DESCRIPTIONS.get(test_name, "未登记说明。")
        lines.append(
            f"| `{test_name}` | {title} | {description} | {_outcome_zh(record.outcome)} | {record.duration:.3f} |"
        )

    failures = [r for r in records if r.outcome == "failed"]
    if failures:
        lines.extend(["", "## 失败详情", ""])
        for record in failures:
            lines.extend(
                [
                    f"### `{record.nodeid}`",
                    "",
                    "```text",
                    record.message[-4000:],
                    "```",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "",
                "## 结论",
                "",
                "当前测试说明：基础物理模型、有限体积离散、线性求解器、多重网格预条件、Picard 线性化、JFNK/NK2 时间步和实验统计辅助函数都通过了独立验证。",
                "这些测试不能保证论文表格迭代数完全一致，但可以帮助定位代码错误：如果后续修改某个技术模块导致测试失败，就能更快知道问题出在哪一层。",
            ]
        )

    # UTF-8 with BOM helps Windows Markdown editors and Notepad detect Chinese text.
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines) + "\n")


def write_html(records: list[TestRecord], exit_code: int, elapsed: float, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    total, passed, failed, skipped = _counts(records)
    modules = sorted({_module_name(r.nodeid) for r in records})
    status_text = "全部通过" if exit_code == 0 else "存在失败"

    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    module_rows: list[str] = []
    for module in modules:
        rows = [r for r in records if _module_name(r.nodeid) == module]
        title, description = MODULE_DESCRIPTIONS.get(module, (module, "未登记说明。"))
        module_rows.append(
            "<tr>"
            f"<td><strong>{esc(title)}</strong><br><code>{esc(module)}</code></td>"
            f"<td>{esc(description)}</td>"
            f"<td class=\"num pass-text\">{sum(r.outcome == 'passed' for r in rows)}</td>"
            f"<td class=\"num fail-text\">{sum(r.outcome == 'failed' for r in rows)}</td>"
            f"<td class=\"num skip-text\">{sum(r.outcome == 'skipped' for r in rows)}</td>"
            f"<td class=\"num\">{sum(r.duration for r in rows):.3f}</td>"
            "</tr>"
        )

    test_rows: list[str] = []
    for record in sorted(records, key=lambda r: r.nodeid):
        module = _module_name(record.nodeid)
        test_name = _test_name(record.nodeid)
        title, _ = MODULE_DESCRIPTIONS.get(module, (module, ""))
        description = TEST_DESCRIPTIONS.get(test_name, "未登记说明。")
        outcome_class = _outcome_class(record.outcome)
        test_rows.append(
            "<tr>"
            f"<td><code>{esc(test_name)}</code></td>"
            f"<td>{esc(title)}</td>"
            f"<td>{esc(description)}</td>"
            f"<td><span class=\"badge {outcome_class}\">{esc(_outcome_zh(record.outcome))}</span></td>"
            f"<td class=\"num\">{record.duration:.3f}</td>"
            "</tr>"
        )

    failures = [r for r in records if r.outcome == "failed"]
    if failures:
        failure_blocks = []
        for record in failures:
            failure_blocks.append(
                "<section class=\"failure-block\">"
                f"<h3><code>{esc(record.nodeid)}</code></h3>"
                f"<pre>{esc(record.message[-8000:])}</pre>"
                "</section>"
            )
        failure_html = "\n".join(failure_blocks)
    else:
        failure_html = "<p class=\"muted\">没有失败测试。</p>"

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>中文测试报告</title>
  <style>
    body {{
      margin: 0;
      background: #f7f8fa;
      color: #1f2933;
      font-family: Arial, "Microsoft YaHei", "PingFang SC", sans-serif;
      font-size: 14px;
      line-height: 1.55;
    }}
    .container {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1, h2, h3 {{
      margin: 0;
      color: #111827;
      line-height: 1.25;
    }}
    h1 {{ font-size: 28px; margin-bottom: 8px; }}
    h2 {{ font-size: 20px; margin: 28px 0 12px; }}
    h3 {{ font-size: 15px; margin-bottom: 8px; }}
    .muted {{ color: #64748b; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .summary-card {{
      background: #ffffff;
      border: 1px solid #d9e0e8;
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .summary-card .label {{
      color: #64748b;
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .summary-card .value {{
      font-size: 22px;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid #d9e0e8;
      margin-top: 10px;
    }}
    th, td {{
      border: 1px solid #d9e0e8;
      padding: 9px 10px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #eef2f6;
      color: #334155;
      font-weight: 700;
    }}
    code {{
      font-family: Consolas, "Courier New", monospace;
      font-size: 13px;
      color: #334155;
    }}
    .num {{ text-align: right; white-space: nowrap; }}
    .badge {{
      display: inline-block;
      min-width: 44px;
      border-radius: 999px;
      padding: 2px 9px;
      text-align: center;
      font-weight: 700;
      font-size: 12px;
    }}
    .badge.pass {{ background: #dcfce7; color: #166534; }}
    .badge.fail {{ background: #fee2e2; color: #991b1b; }}
    .badge.skip {{ background: #fef3c7; color: #92400e; }}
    .pass-text {{ color: #166534; font-weight: 700; }}
    .fail-text {{ color: #991b1b; font-weight: 700; }}
    .skip-text {{ color: #92400e; font-weight: 700; }}
    .failure-block {{
      background: #ffffff;
      border: 1px solid #fecaca;
      border-radius: 8px;
      padding: 14px;
      margin-top: 12px;
    }}
    pre {{
      margin: 0;
      padding: 12px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      background: #111827;
      color: #f9fafb;
      border-radius: 6px;
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <main class="container">
    <h1>中文测试报告</h1>
    <p class="muted">本报告由 <code>generate_test_report.py</code> 根据当前 pytest 运行结果自动生成。</p>

    <h2>测试总览</h2>
    <div class="summary-grid">
      <div class="summary-card"><div class="label">测试状态</div><div class="value">{esc(status_text)}</div></div>
      <div class="summary-card"><div class="label">测试总数</div><div class="value">{total}</div></div>
      <div class="summary-card"><div class="label">通过数</div><div class="value pass-text">{passed}</div></div>
      <div class="summary-card"><div class="label">失败数</div><div class="value fail-text">{failed}</div></div>
      <div class="summary-card"><div class="label">跳过数</div><div class="value skip-text">{skipped}</div></div>
      <div class="summary-card"><div class="label">总耗时</div><div class="value">{elapsed:.2f}s</div></div>
    </div>

    <h2>模块覆盖说明</h2>
    <table>
      <thead>
        <tr><th>模块</th><th>测试目的</th><th>通过</th><th>失败</th><th>跳过</th><th>耗时 (秒)</th></tr>
      </thead>
      <tbody>
        {"".join(module_rows)}
      </tbody>
    </table>

    <h2>逐项测试说明</h2>
    <table>
      <thead>
        <tr><th>测试函数名</th><th>所属模块</th><th>检查内容</th><th>结果</th><th>耗时 (秒)</th></tr>
      </thead>
      <tbody>
        {"".join(test_rows)}
      </tbody>
    </table>

    <h2>失败详情</h2>
    {failure_html}
  </main>
</body>
</html>
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(document)


def main() -> int:
    plugin = MarkdownReportPlugin()
    start = time.perf_counter()
    exit_code = pytest.main(["tests", "-q"], plugins=[plugin])
    elapsed = time.perf_counter() - start

    records = list(plugin.records.values())
    md_path = os.path.join("output", "test_report.md")
    html_path = os.path.join("output", "test_report.html")
    write_markdown(records, int(exit_code), elapsed, md_path)
    write_html(records, int(exit_code), elapsed, html_path)
    print(f"wrote {md_path}")
    print(f"wrote {html_path}")
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
