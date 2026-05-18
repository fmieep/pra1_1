"""Generate a side-by-side paper-vs-code M3 iteration comparison report.

Each comparison row places the cropped paper figure on the left and the local
checkpoint-derived plot on the right.  The script does not rerun the PDE.
"""

from __future__ import annotations

import csv
import html
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


PDF_PATH = Path(r"f:\研究生教材\1-s2.0-S002199919996240X-main(1).pdf")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("output") / "m3_iteration_side_by_side"
HTML_PATH = OUTPUT_DIR / "m3_iteration_side_by_side.html"
SUMMARY_CSV_PATH = OUTPUT_DIR / "m3_iteration_summary.csv"
TMP_PAGE_DIR = OUTPUT_DIR / "_rendered_pages"

PLOT_SCRIPT = Path("generate_paper_fig9_m3_picard_iters.py")


@dataclass(frozen=True)
class Comparison:
    title: str
    paper_label: str
    paper_page: int
    paper_a_crop: tuple[int, int, int, int]
    paper_b_crop: tuple[int, int, int, int]
    paper_output: str
    method: str
    metric: str
    grids: tuple[int, ...]
    local_output: str
    note: str


COMPARISONS = [
    Comparison(
        title="Picard linear iterations on M3",
        paper_label="Paper Figure 8",
        paper_page=21,
        paper_a_crop=(360, 340, 1585, 1095),
        paper_b_crop=(360, 1120, 1585, 1830),
        paper_output="paper_fig8_picard_linear.png",
        method="picard",
        metric="linear",
        grids=(32, 64, 128),
        local_output="m3_picard_linear_iterations_paper_style.png",
        note="论文图 8 与本地 Picard linear_iters_history 对照。",
    ),
    Comparison(
        title="Picard nonlinear iterations on M3",
        paper_label="Paper Figure 9",
        paper_page=22,
        paper_a_crop=(360, 345, 1585, 1095),
        paper_b_crop=(360, 1110, 1585, 1810),
        paper_output="paper_fig9_picard_nonlinear.png",
        method="picard",
        metric="nonlinear",
        grids=(32, 64, 128),
        local_output="m3_picard_nonlinear_iterations_paper_style.png",
        note="论文图 9 与本地 Picard nonlinear_iters_history 对照。",
    ),
    Comparison(
        title="Newton-Krylov linear iterations on M3",
        paper_label="Paper Figure 10",
        paper_page=24,
        paper_a_crop=(360, 710, 1585, 1455),
        paper_b_crop=(360, 1515, 1585, 2195),
        paper_output="paper_fig10_nk2_linear.png",
        method="nk2",
        metric="linear",
        grids=(32, 64, 128, 256),
        local_output="m3_nk2_linear_iterations_paper_style.png",
        note="论文图 10 与本地 NK2 linear_iters_history 对照；本地图包含 256x256。",
    ),
    Comparison(
        title="Newton-Krylov nonlinear iterations on M3",
        paper_label="Paper Figure 11",
        paper_page=25,
        paper_a_crop=(360, 345, 1585, 1075),
        paper_b_crop=(360, 1105, 1585, 1840),
        paper_output="paper_fig11_nk2_nonlinear.png",
        method="nk2",
        metric="nonlinear",
        grids=(32, 64, 128, 256),
        local_output="m3_nk2_nonlinear_iterations_paper_style.png",
        note="论文图 11 与本地 NK2 nonlinear_iters_history 对照；本地图包含 256x256。",
    ),
]


def run_command(args: list[str]) -> None:
    subprocess.run(args, check=True)


def render_pdf_pages() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Missing paper PDF: {PDF_PATH}")

    TMP_PAGE_DIR.mkdir(parents=True, exist_ok=True)
    prefix = TMP_PAGE_DIR / "paper_page"
    run_command(
        [
            "pdftoppm",
            "-f",
            "21",
            "-l",
            "25",
            "-r",
            "220",
            "-png",
            str(PDF_PATH),
            str(prefix),
        ]
    )


def resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    if image.height == target_height:
        return image
    scale = target_height / image.height
    target_width = max(1, int(round(image.width * scale)))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def make_paper_composite(comp: Comparison) -> None:
    page_path = TMP_PAGE_DIR / f"paper_page-{comp.paper_page}.png"
    if not page_path.exists():
        raise FileNotFoundError(f"Rendered page missing: {page_path}")

    page = Image.open(page_path).convert("RGB")
    panel_a = page.crop(comp.paper_a_crop)
    panel_b = page.crop(comp.paper_b_crop)
    target_height = max(panel_a.height, panel_b.height)
    panel_a = resize_to_height(panel_a, target_height)
    panel_b = resize_to_height(panel_b, target_height)

    gap = 64
    margin = 24
    out = Image.new(
        "RGB",
        (panel_a.width + panel_b.width + gap + 2 * margin, target_height + 2 * margin),
        "white",
    )
    out.paste(panel_a, (margin, margin))
    out.paste(panel_b, (margin + panel_a.width + gap, margin))
    out.save(OUTPUT_DIR / comp.paper_output)


def generate_local_plots() -> None:
    if not PLOT_SCRIPT.exists():
        raise FileNotFoundError(f"Missing plot script: {PLOT_SCRIPT}")

    for comp in COMPARISONS:
        args = [
            sys.executable,
            "-B",
            str(PLOT_SCRIPT),
            "--checkpoint-dir",
            str(CHECKPOINT_DIR),
            "--output-dir",
            str(OUTPUT_DIR),
            "--method",
            comp.method,
            "--metric",
            comp.metric,
            "--grids",
            *[str(grid) for grid in comp.grids],
        ]
        run_command(args)


def checkpoint_path(method: str, eta: float, grid: int) -> Path:
    return CHECKPOINT_DIR / f"M3_eta{eta:g}_grid{grid}_method{method}.npz"


def summarize_iterations() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for comp in COMPARISONS:
        for eta in (0.10, 0.50):
            for grid in comp.grids:
                path = checkpoint_path(comp.method, eta, grid)
                if not path.exists():
                    rows.append(
                        {
                            "comparison": comp.title,
                            "method": comp.method,
                            "metric": comp.metric,
                            "eta": f"{eta:g}",
                            "grid": f"{grid}",
                            "steps": "missing",
                            "average_iterations": "missing",
                            "max_iterations": "missing",
                            "checkpoint": str(path),
                        }
                    )
                    continue

                with np.load(path) as data:
                    t_final = float(data["t"])
                    times = np.asarray(data["time_history"], dtype=float)
                    accepted = int(
                        np.count_nonzero(times <= t_final + max(1e-14, 1e-10 * max(abs(t_final), 1.0)))
                    )
                    history = np.asarray(
                        data[f"{comp.metric}_iters_history"], dtype=float
                    )[:accepted]

                rows.append(
                    {
                        "comparison": comp.title,
                        "method": comp.method,
                        "metric": comp.metric,
                        "eta": f"{eta:g}",
                        "grid": f"{grid}",
                        "steps": str(accepted),
                        "average_iterations": f"{float(np.mean(history)):.6g}",
                        "max_iterations": f"{float(np.max(history)):.6g}",
                        "checkpoint": str(path),
                    }
                )
    return rows


def write_summary_csv(rows: list[dict[str, str]]) -> None:
    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "comparison",
                "method",
                "metric",
                "eta",
                "grid",
                "steps",
                "average_iterations",
                "max_iterations",
                "checkpoint",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def render_html() -> str:
    cards = []
    for comp in COMPARISONS:
        cards.append(
            f"""
            <section class="comparison">
              <header>
                <h2>{html.escape(comp.title)}</h2>
                <p>{html.escape(comp.note)}</p>
              </header>
              <div class="paired">
                <article>
                  <h3>论文图：{html.escape(comp.paper_label)}</h3>
                  <img src="{html.escape(comp.paper_output)}" alt="{html.escape(comp.paper_label)}">
                </article>
                <article>
                  <h3>代码生成图</h3>
                  <img src="{html.escape(comp.local_output)}" alt="{html.escape(comp.title)} local plot">
                </article>
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>M3 Paper vs Code Iteration Comparison</title>
  <style>
    :root {{
      --ink: #1f2328;
      --muted: #636c76;
      --line: #d8dee4;
      --soft: #f6f8fa;
    }}
    body {{
      margin: 0;
      color: var(--ink);
      background: white;
      font-family: "Segoe UI", Arial, sans-serif;
    }}
    main {{
      max-width: 1420px;
      margin: 0 auto;
      padding: 30px 28px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    h2 {{
      margin: 0;
      font-size: 20px;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 15px;
    }}
    p {{
      margin: 6px 0 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .comparison {{
      margin-top: 28px;
      border-top: 1px solid var(--line);
      padding-top: 22px;
    }}
    .paired {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 18px;
      margin-top: 16px;
      align-items: start;
    }}
    article {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: var(--soft);
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      background: white;
      border: 1px solid var(--line);
    }}
    code {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 1px 5px;
    }}
    @media (max-width: 920px) {{
      main {{
        padding: 22px 14px 36px;
      }}
      .paired {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>M3 迭代次数：论文图与代码图对照</h1>
    <p>
      左侧为从论文 PDF 裁出的原图，右侧为读取本地 checkpoint 后生成的代码图。
      本报告不重跑 PDE；数据来自 <code>{html.escape(str(CHECKPOINT_DIR))}</code>。
      简要统计见 <code>{html.escape(SUMMARY_CSV_PATH.name)}</code>。
    </p>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_pdf_pages()
    for comp in COMPARISONS:
        make_paper_composite(comp)
    generate_local_plots()
    rows = summarize_iterations()
    write_summary_csv(rows)
    HTML_PATH.write_text(render_html(), encoding="utf-8")
    if TMP_PAGE_DIR.exists():
        shutil.rmtree(TMP_PAGE_DIR)
    print(f"saved html: {HTML_PATH}")
    print(f"saved csv: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
