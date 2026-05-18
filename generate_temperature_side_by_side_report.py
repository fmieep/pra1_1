"""Generate a side-by-side paper-vs-code temperature contour report.

The report compares the paper's Fig. 4 temperature contours with the local NK2
checkpoint-generated contours.  It does not rerun the PDE.
"""

from __future__ import annotations

import csv
import html
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from physics import energy_to_temperature


PDF_PATH = Path(r"f:\研究生教材\1-s2.0-S002199919996240X-main(1).pdf")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("output") / "temperature_comparison"
HTML_PATH = OUTPUT_DIR / "temperature_side_by_side.html"
SUMMARY_CSV_PATH = OUTPUT_DIR / "temperature_summary.csv"
TMP_PAGE_DIR = OUTPUT_DIR / "_rendered_pages"

LOCAL_FIGURE = "code_fig4_nk2_temperature_paper_exact.png"
PAPER_FIGURE = "paper_fig4_temperature.png"
PLOT_SCRIPT = Path("generate_paper_fig4_nk2.py")

CASES = [
    ("M1", 0.50, 256, 5.0),
    ("M2", 0.50, 128, 0.005),
]


def run_command(args: list[str]) -> None:
    subprocess.run(args, check=True)


def render_paper_page() -> Path:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Missing paper PDF: {PDF_PATH}")
    TMP_PAGE_DIR.mkdir(parents=True, exist_ok=True)
    prefix = TMP_PAGE_DIR / "paper_page"
    run_command(
        [
            "pdftoppm",
            "-f",
            "14",
            "-l",
            "14",
            "-r",
            "220",
            "-png",
            str(PDF_PATH),
            str(prefix),
        ]
    )
    page = TMP_PAGE_DIR / "paper_page-14.png"
    if not page.exists():
        raise FileNotFoundError(f"Rendered paper page missing: {page}")
    return page


def resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    if image.height == target_height:
        return image
    scale = target_height / image.height
    width = max(1, int(round(image.width * scale)))
    return image.resize((width, target_height), Image.Resampling.LANCZOS)


def make_paper_composite(page_path: Path) -> None:
    page = Image.open(page_path).convert("RGB")
    panel_a = page.crop((610, 250, 1510, 1160))
    panel_b = page.crop((610, 1190, 1510, 2130))

    target_height = max(panel_a.height, panel_b.height)
    panel_a = resize_to_height(panel_a, target_height)
    panel_b = resize_to_height(panel_b, target_height)

    gap = 54
    margin = 22
    out = Image.new(
        "RGB",
        (panel_a.width + panel_b.width + gap + 2 * margin, target_height + 2 * margin),
        "white",
    )
    out.paste(panel_a, (margin, margin))
    out.paste(panel_b, (margin + panel_a.width + gap, margin))
    out.save(OUTPUT_DIR / PAPER_FIGURE)


def generate_local_temperature_figure() -> None:
    if not PLOT_SCRIPT.exists():
        raise FileNotFoundError(f"Missing plot script: {PLOT_SCRIPT}")
    run_command(
        [
            sys.executable,
            "-B",
            str(PLOT_SCRIPT),
            "--checkpoint-dir",
            str(CHECKPOINT_DIR),
            "--no-run",
            "--level-mode",
            "paper-exact",
            "--m1-grid",
            "256",
            "--m2-grid",
            "128",
            "--output",
            str(OUTPUT_DIR / LOCAL_FIGURE),
        ]
    )


def checkpoint_path(model: str, eta: float, grid: int) -> Path:
    return CHECKPOINT_DIR / f"{model}_eta{eta:g}_grid{grid}_methodnk2.npz"


def write_summary_csv() -> None:
    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "eta",
                "grid",
                "expected_t_end",
                "checkpoint",
                "t_final",
                "steps",
                "T_min",
                "T_mean",
                "T_max",
            ]
        )
        for model, eta, grid, expected_t in CASES:
            path = checkpoint_path(model, eta, grid)
            with np.load(path) as data:
                E = data["E"]
                T = energy_to_temperature(E)
                t_final = float(data["t"])
                steps = len(data["time_history"]) if "time_history" in data.files else ""
            writer.writerow(
                [
                    model,
                    f"{eta:g}",
                    f"{grid}x{grid}",
                    f"{expected_t:g}",
                    str(path),
                    f"{t_final:.16g}",
                    steps,
                    f"{float(np.min(T)):.8g}",
                    f"{float(np.mean(T)):.8g}",
                    f"{float(np.max(T)):.8g}",
                ]
            )


def render_html() -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Temperature Contour Comparison</title>
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
      max-width: 1440px;
      margin: 0 auto;
      padding: 30px 28px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    p {{
      margin: 6px 0 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .paired {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 18px;
      margin-top: 22px;
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
    <h1>Figure 4 温度等值线：论文图与代码图对照</h1>
    <p>
      左侧为从论文 PDF 裁出的 Figure 4；右侧为读取本地 NK2 checkpoint 后生成的等温线图。
      本报告不重跑 PDE。代码图使用 M1 256x256、M2 128x128，并采用论文 Figure 4 的标注等温线刻度。
      温度统计见 <code>{html.escape(SUMMARY_CSV_PATH.name)}</code>。
    </p>
    <section class="paired">
      <article>
        <h2>论文 Figure 4</h2>
        <img src="{html.escape(PAPER_FIGURE)}" alt="Paper Figure 4 temperature contours">
      </article>
      <article>
        <h2>代码生成图</h2>
        <img src="{html.escape(LOCAL_FIGURE)}" alt="Local NK2 temperature contours">
      </article>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    page_path = render_paper_page()
    make_paper_composite(page_path)
    generate_local_temperature_figure()
    write_summary_csv()
    HTML_PATH.write_text(render_html(), encoding="utf-8")
    if TMP_PAGE_DIR.exists():
        shutil.rmtree(TMP_PAGE_DIR)
    print(f"saved html: {HTML_PATH}")
    print(f"saved csv: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
