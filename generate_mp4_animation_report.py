"""Build an HTML report for paper-time MP4 diffusion animations.

This script only organizes videos already generated in output_mp4.  The report
is intended for the paper-reproduction diffusion process at the paper/default
final times, not for long-time truth-data generation.  For long-time diffusion
or deep-learning truth data, use generate_full_diffusion_animation.py instead.

Usage:
    python generate_mp4_animation_report.py

Outputs:
    output/mp4_animation_report.html
    output/mp4_animation_report.csv
"""

from __future__ import annotations

import csv
import html
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


MP4_DIR = Path("output_mp4")
CONTACT_SHEET_DIR = Path("output_mp4_contact_sheets")
OUTPUT_DIR = Path("output")
HTML_PATH = OUTPUT_DIR / "mp4_animation_report.html"
CSV_PATH = OUTPUT_DIR / "mp4_animation_report.csv"

FILENAME_RE = re.compile(
    r"^(M[123])_eta([0-9]+p[0-9]+)_grid(\d+)_method([A-Za-z0-9_]+)_([A-Za-z0-9_]+)\.mp4$",
    re.IGNORECASE,
)

MODEL_NOTES = {
    "M1": "M1：相对温和的非线性扩散测试。",
    "M2": "M2：更强非线性的扩散测试，论文默认终止时间很短。",
    "M3": "M3：带通量限制的非线性扩散测试。",
}


@dataclass(frozen=True)
class VideoEntry:
    path: Path
    contact_sheet_path: Path | None
    model: str
    eta: float
    grid: int
    method: str
    field: str
    expected_t_end: float
    size_bytes: int


def parse_eta(text: str) -> float:
    return float(text.replace("p", "."))


def expected_t_end(model: str, eta: float) -> float:
    """Paper-inspired defaults used by config.default_run_config."""
    if model == "M1":
        return 5.0
    if model == "M2":
        return 0.005
    if model == "M3":
        return 0.1 if eta <= 0.1000001 else 0.5
    raise ValueError(f"Unknown model: {model}")


def relpath(path: Path, start: Path = OUTPUT_DIR) -> str:
    return Path(os.path.relpath(path.resolve(), start.resolve())).as_posix()


def html_attr(value: object) -> str:
    return html.escape(str(value), quote=True)


def html_text(value: object) -> str:
    return html.escape(str(value), quote=False)


def fmt_float(value: float) -> str:
    return f"{value:.6g}"


def fmt_eta(value: float) -> str:
    return f"{value:.2f}"


def fmt_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def collect_videos() -> list[VideoEntry]:
    entries: list[VideoEntry] = []
    if not MP4_DIR.exists():
        return entries

    for path in sorted(MP4_DIR.glob("*.mp4")):
        match = FILENAME_RE.match(path.name)
        if not match:
            print(f"[warning] skipped unrecognized MP4 name: {path}")
            continue

        model = match.group(1).upper()
        eta = parse_eta(match.group(2))
        grid = int(match.group(3))
        method = match.group(4).lower()
        field = match.group(5)
        sheet = CONTACT_SHEET_DIR / f"{path.stem}_contact_sheet.png"
        entries.append(
            VideoEntry(
                path=path,
                contact_sheet_path=sheet if sheet.exists() else None,
                model=model,
                eta=eta,
                grid=grid,
                method=method,
                field=field,
                expected_t_end=expected_t_end(model, eta),
                size_bytes=path.stat().st_size,
            )
        )

    return sorted(entries, key=lambda item: (item.model, item.eta, item.grid, item.method, item.field))


def write_csv(entries: list[VideoEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "eta",
                "grid",
                "method",
                "field",
                "paper_default_t_end",
                "mp4_path",
                "contact_sheet_path",
                "size_bytes",
            ],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "model": entry.model,
                    "eta": fmt_eta(entry.eta),
                    "grid": entry.grid,
                    "method": entry.method,
                    "field": entry.field,
                    "paper_default_t_end": fmt_float(entry.expected_t_end),
                    "mp4_path": str(entry.path),
                    "contact_sheet_path": str(entry.contact_sheet_path or ""),
                    "size_bytes": entry.size_bytes,
                }
            )


def summary_cards(entries: list[VideoEntry]) -> str:
    by_model = Counter(entry.model for entry in entries)
    by_eta = Counter(fmt_eta(entry.eta) for entry in entries)
    contact_count = sum(1 for entry in entries if entry.contact_sheet_path is not None)
    cards = [
        ("MP4 动画", str(len(entries))),
        ("关键帧拼图", str(contact_count)),
        ("模型", ", ".join(f"{model}: {by_model[model]}" for model in sorted(by_model))),
        ("eta", ", ".join(f"{eta}: {by_eta[eta]}" for eta in sorted(by_eta))),
    ]
    return "\n".join(
        f"""<div class="metric"><div class="metric-value">{html_text(value)}</div><div class="metric-label">{html_text(label)}</div></div>"""
        for label, value in cards
    )


def entry_table(entries: list[VideoEntry]) -> str:
    rows = []
    for entry in entries:
        rows.append(
            "<tr>"
            f"<td>{html_text(entry.model)}</td>"
            f"<td>{html_text(fmt_eta(entry.eta))}</td>"
            f"<td>{entry.grid} x {entry.grid}</td>"
            f"<td>{html_text(entry.method)}</td>"
            f"<td>{html_text(entry.field)}</td>"
            f"<td>{html_text(fmt_float(entry.expected_t_end))}</td>"
            f"<td>{html_text(fmt_size(entry.size_bytes))}</td>"
            f"<td>{'有' if entry.contact_sheet_path else '缺失'}</td>"
            "</tr>"
        )

    return f"""
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>eta</th>
          <th>Grid</th>
          <th>Method</th>
          <th>Field</th>
          <th>论文默认 t_end</th>
          <th>MP4 大小</th>
          <th>拼图</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """


def video_sections(entries: list[VideoEntry]) -> str:
    sections: list[str] = []
    current_group: tuple[str, float] | None = None

    for entry in entries:
        group = (entry.model, entry.eta)
        if group != current_group:
            current_group = group
            note = MODEL_NOTES.get(entry.model, entry.model)
            sections.append(
                f"""
                <section class="group">
                  <h2>{html_text(entry.model)}，eta = {html_text(fmt_eta(entry.eta))}</h2>
                  <p>{html_text(note)} 本组动画均为论文默认终止时间 t = {html_text(fmt_float(entry.expected_t_end))} 的扩散过程。</p>
                </section>
                """
            )

        mp4_src = html_attr(relpath(entry.path))
        mp4_label = html_text(entry.path.name)
        sheet_html = ""
        if entry.contact_sheet_path is not None:
            sheet_src = html_attr(relpath(entry.contact_sheet_path))
            sheet_html = f"""
            <figure class="sheet">
              <img src="{sheet_src}" alt="{mp4_label} contact sheet">
              <figcaption>0%, 20%, 40%, 60%, 80%, 100% 关键帧拼图</figcaption>
            </figure>
            """
        else:
            sheet_html = '<div class="missing">未找到对应关键帧拼图。</div>'

        sections.append(
            f"""
            <article class="case-card">
              <div class="case-head">
                <h3>{mp4_label}</h3>
                <div class="tags">
                  <span>{html_text(entry.model)}</span>
                  <span>eta {html_text(fmt_eta(entry.eta))}</span>
                  <span>{entry.grid} x {entry.grid}</span>
                  <span>{html_text(entry.method)}</span>
                  <span>t_end {html_text(fmt_float(entry.expected_t_end))}</span>
                </div>
              </div>
              <div class="media-grid">
                <figure class="video-box">
                  <video controls muted preload="metadata" src="{mp4_src}"></video>
                  <figcaption><a href="{mp4_src}">打开 MP4</a></figcaption>
                </figure>
                {sheet_html}
              </div>
            </article>
            """
        )

    return "\n".join(sections)


def write_html(entries: list[VideoEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_link = html_attr(relpath(CSV_PATH))

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>论文复现时间扩散过程动画报告</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #172033;
      --muted: #667085;
      --line: #d9dee8;
      --accent: #1f6feb;
      --accent-soft: #e8f1ff;
      --warn: #8a5a00;
      --warn-bg: #fff6df;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      line-height: 1.55;
    }}
    main {{
      width: min(1180px, calc(100vw - 36px));
      margin: 0 auto;
      padding: 30px 0 48px;
    }}
    header {{
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 22px 0 6px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0;
      font-size: 17px;
      letter-spacing: 0;
      word-break: break-word;
    }}
    p {{ margin: 7px 0; }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{ text-decoration: underline; }}
    .lead {{
      max-width: 900px;
      color: var(--muted);
      font-size: 16px;
    }}
    .notice {{
      margin: 18px 0;
      padding: 14px 16px;
      border: 1px solid #f0d38a;
      background: var(--warn-bg);
      color: var(--warn);
      border-radius: 8px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric-value {{
      font-size: 22px;
      font-weight: 700;
    }}
    .metric-label {{
      margin-top: 3px;
      color: var(--muted);
      font-size: 13px;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 18px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #f0f3f8;
      color: #39455f;
      font-size: 13px;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .group {{
      margin-top: 26px;
      padding-top: 4px;
    }}
    .case-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin: 14px 0;
    }}
    .case-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 12px;
    }}
    .tags {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 6px;
      min-width: 280px;
    }}
    .tags span {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 3px 8px;
      border-radius: 8px;
      background: var(--accent-soft);
      color: #184e9f;
      font-size: 12px;
      white-space: nowrap;
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(260px, 1fr);
      gap: 14px;
      align-items: start;
    }}
    figure {{
      margin: 0;
    }}
    figcaption {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    video, img {{
      display: block;
      width: 100%;
      max-width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #111827;
    }}
    .sheet img {{
      background: #ffffff;
    }}
    .missing {{
      display: grid;
      min-height: 220px;
      place-items: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
    }}
    footer {{
      margin-top: 28px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 860px) {{
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .case-head {{ display: block; }}
      .tags {{
        justify-content: flex-start;
        min-width: 0;
        margin-top: 10px;
      }}
      .media-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>论文复现时间扩散过程动画报告</h1>
      <p class="lead">
        本报告整理的是 <code>output_mp4</code> 中已经生成的 MP4 文件，用于展示论文默认终止时间下的扩散过程。
        它不是长时间扩散或深度学习真值数据报告；长时间过程请使用 <code>generate_full_diffusion_animation.py</code> 生成的结果。
      </p>
      <p class="lead">生成时间：{html_text(generated_at)}；索引 CSV：<a href="{csv_link}">mp4_animation_report.csv</a></p>
    </header>

    <div class="notice">
      说明：这里的“论文默认 t_end”是物理时间终止点，MP4 的播放秒数只由帧数和帧率决定，不等于物理时间。
      默认终止时间为 M1: 5.0，M2: 0.005，M3: eta=0.10 时 0.1、eta=0.50 时 0.5。
    </div>

    <section class="metrics">
      {summary_cards(entries)}
    </section>

    <section>
      <h2>文件总览</h2>
      <div class="table-wrap">
        {entry_table(entries)}
      </div>
    </section>

    <section>
      <h2>动画与关键帧</h2>
      {video_sections(entries)}
    </section>

    <footer>
      来源目录：<code>{html_text(str(MP4_DIR))}</code>；关键帧拼图目录：<code>{html_text(str(CONTACT_SHEET_DIR))}</code>。
    </footer>
  </main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    entries = collect_videos()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(entries, CSV_PATH)
    write_html(entries, HTML_PATH)

    print("=" * 60)
    print("MP4 animation report")
    print(f"found MP4 files      = {len(entries)}")
    print(f"with contact sheets  = {sum(1 for item in entries if item.contact_sheet_path is not None)}")
    print(f"HTML report          = {HTML_PATH}")
    print(f"CSV index            = {CSV_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
