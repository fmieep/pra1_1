"""Create ChatGPT-friendly contact sheets from MP4 animation files.

This utility does not run or modify the PDE solver.  It only converts each
MP4 animation in output_mp4 into a compact 2x3 PNG key-frame sheet so the
animation process is easier to inspect or share with ChatGPT.

Usage:
    python make_mp4_contact_sheets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import cv2
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    print("Missing dependency:", exc)
    print("Please install dependencies with:")
    print("    pip install opencv-python pillow")
    raise SystemExit(1)


INPUT_DIR = Path("output_mp4")
OUTPUT_DIR = Path("output_mp4_contact_sheets")
PERCENTS = [0, 20, 40, 60, 80, 100]
GRID_ROWS = 2
GRID_COLS = 3
TARGET_TILE_WIDTH = 520
LABEL_HEIGHT = 92
TILE_PADDING = 12
BACKGROUND = (246, 247, 249)
LABEL_BACKGROUND = (255, 255, 255)
TEXT_COLOR = (25, 31, 41)
MUTED_TEXT_COLOR = (75, 85, 99)
WARNING_COLOR = (153, 27, 27)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def ellipsize(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int
) -> str:
    if text_width(draw, text, font) <= max_width:
        return text
    suffix = "..."
    available = max_width - text_width(draw, suffix, font)
    if available <= 0:
        return suffix
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if text_width(draw, text[:mid], font) <= available:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + suffix


def resize_frame(image: Image.Image, target_width: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid frame dimensions.")
    target_height = max(1, round(height * target_width / width))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def read_frame(
    cap: cv2.VideoCapture, frame_index: int, frame_count: int
) -> Image.Image | None:
    candidates = [frame_index]
    for offset in (1, -1, 2, -2, 3, -3):
        nearby = frame_index + offset
        if 0 <= nearby < frame_count:
            candidates.append(nearby)

    for candidate in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ok, frame = cap.read()
        if ok and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
    return None


def placeholder_tile(width: int, height: int, message: str) -> Image.Image:
    image = Image.new("RGB", (width, height), (252, 242, 242))
    draw = ImageDraw.Draw(image)
    font = load_font(18, bold=True)
    line = ellipsize(draw, message, font, width - 32)
    draw.text((16, max(16, height // 2 - 12)), line, fill=WARNING_COLOR, font=font)
    return image


def make_tile(
    frame: Image.Image | None,
    video_name: str,
    frame_index: int,
    percent: int,
    video_time: float,
    frame_height_hint: int | None = None,
) -> Image.Image:
    title_font = load_font(17, bold=True)
    meta_font = load_font(16)

    if frame is None:
        frame_height = frame_height_hint or round(TARGET_TILE_WIDTH * 9 / 16)
        body = placeholder_tile(TARGET_TILE_WIDTH, frame_height, "Frame unavailable")
    else:
        body = resize_frame(frame, TARGET_TILE_WIDTH)

    tile_width = body.width
    tile_height = LABEL_HEIGHT + body.height
    tile = Image.new("RGB", (tile_width, tile_height), LABEL_BACKGROUND)
    draw = ImageDraw.Draw(tile)

    text_x = 12
    max_text_width = tile_width - text_x * 2
    title = ellipsize(draw, video_name, title_font, max_text_width)
    meta_1 = f"frame index: {frame_index}    percent: {percent}%"
    meta_2 = f"video time: {video_time:.3f} s"

    draw.text((text_x, 9), title, fill=TEXT_COLOR, font=title_font)
    draw.text((text_x, 36), meta_1, fill=MUTED_TEXT_COLOR, font=meta_font)
    draw.text((text_x, 61), meta_2, fill=MUTED_TEXT_COLOR, font=meta_font)
    tile.paste(body, (0, LABEL_HEIGHT))
    return tile


def build_contact_sheet(video_path: Path, output_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            print(f"warning: cannot open MP4: {video_path}")
            return False

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            print(f"warning: cannot determine frame count: {video_path}")
            return False
        if fps <= 0:
            fps = 1.0

        last_frame = max(0, frame_count - 1)
        frame_indices = [
            min(last_frame, max(0, round(last_frame * percent / 100.0)))
            for percent in PERCENTS
        ]

        raw_frames: list[Image.Image | None] = []
        for frame_index in frame_indices:
            raw_frames.append(read_frame(cap, frame_index, frame_count))

        first_available = next((frame for frame in raw_frames if frame is not None), None)
        if first_available is None:
            print(f"warning: no readable frames in MP4: {video_path}")
            return False

        frame_height_hint = resize_frame(first_available, TARGET_TILE_WIDTH).height
        tiles = []
        for percent, frame_index, frame in zip(PERCENTS, frame_indices, raw_frames):
            video_time = frame_index / fps
            tiles.append(
                make_tile(
                    frame,
                    video_path.name,
                    frame_index,
                    percent,
                    video_time,
                    frame_height_hint=frame_height_hint,
                )
            )

        tile_width = max(tile.width for tile in tiles)
        tile_height = max(tile.height for tile in tiles)
        sheet_width = GRID_COLS * tile_width + (GRID_COLS + 1) * TILE_PADDING
        sheet_height = GRID_ROWS * tile_height + (GRID_ROWS + 1) * TILE_PADDING
        sheet = Image.new("RGB", (sheet_width, sheet_height), BACKGROUND)

        for index, tile in enumerate(tiles):
            row = index // GRID_COLS
            col = index % GRID_COLS
            x = TILE_PADDING + col * (tile_width + TILE_PADDING)
            y = TILE_PADDING + row * (tile_height + TILE_PADDING)
            if tile.size != (tile_width, tile_height):
                normalized = Image.new("RGB", (tile_width, tile_height), LABEL_BACKGROUND)
                normalized.paste(tile, (0, 0))
                tile = normalized
            sheet.paste(tile, (x, y))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path)
        return True
    except Exception as exc:
        print(f"warning: failed to process {video_path}: {exc}")
        return False
    finally:
        cap.release()


def main() -> int:
    mp4_files = sorted(INPUT_DIR.glob("*.mp4"))
    generated: list[Path] = []
    failures = 0

    if not INPUT_DIR.exists():
        print(f"warning: input folder does not exist: {INPUT_DIR}")

    for video_path in mp4_files:
        output_path = OUTPUT_DIR / f"{video_path.stem}_contact_sheet.png"
        print(f"processing: {video_path} -> {output_path}")
        if build_contact_sheet(video_path, output_path):
            generated.append(output_path)
        else:
            failures += 1

    print()
    print("Summary")
    print(f"MP4 found: {len(mp4_files)}")
    print(f"PNG generated: {len(generated)}")
    print(f"Failed: {failures}")
    if generated:
        print("Generated files:")
        for path in generated:
            print(f"  {path}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
