"""Render the eval scorecard as a PNG thumbnail for the Upwork portfolio listing.

Pillow only -- no browser, no headless Chrome, matching this repo's "every dependency
is justifiable" stance.

    python -m eval.scorecard_png --scorecard eval/report/latest.json --out eval/report/scorecard.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1200
ROW_HEIGHT = 34
HEADER_HEIGHT = 90
PADDING = 24
COLUMNS = ("Metric", "Value", "95% CI", "n")
COLUMN_X = (PADDING, 520, 640, 860)

BG = "#ffffff"
FG = "#1a1a1a"
HEADER_BG = "#1a1a1a"
HEADER_FG = "#ffffff"
GRID = "#dddddd"
ACCENT = "#0a6b45"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Pillow's built-in bitmap font has no size control. Try a couple of common
    # system TrueType fonts and fall back to that default rather than hard-failing
    # on a machine that has neither installed -- the PNG is still readable, just
    # not as polished.
    candidates = (["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]) + \
        ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(scorecard: dict[str, Any], *, title: str | None = None) -> Image.Image:
    metrics = scorecard.get("metrics", [])
    height = HEADER_HEIGHT + ROW_HEIGHT * (len(metrics) + 1) + PADDING

    img = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(img)

    title_font = _font(28, bold=True)
    sub_font = _font(15)
    header_font = _font(15, bold=True)
    row_font = _font(15)

    label = title or scorecard.get("label", "eval scorecard")
    draw.text((PADDING, PADDING), label, font=title_font, fill=FG)
    n_cases = scorecard.get("n_cases", "?")
    kappa = (scorecard.get("meta") or {}).get("kappa")
    kappa_str = "pending" if kappa is None else f"{kappa:.3f}"
    draw.text((PADDING, PADDING + 38), f"n = {n_cases}    judge-vs-human kappa: {kappa_str}",
             font=sub_font, fill="#555555")

    y = HEADER_HEIGHT
    draw.rectangle((PADDING, y, WIDTH - PADDING, y + ROW_HEIGHT), fill=HEADER_BG)
    for x, col in zip(COLUMN_X, COLUMNS, strict=True):
        draw.text((x, y + 8), col, font=header_font, fill=HEADER_FG)
    y += ROW_HEIGHT

    for i, m in enumerate(metrics):
        if i % 2 == 1:
            draw.rectangle((PADDING, y, WIDTH - PADDING, y + ROW_HEIGHT), fill="#f7f7f7")
        draw.text((COLUMN_X[0], y + 8), str(m["name"]), font=row_font, fill=FG)
        draw.text((COLUMN_X[1], y + 8), f"{m['value']:.4f}", font=row_font, fill=ACCENT)
        draw.text((COLUMN_X[2], y + 8), f"[{m['ci'][0]:.4f}, {m['ci'][1]:.4f}]", font=row_font, fill="#555555")
        draw.text((COLUMN_X[3], y + 8), str(m["n"]), font=row_font, fill="#555555")
        y += ROW_HEIGHT

    draw.rectangle((PADDING, HEADER_HEIGHT, WIDTH - PADDING, y), outline=GRID, width=1)
    return img


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorecard", default="eval/report/latest.json")
    ap.add_argument("--out", default="eval/report/scorecard.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    scorecard = json.loads(Path(args.scorecard).read_text())
    img = render(scorecard, title=args.title)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"wrote {out} ({img.width}x{img.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
