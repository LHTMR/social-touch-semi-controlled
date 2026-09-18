#!/usr/bin/env python
"""Visual check of the contact-point interpolation.

For each block, randomly picks a few single touches (``single_touch_id``) and, for
each, plots the xy-projected contact point cloud during that touch:

    left  column = original (kinect-rate) contacts
    right column = interpolated contacts

so you can see the interpolation fill the temporal gaps. Points are coloured by
relative time within the touch (0 = start -> 1 = end): the original panel shows a
few discrete time bands, the interpolated panel a continuous sweep.

One figure per block -> plot/<block>_interp_check.png

Rendering uses Pillow directly (matplotlib's rasteriser segfaults in this conda
env — a native DLL conflict, unrelated to this code). The viridis colours come
from matplotlib's colormap *data* when available (a safe LUT lookup, no
rendering) with a built-in fallback, so there is no hard matplotlib dependency.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

ORIG_COL = "contact_points"
INTERP_COL = "contact_points_interpolated"
TOUCH_COL = "single_touch_id"
DEFAULT_TIME_COL = "time_nerve"
MAX_RENDER_POINTS = 15000  # subsample per panel so rendering stays fast/readable

# layout (px)
PANEL = 300
TITLE_H = 30
LEFT_M = 60
TOP_M = 52
COL_GAP = 74
ROW_GAP = 60
RIGHT_M = 104
BOT_M = 30


# ------------------------------------------------------------------- parsing
def parse_contact_points(cell) -> List[Tuple[float, float, float]]:
    """Parse a ``contact_points`` cell into a list of (x, y, z) tuples."""
    if not isinstance(cell, str):
        return []
    stripped = cell.strip()
    if not stripped or stripped == "[]":
        return []
    out = []
    for m in re.findall(r"\[([^\]]+)\]", stripped):
        parts = m.strip().lstrip("[").replace(",", " ").split()
        if len(parts) == 3:
            try:
                out.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    return out


def collect_xy(cells: Sequence[str], times: Sequence[float]
               ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return an (N, 2) xy array, an (N,) time array, and the number of non-empty rows."""
    xy: List[Tuple[float, float]] = []
    tt: List[float] = []
    n_rows = 0
    for cell, t in zip(cells, times):
        pts = parse_contact_points(cell)
        if pts:
            n_rows += 1
        for x, y, _z in pts:
            xy.append((x, y))
            tt.append(t)
    if not xy:
        return np.empty((0, 2)), np.empty(0), 0
    return np.asarray(xy), np.asarray(tt), n_rows


# --------------------------------------------------------------------- colour
def viridis_lut() -> np.ndarray:
    """(256, 3) uint8 viridis ramp — from matplotlib data if available."""
    try:
        import matplotlib
        cmap = matplotlib.colormaps["viridis"]
        return (np.asarray(cmap(np.linspace(0, 1, 256)))[:, :3] * 255).astype("uint8")
    except Exception:
        anchors = np.array([
            [68, 1, 84], [72, 40, 120], [62, 73, 137], [49, 104, 142], [38, 130, 142],
            [31, 158, 137], [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37],
        ], dtype=float)
        xs = np.linspace(0, 1, len(anchors))
        grid = np.linspace(0, 1, 256)
        return np.stack([np.interp(grid, xs, anchors[:, i]) for i in range(3)], 1).astype("uint8")


LUT = viridis_lut()


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


F_TITLE = _font(15)
F_PANEL = _font(12)
F_TICK = _font(10)


# ----------------------------------------------------------------- rendering
def _limits(pts_a: np.ndarray, pts_b: np.ndarray) -> Tuple[float, float, float, float]:
    stack = [p for p in (pts_a, pts_b) if len(p)]
    both = np.vstack(stack) if stack else np.zeros((1, 2))
    px = max((both[:, 0].max() - both[:, 0].min()) * 0.08, 1.0)
    py = max((both[:, 1].max() - both[:, 1].min()) * 0.08, 1.0)
    return (both[:, 0].min() - px, both[:, 0].max() + px,
            both[:, 1].min() - py, both[:, 1].max() + py)


def _to_px(xy: np.ndarray, x0: int, y0: int, lim) -> np.ndarray:
    xmin, xmax, ymin, ymax = lim
    scale = min(PANEL / (xmax - xmin), PANEL / (ymax - ymin))
    dw, dh = (xmax - xmin) * scale, (ymax - ymin) * scale
    ox, oy = x0 + (PANEL - dw) / 2, y0 + (PANEL - dh) / 2
    px = ox + (xy[:, 0] - xmin) * scale
    py = oy + (dh - (xy[:, 1] - ymin) * scale)  # flip y (image y grows down)
    return np.column_stack([px, py])


def _draw_panel(base: Image.Image, overlay: Image.Image, x0: int, y0: int,
                xy: np.ndarray, tt: np.ndarray, t0: float, t1: float, lim,
                title: str, rng: np.random.Generator) -> None:
    d = ImageDraw.Draw(base)
    d.rectangle([x0, y0, x0 + PANEL, y0 + PANEL], outline=(170, 170, 170), width=1)
    d.text((x0, y0 - TITLE_H + 3), title, font=F_PANEL, fill=(35, 35, 35))

    if len(xy):
        idx = np.arange(len(xy))
        if len(idx) > MAX_RENDER_POINTS:
            idx = rng.choice(idx, MAX_RENDER_POINTS, replace=False)
        pix = _to_px(xy[idx], x0, y0, lim)
        rel = (tt[idx] - t0) / (t1 - t0) if t1 > t0 else np.zeros(len(idx))
        cidx = np.clip((rel * 255).astype(int), 0, 255)
        od = ImageDraw.Draw(overlay)
        r = 2
        for (px, py), ci in zip(pix, cidx):
            rr, gg, bb = LUT[ci]
            od.ellipse([px - r, py - r, px + r, py + r], fill=(int(rr), int(gg), int(bb), 110))
    else:
        d.text((x0 + PANEL / 2 - 25, y0 + PANEL / 2), "no contacts", font=F_PANEL, fill=(150, 150, 150))

    # scale ticks (data mm at the panel corners)
    xmin, xmax, ymin, ymax = lim
    d.text((x0, y0 + PANEL + 3), f"{xmin:.0f}", font=F_TICK, fill=(120, 120, 120))
    d.text((x0 + PANEL - 26, y0 + PANEL + 3), f"{xmax:.0f}", font=F_TICK, fill=(120, 120, 120))
    d.text((x0 - 30, y0 + PANEL - 8), f"{ymin:.0f}", font=F_TICK, fill=(120, 120, 120))
    d.text((x0 - 30, y0 - 2), f"{ymax:.0f}", font=F_TICK, fill=(120, 120, 120))


def _draw_colorbar(base: Image.Image, x: int, y0: int, y1: int) -> None:
    d = ImageDraw.Draw(base)
    h = y1 - y0
    for i in range(h):
        rel = 1 - i / max(h - 1, 1)  # top = 1 (end), bottom = 0 (start)
        rr, gg, bb = LUT[int(np.clip(rel * 255, 0, 255))]
        d.line([(x, y0 + i), (x + 18, y0 + i)], fill=(int(rr), int(gg), int(bb)))
    d.rectangle([x, y0, x + 18, y1], outline=(150, 150, 150))
    d.text((x - 2, y0 - 16), "time", font=F_TICK, fill=(60, 60, 60))
    d.text((x + 22, y0 - 4), "1 end", font=F_TICK, fill=(90, 90, 90))
    d.text((x + 22, y1 - 8), "0 start", font=F_TICK, fill=(90, 90, 90))


def render_block(input_csv: Path, interp_csv: Path, n_touches: int,
                 rng: np.random.Generator, time_col: str) -> Image.Image:
    df = pd.read_csv(input_csv, usecols=[TOUCH_COL, ORIG_COL, time_col], low_memory=False)
    interp = pd.read_csv(interp_csv, usecols=[INTERP_COL], low_memory=False)
    if len(df) != len(interp):
        raise ValueError(f"Row mismatch: {input_csv.name}={len(df)} vs {interp_csv.name}={len(interp)}")

    times = df[time_col].to_numpy()
    touch = df[TOUCH_COL].to_numpy()
    valid = np.unique(touch[~np.isnan(touch)])
    valid = valid[valid > 0]
    if len(valid) == 0:
        raise ValueError(f"No valid {TOUCH_COL} in {input_csv.name}")
    picked = sorted(rng.choice(valid, size=min(n_touches, len(valid)), replace=False))

    n = len(picked)
    grid_w = 2 * PANEL + COL_GAP
    grid_h = n * (TITLE_H + PANEL) + (n - 1) * ROW_GAP
    W, H = LEFT_M + grid_w + RIGHT_M, TOP_M + grid_h + BOT_M

    base = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    overlay = Image.new("RGBA", (W, H), (255, 255, 255, 0))

    block = input_csv.stem.replace("_merged_data", "")
    ImageDraw.Draw(base).text((LEFT_M, 14),
                              f"{block}   —   contact xy   (left: original    right: interpolated)",
                              font=F_TITLE, fill=(20, 20, 20))

    for r, tid in enumerate(picked):
        rows = np.where(touch == tid)[0]
        r0, r1 = int(rows.min()), int(rows.max())
        sl = slice(r0, r1 + 1)
        lx, lt, lrows = collect_xy(df[ORIG_COL].iloc[sl].to_numpy(), times[sl])
        rx, rt, rrows = collect_xy(interp[INTERP_COL].iloc[sl].to_numpy(), times[sl])

        lim = _limits(lx, rx)
        t_all = np.concatenate([a for a in (lt, rt) if len(a)]) if (len(lt) or len(rt)) else np.array([0.0])
        t0, t1 = float(t_all.min()), float(t_all.max())

        y0 = TOP_M + r * (TITLE_H + PANEL + ROW_GAP) + TITLE_H
        xL = LEFT_M
        xR = LEFT_M + PANEL + COL_GAP
        _draw_panel(base, overlay, xL, y0, lx, lt, t0, t1, lim,
                    f"touch {int(tid)} — original: {lrows} frames, {len(lx)} pts", rng)
        _draw_panel(base, overlay, xR, y0, rx, rt, t0, t1, lim,
                    f"touch {int(tid)} — interpolated: {rrows} rows, {len(rx)} pts", rng)

    base = Image.alpha_composite(base, overlay)
    _draw_colorbar(base, LEFT_M + grid_w + 30, TOP_M + TITLE_H, TOP_M + grid_h)
    return base.convert("RGB")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/2022-06-14_ST13-01/blocks"))
    ap.add_argument("--interp-dir", type=Path, default=Path("output"))
    ap.add_argument("--plot-dir", type=Path, default=Path("plot"))
    ap.add_argument("--n-touches", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-col", default=DEFAULT_TIME_COL)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # "*_merged_data*.csv" so both the original blocks/ naming and the newer
    # blocks_rf_centered/ "..._merged_data_pca-xyz.csv" are picked up.
    inputs = sorted(args.data_dir.glob("*_merged_data*.csv"))
    if not inputs:
        logger.error("No *_merged_data*.csv in %s", args.data_dir)
        return 1

    for input_csv in inputs:
        interp_csv = args.interp_dir / f"{input_csv.stem}_contact-interpolated.csv"
        if not interp_csv.exists():
            logger.warning("Missing interpolated CSV, skipping: %s", interp_csv.name)
            continue
        img = render_block(input_csv, interp_csv, args.n_touches, rng, args.time_col)
        out = args.plot_dir / f"{input_csv.stem.replace('_merged_data', '')}_interp_check.png"
        img.save(out)
        logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
