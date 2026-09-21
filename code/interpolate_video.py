#!/usr/bin/env python
"""Frame-by-frame 3D videos of the contact point cloud, original vs interpolated.

For each block, randomly picks 5 single touches (``single_touch_id``) and writes
the same touch on two different timelines, one per sub-folder:

    <video-dir>/unaligned/<block>_touch<id>_1_original.mp4
    <video-dir>/unaligned/<block>_touch<id>_2_interpolated.mp4
    <video-dir>/aligned/  <block>_touch<id>_1_original.mp4
    <video-dir>/aligned/  <block>_touch<id>_2_interpolated.mp4

The file names are identical in both folders, so the same touch on the two
timelines is easy to line up. ``_1_original`` is the uninterpolated data at
kinect rate, held between samples so it plays on the dense timeline;
``_2_interpolated`` is every row. Each video shows the contact patch evolving
inside a fixed 3D grid box, with a faint accumulating trail so the swept path
builds up — the original stair-steps between the few kinect frames, the
interpolated glides continuously.

Timelines
---------
``unaligned`` samples rows evenly across the touch. Kinect anchors are 33-34
rows apart and the sampling step rarely divides that, so the held original steps
at uneven frame counts (e.g. 10, 11, 10, 11 ...) and anchors mostly fall between
frames. ``aligned`` instead samples a fixed number of frames inside each anchor
interval, so every anchor lands exactly on a frame and the original steps at a
perfectly regular cadence. It spans anchors[0]..anchors[-1] rather than the full
touch, since outside that range there is nothing measured to show. Both share
one view and one patch-colour axis, so all four videos are comparable.

Points are coloured by contact patch (finger), using the same clustering the
interpolation itself uses. On a whole-hand block that makes the fix checkable by
eye: each finger keeps its own colour and the gaps between them stay open,
instead of one colour bleeding across a gap into the next finger.
``--no-cluster-colors`` renders everything in one colour instead.

Rendered by hand-rolled 3D projection + OpenCV drawing/encoding (mp4v). No
matplotlib (its rasteriser segfaults in this conda env) and no OpenGL.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

# Same clustering as the tool, so the colours show what the morph actually saw.
from surface_morph import (
    DEFAULT_CLUSTER_EPS, DEFAULT_MIN_CLUSTER, cluster_points,
)

logger = logging.getLogger(__name__)

ORIG_COL = "contact_points"
INTERP_COL = "contact_points_interpolated"
TOUCH_COL = "single_touch_id"

SIZE = 400          # canvas W = H (px) — small on purpose
MARGIN = 46
FPS = 20
MAX_FRAMES = 120    # cap per video (subsample longer touches) -> <= 6 s, small files
ELEV, AZIM = 22.0, -58.0
CUR_COLOR = (0, 90, 230)      # BGR — current patch (orange-red)
TRAIL_COLOR = (205, 205, 205) # BGR — accumulated trail
BOX_COLOR = (150, 150, 150)
FLOOR_COLOR = (218, 218, 218)
# Per-patch colours (BGR). Index 0 is CUR_COLOR, so a single-finger block looks
# exactly as it did before clustering was introduced.
PATCH_COLORS = [CUR_COLOR, (200, 60, 0), (40, 160, 40), (170, 0, 160),
                (20, 190, 210), (130, 120, 0), (60, 60, 140)]


def parse_contact_points(cell) -> List[Tuple[float, float, float]]:
    if not isinstance(cell, str):
        return []
    s = cell.strip()
    if not s or s == "[]":
        return []
    out = []
    for m in re.findall(r"\[([^\]]+)\]", s):
        parts = m.strip().lstrip("[").replace(",", " ").split()
        if len(parts) == 3:
            try:
                out.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    return out


# --------------------------------------------------------------- 3D projection
def _rot(elev: float, azim: float) -> np.ndarray:
    # Rx(elev) @ Rz(azim), built analytically — this env's BLAS matmul segfaults.
    e, a = np.radians(elev), np.radians(azim)
    ce, se, ca, sa = np.cos(e), np.sin(e), np.cos(a), np.sin(a)
    return np.array([
        [ca,      -sa,      0.0],
        [ce * sa,  ce * ca, -se],
        [se * sa,  se * ca,  ce],
    ])


def _rotate(d: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate centered points ``d`` (N,3); return screen u, v and depth.

    Element-wise (no ``@`` / ``dot``) because numpy's BLAS matmul crashes here.
    Screen u = rotated x, v = rotated z (up), depth = rotated y.
    """
    u = d[:, 0] * R[0, 0] + d[:, 1] * R[0, 1] + d[:, 2] * R[0, 2]
    v = d[:, 0] * R[2, 0] + d[:, 1] * R[2, 1] + d[:, 2] * R[2, 2]
    depth = d[:, 0] * R[1, 0] + d[:, 1] * R[1, 1] + d[:, 2] * R[1, 2]
    return u, v, depth


def _make_view(bbox) -> dict:
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])
    R = _rot(ELEV, AZIM)
    corners = np.array([[x, y, z] for x in (xmin, xmax) for y in (ymin, ymax) for z in (zmin, zmax)])
    u, v, _ = _rotate(corners - center, R)
    scale = min((SIZE - 2 * MARGIN) / (u.max() - u.min() + 1e-9),
                (SIZE - 2 * MARGIN) / (v.max() - v.min() + 1e-9))
    return dict(R=R, center=center, scale=scale, umin=u.min(), vmin=v.min())


def _project(pts: np.ndarray, view: dict) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pts, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    u, v, depth = _rotate(p - view["center"], view["R"])
    x = MARGIN + (u - view["umin"]) * view["scale"]
    y = SIZE - (MARGIN + (v - view["vmin"]) * view["scale"])
    return np.column_stack([x, y]), depth


def _line3d(canvas, a, b, view, color, thick=1):
    seg, _ = _project([a, b], view)
    p0, p1 = seg[0], seg[1]
    cv2.line(canvas, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), color, thick, cv2.LINE_AA)


def _draw_grid(canvas, bbox, view):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    # floor grid on z = zmin
    for xi in np.linspace(xmin, xmax, 5):
        _line3d(canvas, [xi, ymin, zmin], [xi, ymax, zmin], view, FLOOR_COLOR)
    for yi in np.linspace(ymin, ymax, 5):
        _line3d(canvas, [xmin, yi, zmin], [xmax, yi, zmin], view, FLOOR_COLOR)
    # box edges
    corners = {(i, j, k): [x, y, z]
               for i, x in enumerate((xmin, xmax))
               for j, y in enumerate((ymin, ymax))
               for k, z in enumerate((zmin, zmax))}
    for (i, j, k), a in corners.items():
        for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            n = (i + di, j + dj, k + dk)
            if n in corners:
                _line3d(canvas, a, corners[n], view, BOX_COLOR)
    # axis hints from the min corner
    for lab, pt in (("x", [xmax, ymin, zmin]), ("y", [xmin, ymax, zmin]), ("z", [xmin, ymin, zmax])):
        p = _project(pt, view)[0][0]
        cv2.putText(canvas, lab, (int(p[0]) + 3, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (120, 120, 120), 1, cv2.LINE_AA)


def _draw_points(canvas, pts, view, color, radius):
    """Draw ``pts``; ``color`` is one BGR tuple, or one per point."""
    if not len(pts):
        return
    px, depth = _project(pts, view)
    per_point = not isinstance(color, tuple)
    for i in np.argsort(-depth):  # painter's order: far first
        x, y = px[i]
        if 0 <= x < SIZE and 0 <= y < SIZE:
            cv2.circle(canvas, (int(x), int(y)), radius,
                       color[i] if per_point else color, -1, cv2.LINE_AA)


# ------------------------------------------------------------ patch colouring
def _patch_axis(frames_pts: Sequence[Sequence[Tuple[float, float, float]]]) -> int:
    """World axis along which the patches of this touch are spread out.

    Patch labels come out in arbitrary order per frame, so colours would flicker.
    Ranking patches along one fixed axis pins each finger to its own colour for
    the whole video.
    """
    spread = np.zeros(3)
    for pts in frames_pts:
        if len(pts) < 2:
            continue
        arr = np.asarray(pts, dtype=float)
        labels = cluster_points(pts, DEFAULT_CLUSTER_EPS, DEFAULT_MIN_CLUSTER)
        if labels.max() < 1:
            continue
        centers = np.stack([arr[labels == g].mean(axis=0) for g in range(labels.max() + 1)])
        spread += centers.max(axis=0) - centers.min(axis=0)
    return int(np.argmax(spread))


def _patch_colors(pts, axis: Optional[int]):
    """One BGR colour per point — by contact patch, ranked along ``axis``."""
    if axis is None or not pts:
        return CUR_COLOR
    arr = np.asarray(pts, dtype=float)
    labels = cluster_points(pts, DEFAULT_CLUSTER_EPS, DEFAULT_MIN_CLUSTER)
    n_patches = int(labels.max()) + 1
    means = [arr[labels == g][:, axis].mean() for g in range(n_patches)]
    rank = np.empty(n_patches, dtype=int)
    rank[np.argsort(means)] = np.arange(n_patches)
    return [PATCH_COLORS[r % len(PATCH_COLORS)] for r in rank[labels]]


# ------------------------------------------------------------------ per touch
def _bbox(all_pts: List[Tuple[float, float, float]]):
    a = np.asarray(all_pts, dtype=float)
    lo, hi = a.min(0), a.max(0)
    pad = np.maximum((hi - lo) * 0.10, 1.0)
    lo, hi = lo - pad, hi + pad
    return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


def _write_video(path: Path, frame_rows, points_of_row, view, bbox, block, tid, label, fps,
                 axis: Optional[int] = None):
    grid = np.full((SIZE, SIZE, 3), 255, np.uint8)
    _draw_grid(grid, bbox, view)
    trail = grid.copy()
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (SIZE, SIZE))
    n = len(frame_rows)
    for k, row in enumerate(frame_rows):
        pts = points_of_row(row)
        frame = trail.copy()
        _draw_points(frame, pts, view, _patch_colors(pts, axis), 3)
        cv2.putText(frame, f"{block}  touch {tid}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{label}   {k + 1}/{n}", (8, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
        vw.write(frame)
        _draw_points(trail, pts, view, TRAIL_COLOR, 1)  # accumulate
    vw.release()


def _aligned_rows(anchors: np.ndarray, max_frames: int = MAX_FRAMES) -> Optional[np.ndarray]:
    """Rows sampled so every anchor lands exactly on a frame.

    A fixed number of frames per anchor interval, so the held original steps at
    a regular cadence instead of alternating 10/11 frames. ``per`` is capped at
    the shortest interval's length to keep the step at >= 1 row (no duplicate
    frames). Returns None when there are too few anchors to align to.
    """
    if len(anchors) < 2:
        return None
    gaps = np.diff(anchors)
    per = max(1, min(max_frames // len(gaps), int(gaps.min())))
    rows: List[int] = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        rows.extend(np.linspace(a, b, per, endpoint=False).round().astype(int).tolist())
    rows.append(int(anchors[-1]))
    return np.unique(rows)


def videos_for_touch(df_orig, interp_cells, touch, tid, out_dir, block, fps,
                     cluster_colors: bool = True, aligned: bool = True):
    rows = np.where(touch == tid)[0]
    r0, r1 = int(rows.min()), int(rows.max())
    span = np.arange(r0, r1 + 1)
    frame_rows = (np.unique(np.linspace(r0, r1, MAX_FRAMES).round().astype(int))
                  if len(span) > MAX_FRAMES else span)

    # original anchors (kinect rows carrying a measurement) within the touch
    anchor_rows = [int(r) for r in span if parse_contact_points(df_orig[r])]
    anchor_pts = {r: parse_contact_points(df_orig[r]) for r in anchor_rows}
    anchor_arr = np.asarray(anchor_rows)

    def held_original(row):
        if len(anchor_arr) == 0:
            return []
        idx = int(np.searchsorted(anchor_arr, row, side="right")) - 1
        return anchor_pts[int(anchor_arr[idx])] if idx >= 0 else []

    def interp_at(row):
        return parse_contact_points(interp_cells[row])

    aligned_rows = _aligned_rows(anchor_arr) if aligned else None

    all_pts = [p for r in anchor_rows for p in anchor_pts[r]]
    for rows_ in (frame_rows, aligned_rows):
        if rows_ is not None:
            for r in rows_:
                all_pts += interp_at(int(r))
    if not all_pts:
        logger.warning("touch %s: no points, skipping", tid)
        return
    # One view and one colour axis for every video of this touch, so all of them
    # are directly comparable.
    bbox = _bbox(all_pts)
    view = _make_view(bbox)
    axis = _patch_axis([anchor_pts[r] for r in anchor_rows]) if cluster_colors else None

    plain_dir = out_dir / "unaligned"
    plain_dir.mkdir(parents=True, exist_ok=True)
    _write_video(plain_dir / f"{block}_touch{tid}_1_original.mp4",
                 frame_rows, held_original, view, bbox, block, tid, "ORIGINAL (uninterp)", fps,
                 axis=axis)
    _write_video(plain_dir / f"{block}_touch{tid}_2_interpolated.mp4",
                 frame_rows, interp_at, view, bbox, block, tid, "INTERPOLATED", fps,
                 axis=axis)
    if aligned_rows is None:
        if aligned:
            logger.info("touch %s: %d anchor(s), too few to align — unaligned only",
                        tid, len(anchor_rows))
        return

    # Same file names as above, one folder over — easy to compare the two timelines.
    aligned_dir = out_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    _write_video(aligned_dir / f"{block}_touch{tid}_1_original.mp4",
                 aligned_rows, held_original, view, bbox, block, tid,
                 "ORIGINAL (aligned)", fps, axis=axis)
    _write_video(aligned_dir / f"{block}_touch{tid}_2_interpolated.mp4",
                 aligned_rows, interp_at, view, bbox, block, tid,
                 "INTERPOLATED (aligned)", fps, axis=axis)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/2022-06-14_ST13-01/blocks"))
    ap.add_argument("--interp-dir", type=Path, default=Path("output"))
    ap.add_argument("--video-dir", type=Path, default=Path("plot/videos"))
    ap.add_argument("--n-touches", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--no-cluster-colors", action="store_true",
                    help="Draw every point in one colour instead of one per contact patch")
    ap.add_argument("--no-aligned", action="store_true",
                    help="Skip the aligned/ folder; write only the unaligned/ videos")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args.video_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # "*_merged_data*.csv" so both the original blocks/ naming and the newer
    # blocks_rf_centered/ "..._merged_data_pca-xyz.csv" are picked up.
    inputs = sorted(args.data_dir.glob("*_merged_data*.csv"))
    for input_csv in inputs:
        interp_csv = args.interp_dir / f"{input_csv.stem}_contact-interpolated.csv"
        if not interp_csv.exists():
            logger.warning("Missing interpolated CSV, skipping: %s", interp_csv.name)
            continue
        block = input_csv.stem.replace("_merged_data", "").replace("2022-06-14_ST13-01_semicontrolled_", "")
        df = pd.read_csv(input_csv, usecols=[TOUCH_COL, ORIG_COL], low_memory=False)
        interp = pd.read_csv(interp_csv, usecols=[INTERP_COL], low_memory=False)
        touch = df[TOUCH_COL].to_numpy()
        orig_cells = df[ORIG_COL].to_numpy()
        interp_cells = interp[INTERP_COL].to_numpy()

        valid = np.unique(touch[~np.isnan(touch)])
        valid = valid[valid > 0]
        picked = sorted(rng.choice(valid, size=min(args.n_touches, len(valid)), replace=False))
        logger.info("%s: touches %s", block, [int(t) for t in picked])
        for tid in picked:
            videos_for_touch(orig_cells, interp_cells, touch, int(tid), args.video_dir, block,
                             args.fps, cluster_colors=not args.no_cluster_colors,
                             aligned=not args.no_aligned)
    logger.info("Done. Videos in %s", args.video_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
