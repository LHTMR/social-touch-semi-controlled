#!/usr/bin/env python
"""Geodesic interpolation of the contact *depth field* across merge gaps.

The companion to ``interpolate_contact_points.py``. That tool fills the gaps in
*where* the hand touched; this one fills the gaps in *how hard*, per point.

The depth field (``*_contact_depth_field_pca-xyz.parquet``) is long form: one row
per contact sample, carrying ``signed_depth_mm`` (negative = penetrating) and the
``vertex_id`` of the forearm vertex it sits on. Like the contact points it exists
only on kinect frames, so on the neural timeline it is a value every ~33 rows and
nothing in between.

Method
------
Exactly the morph used for contact points, from ``surface_morph.py``: cluster
each frame's contacts into patches, pair the patches, match contacts within a
pair, and walk each matched pair along the surface. The one addition is that
each path also carries a value. ``morph_paths`` gives paths whose first vertex is
the source and whose last is the target, so at fraction ``t`` along a path the
interpolated depth is

    (1 - t) * depth_at_source  +  t * depth_at_target

blended linearly while the *position* follows the geodesic. A point therefore
gets deeper or shallower as it slides across the arm, and never leaves it.

Gap rule
--------
Stricter and more exact than the contact tool's ``--max-gap`` row heuristic.
The parquet carries ``frame_index``, so two contacting frames are adjacent
kinect samples exactly when their frame indices differ by 1. Anything larger
means a kinect frame in between recorded no contact at all — the merged CSV
stores ``contact_depth == 0`` on those rows — so the span is left empty rather
than interpolated through. ``--max-frame-gap`` relaxes this if you want it to.

Duplicate vertices
------------------
These files carry several rows for the same ``(frame_index, vertex_id)``, at the
same position but with different depths (8-19% of pairs; see
``inspect_rf_centered.py``). A morph needs one value per vertex, so duplicates
are reduced by ``--aggregate``, default ``min`` = deepest. That matches the
pipeline's own frame summary: the merged CSV's ``contact_depth`` is exactly
``-min(signed_depth_mm)`` over the frame. The output therefore has a unique
``(row, vertex_id)``, unlike the input.

Output
------
One parquet per block, long form like the input:

    row              int32    index into the merged CSV / neural timeline
    time_nerve       float64  dense timestamp, for aligning to the nerve signal
    x, y, z          float32  vertex position, same space as the input
    signed_depth_mm  float64  negative = penetrating, as the input
    vertex_id        int32    index into the reference forearm PLY
    is_measured      bool     True on kinect-frame rows, False on filled rows

Measured rows carry the original (aggregated) depths; only the in-between rows
are interpolated. To view one with ``plot_contact_depth_field.py``, rename
``row`` -> ``frame_index`` and ``time_nerve`` -> ``time_s``.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from surface_morph import (
    DEFAULT_CLUSTER_EPS,
    DEFAULT_K,
    DEFAULT_MIN_CLUSTER,
    build_surface_graph,
    detect_contact_mode,
    load_forearm_vertices,
    morph_paths,
    precompute_geodesics,
    walk_fractions,
)

logger = logging.getLogger(__name__)

DEPTH_COL = "signed_depth_mm"
VERTEX_COL = "vertex_id"
FRAME_COL = "frame_index"
DEFAULT_TIME_COL = "time_nerve"
DEFAULT_MAX_FRAME_GAP = 1  # adjacent kinect frames only
DEFAULT_AGGREGATE = "min"  # deepest; matches the CSV's own contact_depth summary

AGGREGATORS = {"min": np.minimum, "max": np.maximum}


# ----------------------------------------------------------------- loading

def load_depth_field(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Long-form depth table plus the file's own metadata."""
    handle = pq.ParquetFile(path)
    meta = {k.decode(): v.decode() for k, v in (handle.schema_arrow.metadata or {}).items()}
    if meta.get("units", "mm") != "mm":
        raise ValueError(f"expected millimetres, file declares {meta.get('units')!r}")
    table = handle.read().to_pandas()
    logger.info("  %s: %d rows over %d contacting frames",
                path.name, len(table), table[FRAME_COL].nunique())
    return table, meta


def aggregate_duplicates(table: pd.DataFrame, how: str = DEFAULT_AGGREGATE) -> pd.DataFrame:
    """Collapse repeated (frame, vertex) rows to one depth each."""
    before = len(table)
    reduced = (table.groupby([FRAME_COL, VERTEX_COL], sort=True)[DEPTH_COL]
               .agg(how).reset_index())
    if len(reduced) != before:
        logger.info("  aggregated %d rows -> %d unique (frame, vertex) by '%s'",
                    before, len(reduced), how)
    return reduced


def frame_rows(csv_path: Path, time_col: str) -> Tuple[Dict[int, int], np.ndarray, int]:
    """Map kinect frame_index -> merged-CSV row, plus the dense time column."""
    frame = pd.read_csv(csv_path, low_memory=False)
    if FRAME_COL not in frame.columns:
        raise KeyError(f"'{FRAME_COL}' not in {csv_path.name}")
    times = (frame[time_col].to_numpy() if time_col in frame.columns
             else np.full(len(frame), np.nan))
    if time_col not in frame.columns:
        logger.warning("  time column '%s' not in %s; output times will be NaN",
                       time_col, csv_path.name)

    values = frame[FRAME_COL].to_numpy()
    rows = np.flatnonzero(~pd.isna(values))
    mapping: Dict[int, int] = {}
    for row in rows:  # first occurrence wins if a frame spans several rows
        mapping.setdefault(int(values[row]), int(row))
    return mapping, times, len(frame)


# ------------------------------------------------------------------- morph

def interpolate_depth_field(
    table: pd.DataFrame,
    vertices: np.ndarray,
    geo_dist: np.ndarray,
    geo_pred: np.ndarray,
    tree: cKDTree,
    mapping: Dict[int, int],
    times: np.ndarray,
    cluster_eps: Optional[float],
    min_cluster_size: int = DEFAULT_MIN_CLUSTER,
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    how: str = DEFAULT_AGGREGATE,
) -> pd.DataFrame:
    """Fill the rows between consecutive kinect frames with a morphed depth field."""
    combine = AGGREGATORS[how] if how in AGGREGATORS else None
    by_frame = {int(f): g for f, g in table.groupby(FRAME_COL, sort=True)}
    frames = sorted(by_frame)

    out_row: List[np.ndarray] = []
    out_vertex: List[np.ndarray] = []
    out_depth: List[np.ndarray] = []
    out_measured: List[np.ndarray] = []

    # Measured frames first: keep their own (aggregated) depths untouched.
    n_missing = 0
    for f in frames:
        row = mapping.get(f)
        if row is None:
            n_missing += 1
            continue
        group = by_frame[f]
        out_row.append(np.full(len(group), row, dtype=np.int64))
        out_vertex.append(group[VERTEX_COL].to_numpy(np.int64))
        out_depth.append(group[DEPTH_COL].to_numpy(np.float64))
        out_measured.append(np.ones(len(group), dtype=bool))
    if n_missing:
        logger.warning("  %d contacting frames have no row in the merged CSV; skipped", n_missing)

    n_filled = n_rows = n_skipped = 0
    for f0, f1 in zip(frames, frames[1:]):
        if f1 - f0 > max_frame_gap:
            n_skipped += 1  # a kinect frame in between recorded no contact
            continue
        r0, r1 = mapping.get(f0), mapping.get(f1)
        if r0 is None or r1 is None:
            continue
        n_inner = r1 - r0 - 1
        if n_inner <= 0:
            continue

        start, end = by_frame[f0], by_frame[f1]
        start_ids = start[VERTEX_COL].to_numpy(np.int64)
        end_ids = end[VERTEX_COL].to_numpy(np.int64)
        # Depth looked up by vertex; the ids are unique after aggregation.
        depth_of_start = dict(zip(start_ids.tolist(), start[DEPTH_COL].to_numpy(np.float64)))
        depth_of_end = dict(zip(end_ids.tolist(), end[DEPTH_COL].to_numpy(np.float64)))

        # Every depth-field point already sits on a forearm vertex, so feed the
        # morph the vertex positions themselves rather than the stored x/y/z
        # (which are the same points at float32).
        paths = morph_paths(vertices, geo_dist, geo_pred, tree,
                            vertices[start_ids], vertices[end_ids],
                            cluster_eps, min_cluster_size)
        if not paths:
            continue

        fractions = [step / (n_inner + 1) for step in range(1, n_inner + 1)]
        landed = [walk_fractions(path, vertices, fractions) for path in paths]
        ends = [(path[0], path[-1]) for path in paths]

        for step, t in enumerate(fractions):
            # Several paths can land on one vertex; combine their depths the
            # same way duplicates were combined in the first place.
            reached: Dict[int, float] = {}
            for (s, e), positions in zip(ends, landed):
                depth = (1.0 - t) * depth_of_start[s] + t * depth_of_end[e]
                v = positions[step]
                previous = reached.get(v)
                if previous is None:
                    reached[v] = depth
                elif combine is not None:
                    reached[v] = float(combine(previous, depth))
                else:
                    reached[v] = 0.5 * (previous + depth)
            if not reached:
                continue
            ids = np.fromiter(reached.keys(), dtype=np.int64, count=len(reached))
            order = np.argsort(ids)
            out_row.append(np.full(len(ids), r0 + 1 + step, dtype=np.int64))
            out_vertex.append(ids[order])
            out_depth.append(np.fromiter(reached.values(), dtype=np.float64,
                                         count=len(reached))[order])
            out_measured.append(np.zeros(len(ids), dtype=bool))
            n_rows += 1
        n_filled += n_inner

    logger.info("  %d rows interpolated, %d spans skipped (contact ended in between)",
                n_rows, n_skipped)

    row = np.concatenate(out_row) if out_row else np.empty(0, np.int64)
    vertex = np.concatenate(out_vertex) if out_vertex else np.empty(0, np.int64)
    depth = np.concatenate(out_depth) if out_depth else np.empty(0, np.float64)
    measured = np.concatenate(out_measured) if out_measured else np.empty(0, bool)

    order = np.lexsort((vertex, row))
    row, vertex, depth, measured = row[order], vertex[order], depth[order], measured[order]
    positions = vertices[vertex]

    return pd.DataFrame({
        "row": row.astype(np.int32),
        DEFAULT_TIME_COL: times[row] if len(times) else np.full(len(row), np.nan),
        "x": positions[:, 0].astype(np.float32),
        "y": positions[:, 1].astype(np.float32),
        "z": positions[:, 2].astype(np.float32),
        DEPTH_COL: depth,
        VERTEX_COL: vertex.astype(np.int32),
        "is_measured": measured,
    })


# ------------------------------------------------------------------ driver

def _match_csv(parquet: Path, candidates: Sequence[Path]) -> Optional[Path]:
    """The merged CSV belonging to the same block as ``parquet``."""
    found = re.search(r"block-order-?(\d+)", parquet.name)
    if not found:
        return None
    tag = found.group(1)
    for csv in candidates:
        hit = re.search(r"block-order-?(\d+)", csv.name)
        if hit and hit.group(1) == tag:
            return csv
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", required=True, type=Path,
                        help="Reference forearm PLY the vertex_ids index "
                             "(must match reference_ply_vertex_count)")
    parser.add_argument("--fields", required=True, nargs="+", type=Path,
                        help="One or more *_contact_depth_field_*.parquet")
    parser.add_argument("--merged", required=True, nargs="+", type=Path,
                        help="The matching merged CSVs (for frame_index -> row and the time column)")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"kNN neighbours (default: {DEFAULT_K})")
    parser.add_argument("--time-col", default=DEFAULT_TIME_COL,
                        help=f"Dense time column copied from the merged CSV (default: {DEFAULT_TIME_COL})")
    parser.add_argument("--contact-mode", choices=("auto", "single", "multi"), default="auto",
                        help="'single' = one contact patch per frame; 'multi' = several finger "
                             "patches morphed cluster-by-cluster; 'auto' decides once per block")
    parser.add_argument("--cluster-eps", type=float, default=DEFAULT_CLUSTER_EPS,
                        help=f"Patch clustering distance (default: {DEFAULT_CLUSTER_EPS} mm)")
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER,
                        help=f"Patches smaller than this are absorbed (default: {DEFAULT_MIN_CLUSTER})")
    parser.add_argument("--max-frame-gap", type=int, default=DEFAULT_MAX_FRAME_GAP,
                        help=f"Max kinect-frame distance to interpolate across "
                             f"(default: {DEFAULT_MAX_FRAME_GAP} = adjacent frames only)")
    parser.add_argument("--aggregate", choices=("min", "max", "mean"), default=DEFAULT_AGGREGATE,
                        help=f"How to reduce duplicate (frame, vertex) depths "
                             f"(default: {DEFAULT_AGGREGATE} = deepest)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    vertices = load_forearm_vertices(args.ply)
    geo_dist, geo_pred = precompute_geodesics(build_surface_graph(vertices, k=args.k))
    tree = cKDTree(vertices)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for parquet in args.fields:
        logger.info("%s", parquet.name)
        csv_path = _match_csv(parquet, args.merged)
        if csv_path is None:
            logger.error("  no merged CSV matches this block; skipping")
            continue

        table, meta = load_depth_field(parquet)
        declared = meta.get("reference_ply_vertex_count")
        if declared is not None and int(declared) != len(vertices):
            raise ValueError(
                f"{parquet.name} was written against a PLY of {declared} vertices but "
                f"{args.ply.name} has {len(vertices)}. These vertex_ids do not address it."
            )

        table = aggregate_duplicates(table, args.aggregate)
        mapping, times, n_csv_rows = frame_rows(csv_path, args.time_col)

        mode = args.contact_mode
        if mode == "auto":
            frames = [vertices[g[VERTEX_COL].to_numpy(np.int64)]
                      for _, g in table.groupby(FRAME_COL, sort=True)]
            mode = detect_contact_mode(frames, args.cluster_eps, args.min_cluster_size)
        else:
            logger.info("  contact mode forced to '%s'", mode)
        eps = args.cluster_eps if mode == "multi" else None

        result = interpolate_depth_field(
            table, vertices, geo_dist, geo_pred, tree, mapping, times,
            cluster_eps=eps, min_cluster_size=args.min_cluster_size,
            max_frame_gap=args.max_frame_gap, how=args.aggregate,
        )

        # Carry the input's self-description forward, plus how we filled it.
        out_meta = {k: v for k, v in meta.items()}
        out_meta.update({
            "interpolated_by": "interpolate_depth_field.py",
            "interpolated_from": parquet.name,
            "merged_csv": csv_path.name,
            "row_index": f"index into {csv_path.name} (0..{n_csv_rows - 1})",
            "duplicate_aggregate": args.aggregate,
            "contact_mode": mode,
            "cluster_eps_mm": str(args.cluster_eps),
            "max_frame_gap": str(args.max_frame_gap),
        })
        arrow = pa.Table.from_pandas(result, preserve_index=False)
        arrow = arrow.replace_schema_metadata({k: str(v) for k, v in out_meta.items()})

        destination = args.outdir / f"{parquet.stem}_interpolated.parquet"
        pq.write_table(arrow, destination, compression="zstd")
        size = destination.stat().st_size / 1e6
        logger.info("  wrote %s (%d rows over %d timeline rows, %.1f MB)",
                    destination.name, len(result), result["row"].nunique(), size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
