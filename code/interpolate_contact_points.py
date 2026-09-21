#!/usr/bin/env python
"""Geodesic interpolation of contact points across kinect->neural merge gaps.

When the kinect contact data is merged with the higher-rate neural recording,
each kinect frame lands on one row and the rows in between are empty (a gap of
~33 rows). This script fills those gaps by morphing the contact set from one
kinect frame to the next *along the forearm surface*.

Every interpolated point is an actual vertex of the forearm point cloud, so no
interpolated contact can ever fall outside the point-cloud domain (no cutting
through the arm, no floating points).

The surface machinery — kNN graph, all-pairs geodesics, clustering, contact-mode
detection, cluster pairing and the morph itself — lives in ``surface_morph.py``
and is shared with ``interpolate_depth_field.py``. See that module for the
method and for how one-finger and whole-hand blocks are told apart. This file
holds only what is specific to the ``contact_points`` column: reading it,
walking the morph into positions, and writing it back.

Gap rule
--------
Interpolation only happens between two *consecutive* kinect samples that both
have contacts. Consecutive non-empty rows further apart than ``--max-gap``
imply that a kinect sample in between was genuinely empty (contact ended), so
that span is left empty rather than interpolated through.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from surface_morph import (
    DEFAULT_CLUSTER_EPS,
    DEFAULT_K,
    DEFAULT_MIN_CLUSTER,
    Point,
    build_surface_graph,
    detect_contact_mode,
    load_forearm_vertices,
    morph_paths,
    parse_contact_points,
    precompute_geodesics,
    serialize_contact_points,
    walk_fractions,
)

logger = logging.getLogger(__name__)

CONTACT_COL = "contact_points"
AREA_COL = "contact_area_metadata"  # optional; logged as a cross-check on the detected mode
DEFAULT_MAX_GAP = 33  # kinect->neural upsampling factor
DEFAULT_PRECISION = 1  # decimals; matches the input's ~0.1 mm precision
DEFAULT_TIME_COL = "time_nerve"  # dense (neural-rate) time column carried into the output


def interpolate_between(
    vertices: np.ndarray,
    geo_dist: np.ndarray,
    geo_pred: np.ndarray,
    tree: cKDTree,
    start_points: Sequence[Point],
    end_points: Sequence[Point],
    n_inner: int,
    cluster_eps: Optional[float] = None,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER,
) -> List[List[Point]]:
    """Morph ``start_points`` into ``end_points`` over ``n_inner`` rows."""
    if n_inner <= 0:
        return []

    paths = morph_paths(vertices, geo_dist, geo_pred, tree, start_points, end_points,
                        cluster_eps, min_cluster_size)
    if not paths:
        return [[] for _ in range(n_inner)]

    fractions = [step / (n_inner + 1) for step in range(1, n_inner + 1)]
    # One walk per path across every fraction, so each path's arc length is
    # measured once instead of once per row.
    per_path = [walk_fractions(path, vertices, fractions) for path in paths]

    frames: List[List[Point]] = []
    for step in range(n_inner):
        reached = {landed[step] for landed in per_path}
        frames.append([tuple(vertices[i]) for i in sorted(reached)])
    return frames


# --------------------------------------------------------------------- driver

def interpolate_contact_column(
    csv_path: Path,
    vertices: np.ndarray,
    geo_dist: np.ndarray,
    geo_pred: np.ndarray,
    tree: cKDTree,
    column: str = CONTACT_COL,
    max_gap: int = DEFAULT_MAX_GAP,
    precision: Optional[int] = DEFAULT_PRECISION,
    time_col: Optional[str] = DEFAULT_TIME_COL,
    contact_mode: str = "auto",
    cluster_eps: float = DEFAULT_CLUSTER_EPS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER,
) -> pd.DataFrame:
    """Return an aligned frame: a dense time column plus the interpolated contacts."""
    frame = pd.read_csv(csv_path, low_memory=False)
    if column not in frame.columns:
        raise KeyError(f"'{column}' not found in {csv_path.name}. Columns: {list(frame.columns)}")

    parsed = [parse_contact_points(cell) for cell in frame[column]]
    output = ["[]"] * len(parsed)

    # Anchors are the rows that actually carry a kinect measurement. Keep the
    # measured contacts as-is (only reformatted / rounded) — do NOT snap them;
    # snapping is applied only to the interpolated in-between rows.
    anchors = [i for i, points in enumerate(parsed) if points]
    for i in anchors:
        output[i] = serialize_contact_points(parsed[i], precision=precision)

    # One decision per block, held for the whole block.
    logger.info("%s: %d anchor frames", csv_path.name, len(anchors))
    mode = contact_mode
    if mode == "auto":
        mode = detect_contact_mode(parsed, cluster_eps, min_cluster_size)
    else:
        logger.info("  contact mode forced to '%s'", mode)
    if AREA_COL in frame.columns:  # cross-check only; never drives the algorithm
        recorded = frame[AREA_COL].dropna().unique()
        logger.info("  (%s says: %s)", AREA_COL, ", ".join(map(str, recorded[:4])) or "-")
    eps = cluster_eps if mode == "multi" else None

    n_filled = n_skipped = 0
    for start, end in zip(anchors, anchors[1:]):
        n_inner = end - start - 1
        if n_inner <= 0:
            continue
        if (end - start) > max_gap:
            # A kinect sample in between was genuinely empty -> contact ended.
            n_skipped += 1
            continue

        for offset, points in enumerate(
            interpolate_between(
                vertices, geo_dist, geo_pred, tree, parsed[start], parsed[end], n_inner,
                cluster_eps=eps, min_cluster_size=min_cluster_size,
            ),
            start=1,
        ):
            output[start + offset] = serialize_contact_points(points, precision=precision)
        n_filled += n_inner

    logger.info(
        "  %d rows interpolated, %d spans skipped (empty endpoint)", n_filled, n_skipped,
    )

    columns = {}
    if time_col is not None:
        if time_col in frame.columns:
            columns[time_col] = frame[time_col].to_numpy()  # dense timestamp per row
        else:
            logger.warning(
                "Time column '%s' not in %s; output will have no time column.",
                time_col, csv_path.name,
            )
    columns[f"{column}_interpolated"] = output
    return pd.DataFrame(columns)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", required=True, type=Path,
                        help="Forearm point cloud (e.g. <session>_unified_registered.ply)")
    parser.add_argument("--contacts", required=True, nargs="+", type=Path,
                        help="One or more block CSVs containing the contact column")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory")
    parser.add_argument("--column", default=CONTACT_COL, help=f"Contact column (default: {CONTACT_COL})")
    parser.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP,
                        help=f"Max row distance between consecutive kinect samples (default: {DEFAULT_MAX_GAP})")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"kNN neighbours (default: {DEFAULT_K})")
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION,
                        help=f"Decimals to round output coords (default: {DEFAULT_PRECISION}; -1 = full precision)")
    parser.add_argument("--time-col", default=DEFAULT_TIME_COL,
                        help=f"Dense time column to include as the first output column "
                             f"(default: {DEFAULT_TIME_COL}; 'none' to omit)")
    parser.add_argument("--contact-mode", choices=("auto", "single", "multi"), default="auto",
                        help="'single' = one contact patch per frame (one finger tip); "
                             "'multi' = several finger patches, morphed cluster-by-cluster "
                             "(whole hand); 'auto' (default) decides once per block")
    parser.add_argument("--cluster-eps", type=float, default=DEFAULT_CLUSTER_EPS,
                        help=f"Distance below which contact points join the same patch, in the "
                             f"data's units (default: {DEFAULT_CLUSTER_EPS} mm)")
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER,
                        help=f"Patches smaller than this are absorbed into their nearest "
                             f"neighbour (default: {DEFAULT_MIN_CLUSTER} points)")
    args = parser.parse_args(argv)
    precision = None if args.precision is not None and args.precision < 0 else args.precision
    time_col = None if args.time_col.lower() == "none" else args.time_col

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    vertices = load_forearm_vertices(args.ply)
    graph = build_surface_graph(vertices, k=args.k)
    geo_dist, geo_pred = precompute_geodesics(graph)
    tree = cKDTree(vertices)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for csv_path in args.contacts:
        result = interpolate_contact_column(
            csv_path, vertices, geo_dist, geo_pred, tree,
            column=args.column, max_gap=args.max_gap, precision=precision,
            time_col=time_col, contact_mode=args.contact_mode,
            cluster_eps=args.cluster_eps, min_cluster_size=args.min_cluster_size,
        )
        destination = args.outdir / f"{csv_path.stem}_contact-interpolated.csv"
        result.to_csv(destination, index=False)
        logger.info("Wrote %s (%d rows)", destination.name, len(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
