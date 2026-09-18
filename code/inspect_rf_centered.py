#!/usr/bin/env python
"""Examine the blocks_rf_centered/ data and diff it against blocks/.

Answers, per block:

1. What is in the parquet that is not in the CSV (and vice versa)?
2. Which columns changed between blocks/ and blocks_rf_centered/?
3. Do the CSV and the parquet describe the same contacts?
4. What happened to depth?

Read-only. Nothing here modifies the data or ``plot_contact_depth_field.py`` --
the parquet metadata handling is deliberately the same shape as that script's
``load_depth_field`` so the two agree on how a file describes itself.

Usage
-----
    python code/inspect_rf_centered.py
    python code/inspect_rf_centered.py --block block-order-02
    python code/inspect_rf_centered.py --session data/2022-06-14_ST13-01
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from interpolate_contact_points import parse_contact_points

RULE = "=" * 78
# The CSV serialises coordinates at 1 decimal, so a point that came from the
# parquet's float32 can sit up to ~0.087 mm away in 3-D. Anything under this is
# a rounding difference, not a different point.
ROUNDING_TOL = 0.15


# --------------------------------------------------------------------- parquet

def describe_parquet(path: Path) -> pd.DataFrame:
    """Print the parquet's self-description and its internal consistency."""
    pf = pq.ParquetFile(path)
    meta = {k.decode(): v.decode() for k, v in (pf.schema_arrow.metadata or {}).items()}

    print("  -- file metadata (the only trustworthy statement of what the numbers mean) --")
    for key in sorted(meta):
        value = meta[key]
        if len(value) > 90:  # source_artifact is a very long absolute path
            value = value[:44] + " … " + value[-40:]
        print(f"     {key:28s} {value}")

    print("  -- schema --")
    print("     " + ", ".join(f"{f.name}:{f.type}" for f in pf.schema_arrow))

    table = pf.read().to_pandas()
    frames = table["frame_index"].nunique()
    print(f"  -- {len(table):,} rows over {frames:,} contacting frames "
          f"({len(table) / max(frames, 1):.1f} rows per frame) --")

    # The viewer documents "one row per (frame, contact vertex)". Check it.
    groups = table.groupby(["frame_index", "vertex_id"])["signed_depth_mm"]
    sizes = groups.size()
    duplicated = sizes[sizes > 1]
    if len(duplicated) == 0:
        print("     (frame, vertex_id) is unique — matches the documented contract")
    else:
        spread = groups.max() - groups.min()
        spread = spread[sizes > 1]
        print(f"     ** (frame, vertex_id) is NOT unique: {len(duplicated):,} of "
              f"{len(sizes):,} pairs repeat (up to {int(sizes.max())}x)")
        print(f"        and the repeats disagree on depth: spread up to "
              f"{spread.max():.3f} mm, median {spread.median():.3f} mm")
        print(f"        -> vertex_id is not a key here; {table['vertex_id'].nunique():,} "
              f"distinct vertices carry {len(table):,} rows")

    depth = table["signed_depth_mm"]
    sign = meta.get("sign_convention", "?")
    print(f"  -- signed_depth_mm: [{depth.min():.3f}, {depth.max():.3f}] mm, "
          f"{100 * (depth < 0).mean():.1f}% negative (convention: {sign}) --")

    ply = meta.get("reference_ply")
    if ply:
        count = meta.get("reference_ply_vertex_count", "?")
        found = list(path.parents[2].rglob(ply)) if ply else []
        where = found[0].relative_to(path.parents[2]) if found else "NOT PRESENT in this repo"
        print(f"  -- vertex_id indexes {ply} ({count} vertices): {where} --")
    return table


# ------------------------------------------------------------------- csv diff

def compare_csv(old_path: Path, new_path: Path) -> None:
    """Column-level diff between the blocks/ CSV and the rf-centred CSV."""
    old = pd.read_csv(old_path, low_memory=False)
    new = pd.read_csv(new_path, low_memory=False)
    print(f"  rows: old {len(old):,} | new {len(new):,} "
          f"({'same' if len(old) == len(new) else 'DIFFERENT'})")

    added = [c for c in new.columns if c not in old.columns]
    removed = [c for c in old.columns if c not in new.columns]
    print(f"  columns: old {len(old.columns)} | new {len(new.columns)}"
          f" | added {added or 'none'} | removed {removed or 'none'}")

    changed, identical = [], []
    for col in old.columns:
        if col not in new.columns:
            continue
        a, b = old[col], new[col]
        if a.dtype.kind in "biufc" and b.dtype.kind in "biufc":
            same = np.allclose(a.fillna(-9e9), b.fillna(-9e9), equal_nan=True)
        else:
            same = bool((a.fillna("\0") == b.fillna("\0")).all())
        (identical if same else changed).append(col)

    print(f"  {len(identical)} columns byte-identical, {len(changed)} changed")
    print(f"  CHANGED: {', '.join(changed) if changed else 'none'}")


def coordinate_shift(old_path: Path, new_path: Path) -> None:
    """How the contact geometry itself differs, beyond a change of frame."""
    old = pd.read_csv(old_path, low_memory=False, usecols=["contact_points"])["contact_points"]
    new = pd.read_csv(new_path, low_memory=False, usecols=["contact_points"])["contact_points"]

    rows = [i for i, c in enumerate(old) if len(parse_contact_points(c)) > 40][:8]
    if not rows:
        print("  (no frame with enough contacts to compare)")
        return

    distortion, old_nn, new_nn, dupes = [], [], [], []
    for i in rows:
        a = np.asarray(parse_contact_points(old[i]), dtype=float)
        b = np.asarray(parse_contact_points(new[i]), dtype=float)
        if len(a) != len(b):
            continue
        # A pure rotation+translation preserves every pairwise distance.
        da = np.linalg.norm(a[:, None] - a[None], axis=-1)
        db = np.linalg.norm(b[:, None] - b[None], axis=-1)
        upper = np.triu_indices(len(a), 1)
        distortion.append(np.abs(da - db)[upper])
        for d, acc in ((da, old_nn), (db, new_nn)):
            filled = d + np.eye(len(d)) * 1e9
            acc.append(np.median(filled.min(axis=1)))
        dupes.append(100.0 * (1.0 - len(np.unique(b, axis=0)) / len(b)))

    alld = np.concatenate(distortion)
    print(f"  pairwise-distance distortion over {len(distortion)} frames: "
          f"median {np.median(alld):.3f} mm, p95 {np.percentile(alld, 95):.3f}, "
          f"max {alld.max():.3f}")
    print("  -> NOT a rigid transform; the contacts were re-projected, not just rotated"
          if alld.max() > 0.5 else "  -> consistent with a rigid transform")
    print(f"  nearest-neighbour spacing within a frame: old {np.median(old_nn):.3f} mm "
          f"| new {np.median(new_nn):.3f} mm")
    print(f"  duplicate points per frame (new): {np.mean(dupes):.1f}% of rows")


# ------------------------------------------------------- csv <-> parquet link

def link_csv_parquet(new_csv: Path, table: pd.DataFrame) -> None:
    """Do the CSV's contact_points and the parquet's rows describe the same thing?"""
    csv = pd.read_csv(new_csv, low_memory=False,
                      usecols=["frame_index", "contact_points", "contact_depth"])
    counts = csv["contact_points"].apply(lambda c: len(parse_contact_points(c)))

    csv_frames = set(csv.loc[counts > 0, "frame_index"].dropna().astype(int))
    pq_frames = set(table["frame_index"].astype(int))
    print(f"  contacting frames: CSV {len(csv_frames):,} | parquet {len(pq_frames):,} "
          f"| only-CSV {len(csv_frames - pq_frames)} | only-parquet {len(pq_frames - csv_frames)}")

    per_frame = table.groupby("frame_index").size()
    csv_per_frame = csv.loc[counts > 0].assign(n=counts[counts > 0]).set_index("frame_index")["n"]
    csv_per_frame = csv_per_frame[~csv_per_frame.index.duplicated()]
    joined = per_frame.rename("parquet").to_frame().join(csv_per_frame.rename("csv"), how="inner")
    agree = (joined["parquet"] == joined["csv"]).mean() if len(joined) else float("nan")
    print(f"  per-frame point counts agree on {100 * agree:.1f}% of frames "
          f"(mean {joined['parquet'].mean():.1f} points/frame)")

    # Positions: every CSV point should sit on a parquet row, within CSV rounding.
    worst, checked = 0.0, 0
    for frame in list(pq_frames)[:40]:
        P = table.loc[table["frame_index"] == frame, ["x", "y", "z"]].to_numpy(np.float64)
        cells = csv.loc[csv["frame_index"] == frame, "contact_points"].dropna()
        C = np.asarray([p for c in cells for p in parse_contact_points(c)], dtype=np.float64)
        if len(C) == 0 or len(P) == 0:
            continue
        worst = max(worst, cKDTree(P).query(C)[0].max())
        checked += 1
    verdict = "same points" if worst <= ROUNDING_TOL else "DIFFERENT points"
    print(f"  positions over {checked} frames: worst CSV->parquet distance {worst:.4f} mm "
          f"-> {verdict} (CSV rounds to 0.1 mm)")


def depth_relation(new_csv: Path, table: pd.DataFrame) -> None:
    """How the CSV's scalar contact_depth relates to the parquet's depth field."""
    csv = pd.read_csv(new_csv, low_memory=False, usecols=["frame_index", "contact_depth"])
    csv = csv.dropna(subset=["contact_depth", "frame_index"])
    csv = csv[~csv["frame_index"].duplicated()].set_index("frame_index")["contact_depth"]

    deepest = table.groupby("frame_index")["signed_depth_mm"].min().rename("min_signed")
    joined = deepest.to_frame().join(csv, how="inner").dropna()
    if joined.empty:
        print("  (no frames in common)")
        return

    residual = (joined["contact_depth"] + joined["min_signed"]).abs()
    print(f"  joined {len(joined):,} frames | "
          f"corr(contact_depth, -min signed_depth) = "
          f"{np.corrcoef(joined['contact_depth'], -joined['min_signed'])[0, 1]:.6f}")
    print(f"  |contact_depth - (-min signed_depth)|: max {residual.max():.6f} mm")
    if residual.max() < 1e-4:
        print("  -> the CSV's contact_depth is exactly the peak penetration of the field;")
        print("     the parquet is that same quantity resolved per contact point.")


# --------------------------------------------------------------------- driver

def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, default=Path("data/2022-06-14_ST13-01"))
    ap.add_argument("--block", default=None, help="e.g. block-order-02 (default: all)")
    args = ap.parse_args(argv)

    new_dir, old_dir = args.session / "blocks_rf_centered", args.session / "blocks"
    parquets = sorted(new_dir.glob("*_contact_depth_field_pca-xyz.parquet"))
    if args.block:
        parquets = [p for p in parquets if args.block in p.name]
    if not parquets:
        print(f"No parquet found in {new_dir}")
        return 1

    for parquet in parquets:
        tag = re.search(r"(block-order-\d+)", parquet.name)
        block = tag.group(1) if tag else parquet.stem
        new_csv = next(new_dir.glob(f"*{block}_merged_data_pca-xyz.csv"), None)
        old_csv = next(old_dir.glob(f"*{block}_merged_data.csv"), None)

        print(f"\n{RULE}\n{block}\n{RULE}")
        print("\n[1] PARQUET — what it is and whether it is self-consistent")
        table = describe_parquet(parquet)

        if old_csv and new_csv:
            print("\n[2] CSV blocks/ vs blocks_rf_centered/ — which columns moved")
            compare_csv(old_csv, new_csv)
            print("\n[3] CONTACT GEOMETRY — is it the same shape in a new frame?")
            coordinate_shift(old_csv, new_csv)
        else:
            print("\n[2/3] skipped — missing the matching CSV in blocks/ or blocks_rf_centered/")

        if new_csv:
            print("\n[4] CSV <-> PARQUET — do they describe the same contacts?")
            link_csv_parquet(new_csv, table)
            print("\n[5] DEPTH — scalar column vs the field")
            depth_relation(new_csv, table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
