"""Load a contact-depth-field parquet and step through its frames one by one.

A minimal, self-contained example for collaborators. Nothing from the pipeline
repository is imported — the parquet is plain pyarrow and carries everything
needed to interpret itself in its file-level metadata.

Usage
-----
    # no arguments: every block, forearm underneath, 24 frames each, into
    # plot/depth_field/<block-order-NN>/
    python code/plot_contact_depth_field.py

    # just the two stroke blocks, more frames each
    python code/plot_contact_depth_field.py \
        data/.../blocks_rf_centered/*block-order-0[34]_contact_depth_field_pca-xyz.parquet \
        --max-frames 60

    # step through interactively instead: left / right arrows, q to quit
    python code/plot_contact_depth_field.py --show

    # no forearm underneath, elsewhere
    python code/plot_contact_depth_field.py --no-forearm --save out/

The interactive mode calls ``plt.show()``. Under the VS Code debugger that hangs
on some backends — use ``--save`` there and open the PNGs.

Dependencies: pandas, pyarrow, matplotlib. Optional: open3d, for ``--forearm``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# --------------------------------------------------------------------------
# Repo defaults. Added so the script runs in place with no arguments; the
# unmodified original is in git at commit 2c45d86. Paths resolve from this
# file, not the working directory, so it runs from anywhere.
# --------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
_SESSION = _REPO / "data" / "2022-06-14_ST13-01"

DEFAULT_PARQUET_DIR = _SESSION / "blocks_rf_centered"
PARQUET_GLOB = "*_contact_depth_field_pca-xyz.parquet"
DEFAULT_FOREARM = _SESSION / "2022-06-14_ST13-01_forearm.ply"
DEFAULT_SAVE = _REPO / "plot" / "depth_field"
# 96 frames ~= 3-4 s at the kinect rate. 24 was too few to see a stroke travel:
# the contact centroid only moves ~8-25 mm in that time, versus 22-34 mm over 96.
DEFAULT_MAX_FRAMES = 96


def block_tag(path: Path) -> str:
    """'block-order-03' out of a parquet name, for the per-block output folder."""
    found = re.search(r"block-order-?(\d+)", path.name)
    return f"block-order-{found.group(1)}" if found else path.stem


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_depth_field(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return the long-form table and the file's own metadata.

    The table has exactly these columns, in this order:

        frame_index      int32    Kinect frame index; the join key to the CSV
        time_s           float64  frame timestamp, seconds
        x, y, z          float32  vertex position, mm, in the declared space
        signed_depth_mm  float64  signed distance to the hand; NEGATIVE = penetrating
        vertex_id        int32    row index into the reference forearm PLY, file order

    One row per (frame, contact vertex). A frame with no contact contributes
    ZERO rows — that is zero contact, not missing data.
    """
    pf = pq.ParquetFile(path)
    meta = {k.decode(): v.decode() for k, v in pf.schema_arrow.metadata.items()}

    # The metadata is the only trustworthy statement of what the numbers mean.
    # A file sitting in blocks_rf_centered/ can legitimately declare
    # 'pca_calibrated' if receptive-field clustering found no cluster, so read
    # the space rather than inferring it from the directory name.
    print(f"schema_version   : {meta['schema_version']}")
    print(f"coordinate_space : {meta['coordinate_space']}")
    print(f"units            : {meta['units']}")
    print(f"sign_convention  : {meta['sign_convention']}")
    print(f"reference_ply    : {meta.get('reference_ply', '<absent: no vertex_id>')}")

    if meta["units"] != "mm":
        raise ValueError(f"expected millimetres, file declares {meta['units']!r}")

    table = pf.read().to_pandas()

    # Storage stays signed; penetration is the display-side derivation.
    table["penetration_mm"] = -table["signed_depth_mm"]

    print(f"\n{len(table):,} rows over "
          f"{table['frame_index'].nunique():,} contacting frames")
    return table, meta


def load_forearm(ply_path: Path, meta: dict[str, str]) -> np.ndarray:
    """Return the reference forearm vertices, (V, 3) in millimetres.

    ``vertex_id`` indexes this array directly, in file order. The vertex count
    is the proof that it is the same PLY the ids were assigned against — a
    forearm deduplicated at a different epsilon renumbers every vertex, and the
    ids would then point somewhere else with no visible error.
    """
    import open3d as o3d

    verts = np.asarray(o3d.io.read_point_cloud(str(ply_path)).points)

    expected = int(meta["reference_ply_vertex_count"])
    if len(verts) != expected:
        raise ValueError(
            f"{ply_path.name} has {len(verts)} vertices but the parquet was "
            f"written against a PLY of {expected}. These vertex_ids do not "
            f"address this mesh."
        )
    return verts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_frame(ax, table, frames, position, forearm, limits, vmax, space):
    """Redraw the axes for one frame. ``position`` indexes ``frames``."""
    ax.clear()

    frame_index = frames[position]
    rows = table[table["frame_index"] == frame_index]

    if forearm is not None:
        ax.scatter(forearm[:, 0], forearm[:, 1], forearm[:, 2],
                   s=0.4, c="#C8CFD5", alpha=0.25, linewidths=0, zorder=1)

    ax.scatter(rows["x"], rows["y"], rows["z"],
               c=rows["penetration_mm"], cmap="inferno",
               vmin=0.0, vmax=vmax, s=16, linewidths=0, zorder=2)

    (xlo, xhi), (ylo, yhi), (zlo, zhi) = limits
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_zlim(zlo, zhi)
    ax.set_box_aspect((xhi - xlo, yhi - ylo, zhi - zlo))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")

    ax.set_title(
        f"frame {frame_index}   t = {rows['time_s'].iloc[0]:.3f} s   "
        f"{len(rows)} vertices   peak {rows['penetration_mm'].max():.2f} mm\n"
        f"[{position + 1}/{len(frames)}]  space: {space}",
        fontsize=10,
    )


def render(parquet: Path, args) -> None:
    """Render one block. PNGs go to <save>/<block-order-NN>/."""
    print(f"\n=== {block_tag(parquet)} ===")
    table, meta = load_depth_field(parquet)
    forearm = load_forearm(args.forearm, meta) if args.forearm else None
    save_dir = (args.save / block_tag(parquet)) if args.save else None

    # The contacting frames, in order. Frames absent here had no contact at all;
    # there is nothing to draw and nothing to interpolate. The contact patch
    # changes membership frame to frame, so row i of frame n is NOT the same
    # vertex as row i of frame n+1 — never ffill or interpolate this field.
    frames = np.sort(table["frame_index"].unique())
    if args.max_frames:
        frames = frames[:args.max_frames]

    # Fix the axes and the colour scale across every frame, so that what moves
    # on screen is the touch and not the viewport.
    if forearm is not None:
        cloud = forearm
    else:
        cloud = table[["x", "y", "z"]].to_numpy()
    pad = 5.0
    limits = [(cloud[:, i].min() - pad, cloud[:, i].max() + pad) for i in range(3)]
    vmax = float(np.percentile(table["penetration_mm"], 99))

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # One colour bar for the whole session, not one per frame: the scale is
    # fixed above so that colour means the same depth in every frame.
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0.0, vmax), cmap="inferno"),
        ax=ax, shrink=0.62, pad=0.11,
    )
    bar.set_label("penetration (mm)   [= -signed_depth_mm]", fontsize=9)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        for position in range(len(frames)):
            draw_frame(ax, table, frames, position, forearm, limits, vmax,
                       meta["coordinate_space"])
            out = save_dir / f"frame_{frames[position]:05d}.png"
            fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {len(frames)} PNGs to {save_dir}")
        return

    state = {"position": 0}

    def on_key(event):
        if event.key in ("right", "n", " "):
            state["position"] = min(state["position"] + 1, len(frames) - 1)
        elif event.key in ("left", "p"):
            state["position"] = max(state["position"] - 1, 0)
        elif event.key in ("q", "escape"):
            plt.close(fig)
            return
        else:
            return
        draw_frame(ax, table, frames, state["position"], forearm, limits, vmax,
                   meta["coordinate_space"])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_frame(ax, table, frames, 0, forearm, limits, vmax,
               meta["coordinate_space"])
    print("\nleft / right arrows to step, q to quit")
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet", type=Path, nargs="*",
                    help="depth-field parquet(s) (default: every block in "
                         f"{DEFAULT_PARQUET_DIR.name}/)")
    ap.add_argument("--forearm", type=Path,
                    default=DEFAULT_FOREARM if DEFAULT_FOREARM.exists() else None,
                    help="reference forearm PLY, drawn as context")
    ap.add_argument("--save", type=Path, default=DEFAULT_SAVE,
                    help="write PNGs into <dir>/<block-order-NN>/")
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--show", action="store_true",
                    help="open the interactive stepper instead of writing PNGs")
    ap.add_argument("--no-forearm", action="store_true",
                    help="skip the forearm context cloud")
    args = ap.parse_args()
    if args.no_forearm:
        args.forearm = None

    parquets = args.parquet or sorted(DEFAULT_PARQUET_DIR.glob(PARQUET_GLOB))
    if not parquets:
        raise SystemExit(f"no {PARQUET_GLOB} found in {DEFAULT_PARQUET_DIR}")
    if args.show:  # --save has a default now, so interactive needs an opt-in
        args.save = None
        parquets = parquets[:1]  # the stepper handles one block at a time

    for parquet in parquets:
        render(parquet, args)


if __name__ == "__main__":
    main()
