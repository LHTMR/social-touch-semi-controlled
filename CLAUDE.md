# CLAUDE.md — contact_point_interpolation

Context for Claude Code sessions opened in this folder. Read this first.

## What this repo is
Standalone tool that fills the ~33-row gaps in `contact_points` created when the
kinect contact data is merged with the higher-rate neural recording. It morphs the
contact patch from one kinect frame to the next **along the forearm point-cloud
surface** (kNN-graph geodesics), so no interpolated point leaves the forearm.
`code/interpolate_contact_points.py` is the tool; `interpolate_test.py` /
`interpolate_video.py` are verification.

## Git setup (important — non-obvious)
- This is its **own git repo**, but `origin` points at a branch of the lab repo:
  - remote `origin` = `https://github.com/LHTMR/social-touch-semi-controlled.git`
  - branch = **`dev-haonan-interpolate-contact-point`** — an **orphan branch** that
    contains *only this repo* (unrelated history to the lab pipeline). This is
    intentional: the developer applies it to their workflow.
- **Never touch `main` or `dev`** of that lab repo. Only push this one branch.
- Commit + push from here:
  ```bash
  git add <files>
  git commit -m "..."
  git push origin dev-haonan-interpolate-contact-point
  ```
- Local git identity is set for this repo: `qsm9za <qsm9za@virginia.edu>`.
- `core.autocrlf=true` on this machine, so binaries are protected by
  `.gitattributes` (`*.ply *.obj *.pkl *.mp4 *.png` = `binary`). Keep that — without
  it, git line-ending conversion **corrupts** the point-cloud `.ply` files.
- The lab repo moved from `SDAMcIntyre/…` to `LHTMR/…`; the old URL still redirects.

## Layout & what's committed
```
code/    surface_morph.py  = shared library: kNN graph, all-pairs geodesics,
           clustering, contact-mode detection, cluster pairing, morph_paths().
           Imported by interpolate_contact_points.py (morphs positions),
           interpolate_depth_field.py (morphs per-vertex depth), and the
           verification scripts. Put anything both tools need here.
data/    2022-06-14_ST13-01/                            example session
           blocks/              4x *_merged_data.csv           <- what the tool reads
           blocks_rf_centered/  4x *_merged_data_pca-xyz.csv   + *_contact_depth_field_pca-xyz.parquet
           forearm_pointclouds/ *_unified_registered.ply (the scaffold)
plot/    verification PNGs                               (committed, small)
```
The block CSVs used to sit directly in the session folder; they moved into
`blocks/` on 2026-09-17, which is why `--data-dir` defaults to
`data/2022-06-14_ST13-01/blocks`.
Git-ignored (regenerable — do NOT commit, they are large): `output/` (interpolated
CSVs, ~1.5 GB) and `plot/videos/` (~11 MB of mp4s).

## Environment
Python: `D:/conda_envs/social-touch/python.exe` (python 3.10, numpy 2.0, scipy 1.15,
matplotlib 3.10, pyarrow 23, open3d 0.19, opencv 4.13, Pillow 11).
Package set is pinned in `environment.yml`.

**numpy must stay a pip wheel — do not `conda install numpy` here.** Everything in
this env except numpy is a pip wheel with its own bundled OpenBLAS. The conda-forge
numpy instead links the env's shared `libblas` shim, and that pairing had drifted:
numpy 2.0.0 (built against `libblas >=3.9.0`) was running against `libblas 3.11.0` →
`mkl 2025.3.0`. The version constraint still matched so conda allowed it, but the
binary segfaulted (exit 127, no traceback) the moment it called BLAS — `a @ b`,
2-D `np.dot`, `np.linalg.*`, and therefore **every** matplotlib transform, 2-D and
3-D alike. Telltale: `scipy.linalg.inv` worked while `np.linalg.inv` crashed.

Fixed 2026-09-04 with `pip install --force-reinstall --no-deps numpy==2.0.0`, which
brings numpy's own OpenBLAS. Verified afterwards: matmul, `np.linalg`, matplotlib 2-D
and 3-D all fine, and the interpolation output is unchanged (12,123 rows re-checked
cell-for-cell, both single and multi mode). To roll back:
`conda install -n social-touch numpy=2.0.0=py310h1ec8c79_0`.

**matplotlib now works**, but the verification scripts still render with **Pillow**
(plots) and **OpenCV** (videos, `mp4v` — `avc1`/H.264 fails, no OpenH264 dll), with 3D
projection hand-rolled element-wise. That code predates the fix and works; it is not
worth rewriting. New scripts may use matplotlib freely.

## Run
```bash
PY="D:/conda_envs/social-touch/python.exe"
# interpolate (all 4 example blocks) — ~9 min, all-pairs geodesics + dense patches
$PY code/interpolate_contact_points.py \
    --ply data/2022-06-14_ST13-01/forearm_pointclouds/2022-06-14_ST13-01_unified_registered.ply \
    --contacts data/2022-06-14_ST13-01/blocks/*_merged_data.csv --outdir output --max-gap 50
$PY code/interpolate_test.py     # plot/*.png  (3x2 per block: original vs interpolated)
$PY code/interpolate_video.py    # plot/videos/*.mp4  (3D evolution, original vs interp)
```

## Key facts (verified)
- **Scaffold PLY** = `…_unified_registered.ply` (per-experiment, 3718 pts, has normals).
  Contacts in the `*_merged_data.csv` are in the **same registered frame** — they snap
  to it at ~0.4–0.6 mm median (checked). The per-block `…_frames_*.ply` are per-frame
  extractions, not the scaffold.
- **`--max-gap 50`**: the single-kinect-frame gap is 33 **or** 34 rows (jitter), so the
  default 33 is too strict. A gap ≥ ~66 means an intervening kinect frame was empty →
  not interpolated (correct).
- **Contact mode**: blocks are either `one finger tip` (1 contact patch/frame) or
  `whole hand` (2–4 finger patches with gaps). The tool clusters contacts (single
  linkage, 4 mm) and confines the morph to *corresponding* patches, otherwise points
  drift between fingers (was 7.9% of matches on block-02, median 12 mm). Mode is
  detected **once per block** from the first 200 anchor frames — not per frame, since
  a whole-hand frame collapses to 1 patch at tap onset/offset. `single` mode is the
  pre-cluster code path, bit-for-bit (verified on 12.6k rows). Detection agrees with
  `contact_area_metadata` on all 4 example blocks; that column is logged as a
  cross-check but never drives the algorithm. Override: `--contact-mode single|multi`.
- **`blocks_rf_centered/`** (added 2026-09-17) — run `code/inspect_rf_centered.py` to
  re-derive all of this. Per block: a `*_merged_data_pca-xyz.csv` and a
  `*_contact_depth_field_pca-xyz.parquet`.
  - The CSV has the **same 32 columns** as `blocks/`, same row count, nothing added or
    removed. Only the **13 spatial** columns differ (3 `contact_location_*`, 9
    `sticker_*`, and `contact_points`); the other 19 — including `contact_depth`,
    `contact_area`, all timing, nerve and metadata columns — are byte-identical.
  - Those coordinates are **not** the old ones rotated: pairwise distances within a
    frame shift by ~0.4–1.0 mm median (up to 8 mm), and 11–32 % of the new
    `contact_points` are exact duplicates. The contacts were re-projected onto a
    deduplicated forearm (`dedup_epsilon` 0.5), not just moved to a new frame.
  - **The tool cannot read these yet.** They sit ~930 mm from
    `_unified_registered.ply`; `vertex_id` indexes `2022-06-14_ST13-01_forearm.ply`
    (1807 vertices, `coordinate_space=pca_calibrated`) which is **not in this repo**.
    The old `blocks/` CSVs still snap to the registered PLY at 0.56 mm median.
  - Despite the folder name, all four blocks declare `coordinate_space=pca_calibrated`,
    not `rf_centered` — read the metadata, never the directory name.
- **Depth field parquet** (`schema_version 2`, `sign_convention=negative_is_penetrating`).
  Long form, one row per contact sample: `frame_index, time_s, x, y, z,
  signed_depth_mm, vertex_id`. It agrees exactly with the CSV — same contacting
  frames, same per-frame point counts, same positions within the CSV's 0.1 mm
  rounding — and `contact_depth` in the CSV is *exactly* `-min(signed_depth_mm)` per
  frame (residual 0.000000 mm). So the parquet is the CSV's scalar depth resolved per
  contact point, plus full float precision and vertex identity.
  - **Caveat:** `(frame_index, vertex_id)` is **not unique**, contradicting the
    "one row per (frame, contact vertex)" contract in `plot_contact_depth_field.py`.
    8–19 % of pairs repeat (up to 5x), at identical x/y/z but **different** depths
    (spread up to 7 mm). Reading it as one-value-per-vertex silently takes whichever
    row comes last — the viewer overplots them. Aggregate explicitly (e.g. `min`)
    before treating `vertex_id` as a key.
- **Depth field interpolation** (`interpolate_depth_field.py`): the depth field has the
  same holes as `contact_points` — `contact_depth`/`contact_area` sit on exactly the
  3.0% of rows carrying a `frame_index`, gaps of only 33/34 rows. It reuses the same
  morph; `morph_paths()` returns paths whose `path[0]` is the source vertex and
  `path[-1]` the target, so depth blends linearly between the endpoints while the
  position follows the geodesic. Needs the **deduplicated** `<session>_forearm.ply`
  (1807 verts), not `_unified_registered.ply`. Its gap rule is exact rather than
  heuristic — adjacent kinect frames are `Δframe_index == 1`, no `--max-gap` guess —
  and on all 4 blocks it reproduces the contact tool's span structure exactly
  (28515/147, 56361/156, 89399/24, 32007/23). Duplicate `(frame, vertex)` rows are
  reduced by `--aggregate` (default `min` = deepest), which makes the field's per-frame
  peak equal the CSV's `contact_depth` to 0.000000 mm. ~213 MB for 4 blocks.
- **Output** = 2 columns: `time_nerve` (dense, for aligning to the nerve signal) +
  `contact_points_interpolated`. Same row count as input. Anchor (measured) rows keep
  the **original** contacts; only interpolated in-between rows are snapped to the cloud.
  Coords rounded to 1 decimal (matches input; source is genuinely 0.1 mm, not integer).
