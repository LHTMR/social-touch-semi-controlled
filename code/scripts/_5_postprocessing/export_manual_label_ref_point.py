"""Postprocessing step 7: export the manually labelled forearm reference point.

The black-dot reference point labelled in the manual step
``define_manual_label_ref_point`` sits in one block's Kinect colour-camera
frame. This step moves it through exactly the transforms that block's CSV rows
and the forearm PLY receive, so it overlays ``forearm_rf_centered/`` and
``blocks_rf_centered/``:

1. ICP registration — the matrix ``apply_icp_registration`` applies to that
   block's rows at the labelled frame (identity for single-forearm sessions).
2. PCA calibration — ``blocks_pca_calibrated/pca-xyz_transformation-matrices.json``.
3. RF centring — minus ``rf_center`` from ``rf_center_origin.json``; no shift
   when the RF was not found, matching the pass-through of the data.

Writes ``<session_merged_output_dir>/manual_label_ref_point.json``.
"""

import bisect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from postprocessing.xyz_reference_from_gestures.calibration_pca_engine import (
    CalibrationResult,
    PCACalibrationEngine,
)
from preprocessing.forearm_extraction.data_access.manual_label_ref_point_filehandler import (
    ManualLabelRefPointFileHandler,
)
from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    apply_rigid_transform,
    get_transform_schedule,
)
from primary_processing import KinectConfig
from utils.should_process_task import should_process_task

logger = logging.getLogger(__name__)

# A dot on the skin lands within a few mm of a forearm vertex; farther than
# this usually means a wrong frame, a moved arm or a bad registration.
WARN_DISTANCE_TO_FOREARM_MM = 10.0


def registration_matrix_at(
    transforms_data: Optional[Dict[str, Any]],
    session_id: str,
    block_id: str,
    frame_index: int,
) -> Tuple[np.ndarray, Optional[int]]:
    """The 4x4 ICP matrix ``apply_icp_registration`` applies to this block's rows at *frame_index*.

    Same block stem and transform schedule as the stage; the active segment is
    the last one starting at or before the frame. Identity when nothing applies
    (no transforms file, or no snapshot at or before this frame), exactly like
    the CSV rows.

    Returns:
        ``(T_4x4, start frame of the active segment or None)``.
    """
    if transforms_data is None:
        return np.eye(4), None
    video_stem = f"{session_id}_semicontrolled_{block_id}".replace("block-order-", "block-order")
    schedule = get_transform_schedule(transforms_data["transforms"], video_stem, frame_index)
    i = bisect.bisect_right([start for start, _ in schedule], frame_index) - 1
    if i < 0:
        return np.eye(4), None
    start, T = schedule[i]
    return np.asarray(T, dtype=np.float64), int(start)


def transform_camera_point(
    p_cam: Sequence[float],
    T_registration: np.ndarray,
    pca_calibration: CalibrationResult,
    rf_center: Optional[Sequence[float]],
) -> Dict[str, np.ndarray]:
    """Carry one camera-frame point through registration, PCA and RF centring.

    Returns the point in every frame along the way, keyed
    ``kinect_camera``, ``registered``, ``pca_calibrated``, ``rf_centered``.
    """
    p = np.asarray(p_cam, dtype=np.float64).reshape(1, 3)
    p_reg = apply_rigid_transform(p, T_registration)
    p_pca = PCACalibrationEngine.apply_full_transform(p_reg.copy(), pca_calibration)
    p_rf = p_pca - np.asarray(rf_center, dtype=np.float64) if rf_center is not None else p_pca.copy()
    return {
        "kinect_camera": p[0],
        "registered": p_reg[0],
        "pca_calibrated": p_pca[0],
        "rf_centered": p_rf[0],
    }


def _load_registration_transforms(path: Path) -> Optional[Dict[str, Any]]:
    """Registration transforms JSON, or None for a single-forearm session."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_manual_label_ref_point(
    session_configs: List[KinectConfig],
    pca_calibration_path: Path,
    rf_origin_path: Path,
    forearm_ply_path: Path,
    output_path: Path,
    *,
    force_processing: bool = False,
) -> Path:
    """Write the session's reference point in the frame of ``forearm_rf_centered/``.

    Args:
        session_configs: KinectConfig objects of the session; the label and the
            registration transforms live in its ``session_processed_output_dir``.
        pca_calibration_path: ``blocks_pca_calibrated/pca-xyz_transformation-matrices.json``.
        rf_origin_path: ``rf_center_origin.json``.
        forearm_ply_path: ``forearm_rf_centered/<session>_forearm.ply``; the point
            is matched to its nearest vertex as a sanity check.
        output_path: ``<session_merged_output_dir>/manual_label_ref_point.json``.
        force_processing: Re-run even if the output is up-to-date.

    Returns:
        *output_path*.

    Raises:
        FileNotFoundError: the session has no manual label yet (so the task
            shows FAILURE), or one of the transform files is missing.
    """
    first_config = session_configs[0]
    session_id = first_config.session_id
    processed_dir = first_config.session_processed_output_dir
    label_path = ManualLabelRefPointFileHandler.default_path(processed_dir, session_id)
    transforms_path = processed_dir / "forearm_pointclouds" / f"{session_id}_registration_transforms.json"

    label = ManualLabelRefPointFileHandler.load(label_path)
    if label.get("session_id") != session_id:
        raise ValueError(
            f"{label_path.name} belongs to session {label.get('session_id')!r}, not {session_id!r}."
        )

    input_paths = [label_path, pca_calibration_path, rf_origin_path, forearm_ply_path]
    if transforms_path.exists():
        input_paths.append(transforms_path)
    if not should_process_task(input_paths=input_paths, output_paths=[output_path], force=force_processing):
        logger.info("Manual reference point export up-to-date. Skipping.")
        return output_path

    transforms_data = _load_registration_transforms(transforms_path)
    frame_index = int(label["frame_index"])
    T_registration, segment_start = registration_matrix_at(
        transforms_data, session_id, label["block_id"], frame_index
    )
    with open(pca_calibration_path, "r", encoding="utf-8") as f:
        pca_calibration = CalibrationResult.from_dict(json.load(f))
    with open(rf_origin_path, "r", encoding="utf-8") as f:
        rf_origin = json.load(f)
    rf_center = rf_origin.get("rf_center")

    points = transform_camera_point(label["xyz_camera_mm"], T_registration, pca_calibration, rf_center)

    vertices = np.asarray(o3d.io.read_point_cloud(str(forearm_ply_path)).points, dtype=np.float64)
    if len(vertices) == 0:
        raise ValueError(f"Forearm PLY has no vertices: {forearm_ply_path}")
    distance, index = cKDTree(vertices).query(points["rf_centered"])
    if distance > WARN_DISTANCE_TO_FOREARM_MM:
        logger.warning(
            "Reference point is %.1f mm from the nearest forearm vertex — check the label "
            "(frame, square) and the registration.", distance,
        )

    result = {
        "status": "ok",
        "label": label.get("label", "black_dot"),
        "session_id": session_id,
        "ref_point": points["rf_centered"].tolist(),
        "coordinate_space": "rf_centered" if rf_center is not None else "pca_calibrated",
        "units": "mm",
        "same_frame_as": [f"forearm_rf_centered/{forearm_ply_path.name}", "blocks_rf_centered/*.csv"],
        "rf_status": rf_origin.get("status"),
        "rf_center": rf_center,
        "nearest_forearm_vertex": vertices[index].tolist(),
        "nearest_forearm_vertex_index": int(index),
        "distance_to_forearm_mm": float(distance),
        "points_mm": {frame: p.tolist() for frame, p in points.items()},
        "registration": {
            "transforms_file": transforms_path.name if transforms_data is not None else None,
            "segment_start_frame": segment_start,
            "matrix_4x4": T_registration.tolist(),
        },
        "label_source": {
            "file": str(label_path),
            "block_id": label["block_id"],
            "frame_index": frame_index,
            "pixel_uv": label.get("pixel_uv"),
            "pixel_method": label.get("pixel_method"),
            "xyz_method": label.get("xyz_method"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info(
        "Manual reference point for %s: [%.2f, %.2f, %.2f] mm (%s), %.2f mm from the forearm. Wrote %s",
        session_id, *points["rf_centered"], result["coordinate_space"], distance, output_path.name,
    )
    return output_path
