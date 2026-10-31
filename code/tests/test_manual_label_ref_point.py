"""Tests for the manual forearm reference point (black ink dot).

Covers the labelling helpers (dot centre, 3D lookup), the manual step's
save / skip / cancel logic with the GUI stubbed out, and the postprocessing
export, whose transform chain is checked against the pipeline's own CSV
transforms (ICP schedule, PCA calibration, RF translation).
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import pytest

from postprocessing.xyz_reference_from_gestures.calibration_pca_engine import (
    CalibrationResult,
    PCACalibrationEngine,
)
from preprocessing.forearm_extraction.data_access.manual_label_ref_point_filehandler import (
    ManualLabelRefPointFileHandler,
)
from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    get_transform_schedule,
    transform_spatial_columns_in_place,
    transform_spatial_columns_scheduled,
)

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(relative_path: str, module_name: str):
    """Import a pipeline script by path, skipping its package __init__ (GUI/SDK imports)."""
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPTS / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


label_step = _load_script(
    "_3_preprocessing/_3_forearm_extraction/define_manual_label_ref_point.py",
    "define_manual_label_ref_point",
)
export_step = _load_script(
    "_5_postprocessing/export_manual_label_ref_point.py",
    "export_manual_label_ref_point",
)

SESSION = "2022-06-14_ST13-01"
SKIN_BGR = (150, 170, 205)
STICKER_BLUE = ["sticker_blue_position_x", "sticker_blue_position_y", "sticker_blue_position_z"]
XYZ_COLUMNS = [
    f"{prefix}_{axis}"
    for prefix in ("contact_location", "sticker_blue_position", "sticker_green_position", "sticker_yellow_position")
    for axis in "xyz"
]
P_CAM = np.array([12.5, -40.0, 655.0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skin_frame(h: int = 120, w: int = 160, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.empty((h, w, 3), dtype=int)
    frame[:] = SKIN_BGR
    return np.clip(frame + rng.integers(-6, 7, size=frame.shape), 0, 255).astype(np.uint8)


def _plane_cloud(h: int = 60, w: int = 80) -> np.ndarray:
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([u - 40, v - 30, 600 + 0.5 * u], axis=-1).astype(np.float32)


def _rigid(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = q
    T[:3, 3] = rng.normal(scale=50, size=3)
    return T


def _transforms() -> dict:
    stem = f"{SESSION}_semicontrolled_block-order"
    return {
        "mode": "reference",
        "canonical_key": f"{stem}01_kinect:3",
        "transforms": {
            f"{stem}01_kinect:3": {"matrix_4x4": _rigid(1).tolist(), "fitness": 1.0},
            f"{stem}01_kinect:500": {"matrix_4x4": _rigid(2).tolist(), "fitness": 0.9},
            f"{stem}02_kinect:40": {"matrix_4x4": _rigid(3).tolist(), "fitness": 0.9},
        },
    }


def _calibration() -> CalibrationResult:
    c, s = np.cos(0.7), np.sin(0.7)
    R2 = np.eye(3)
    R2[:2, :2] = [[c, -s], [s, c]]
    return CalibrationResult(
        mean_1=np.array([1.5, -2.0, 0.5]), R1=_rigid(9)[:3, :3],
        mean_2=np.array([0.75, -1.25]), R2=R2,
    )


def _row(point, frame_index: int) -> pd.DataFrame:
    """One block-CSV row carrying *point* in every XYZ column group."""
    row = {col: float(point[i % 3]) for i, col in enumerate(XYZ_COLUMNS)}
    row["contact_points"] = "[]"
    row["frame_index"] = frame_index
    return pd.DataFrame([row])


def _pipeline_point(point, block_id: str, frame_index: int, transforms, calib, rf_center):
    """What postprocessing steps 1, 4 and 5 do to a sticker position in that block's CSV row."""
    stem = f"{SESSION}_semicontrolled_{block_id}".replace("block-order-", "block-order")
    df = _row(point, frame_index)
    if transforms is not None:  # step 1: schedule over the whole block, as apply_icp_registration builds it
        transform_spatial_columns_scheduled(df, get_transform_schedule(transforms["transforms"], stem, 10_000))
    pca = PCACalibrationEngine.apply_full_transform(df[STICKER_BLUE].to_numpy(dtype=float), calib)  # step 4
    if rf_center is None:  # step 5 pass-through
        return pca[0]
    T = np.eye(4)
    T[:3, 3] = -np.asarray(rf_center)
    return transform_spatial_columns_in_place(_row(pca[0], frame_index), T)[STICKER_BLUE].to_numpy()[0]


# ---------------------------------------------------------------------------
# Labelling helpers
# ---------------------------------------------------------------------------

def test_refine_dot_center_finds_a_dot_off_the_square_centre():
    frame = _skin_frame()
    cv2.circle(frame, (70, 50), 4, (30, 30, 30), -1)
    u, v, method = label_step.refine_dot_center(frame, {"x": 55, "y": 40, "width": 40, "height": 30})
    assert method == "dark_blob_centroid"
    assert (u, v) == pytest.approx((70, 50), abs=0.3)


def test_refine_dot_center_falls_back_to_the_square_centre_without_a_dot():
    u, v, method = label_step.refine_dot_center(_skin_frame(), {"x": 20, "y": 30, "width": 41, "height": 21})
    assert method == "roi_center"
    assert (u, v) == (40.0, 40.0)


def test_refine_dot_center_ignores_dark_regions_touching_the_square_edge():
    frame = _skin_frame()
    frame[:, :62] = (40, 40, 40)  # shadow entering the square from the left
    cv2.circle(frame, (80, 60), 3, (25, 25, 25), -1)
    u, v, method = label_step.refine_dot_center(frame, {"x": 55, "y": 45, "width": 40, "height": 30})
    assert method == "dark_blob_centroid"
    assert (u, v) == pytest.approx((80, 60), abs=0.3)


def test_lookup_camera_xyz_reads_the_pixel_like_the_sticker_extraction():
    cloud = _plane_cloud()
    xyz, method = label_step.lookup_camera_xyz(cloud, 10.4, 20.6, {"x": 5, "y": 15, "width": 10, "height": 10})
    assert method == "pixel_lookup"
    np.testing.assert_allclose(xyz, cloud[21, 10])  # nearest pixel


def test_lookup_camera_xyz_uses_the_square_median_when_the_ink_has_no_depth():
    cloud = _plane_cloud()
    cloud[18:23, 8:13] = np.nan
    xyz, method = label_step.lookup_camera_xyz(cloud, 10, 20, {"x": 5, "y": 15, "width": 10, "height": 10})
    patch = cloud[15:25, 5:15].reshape(-1, 3)
    assert method == "roi_median"
    np.testing.assert_allclose(xyz, np.median(patch[np.isfinite(patch).all(axis=1)], axis=0))


def test_lookup_camera_xyz_raises_without_any_depth_in_the_square():
    cloud = _plane_cloud()
    cloud[15:25, 5:15] = 0  # SDK zeros mean "no depth"
    with pytest.raises(ValueError, match="No valid depth"):
        label_step.lookup_camera_xyz(cloud, 10, 20, {"x": 5, "y": 15, "width": 10, "height": 10})


# ---------------------------------------------------------------------------
# Manual step (GUI stubbed)
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_gui(monkeypatch):
    frame = _skin_frame(h=60, w=80)
    cv2.circle(frame, (30, 25), 3, (25, 25, 25), -1)
    cloud = _plane_cloud()
    calls = {"select_frame": 0}

    def select_frame(rgb_video_path, start_frame):
        calls["select_frame"] += 1
        return 12

    monkeypatch.setattr(label_step, "_SESSIONS_LABELLED_THIS_RUN", set())
    monkeypatch.setattr(label_step, "_select_frame", select_frame)
    monkeypatch.setattr(label_step, "read_kinect_frame", lambda path, index: (frame, cloud))
    monkeypatch.setattr(label_step, "_draw_square",
                        lambda image, title, predefined=None: {"x": 20, "y": 15, "width": 20, "height": 20})
    return calls, cloud


def test_manual_step_saves_one_label_per_session(tmp_path, stub_gui):
    calls, cloud = stub_gui
    out = ManualLabelRefPointFileHandler.default_path(tmp_path, SESSION)

    saved = label_step.define_manual_label_ref_point(
        Path("block1.mp4"), Path("block1.mkv"), SESSION, "block-order-01", out)

    assert saved == out
    label = ManualLabelRefPointFileHandler.load(out)
    assert (label["block_id"], label["frame_index"], label["pixel_method"]) == ("block-order-01", 12, "dark_blob_centroid")
    assert label["pixel_uv"] == pytest.approx([30, 25], abs=0.3)
    np.testing.assert_allclose(label["xyz_camera_mm"], cloud[25, 30])

    # The next block of the same session does not ask again, even when forced.
    label_step.define_manual_label_ref_point(
        Path("block2.mp4"), Path("block2.mkv"), SESSION, "block-order-02", out, force_processing=True)
    assert calls["select_frame"] == 1


def test_manual_step_saves_nothing_on_cancel_and_skips_a_labelled_session(tmp_path, stub_gui, monkeypatch):
    out = ManualLabelRefPointFileHandler.default_path(tmp_path, SESSION)

    monkeypatch.setattr(label_step, "_select_frame", lambda *args: None)  # frame selector closed
    assert label_step.define_manual_label_ref_point(Path("b.mp4"), Path("b.mkv"), SESSION, "block-order-01", out) is None
    assert not out.exists()

    ManualLabelRefPointFileHandler.save(out, {"session_id": SESSION})
    monkeypatch.setattr(label_step, "_select_frame", lambda *args: pytest.fail("asked again"))
    assert label_step.define_manual_label_ref_point(Path("b.mp4"), Path("b.mkv"), SESSION, "block-order-01", out) == out


# ---------------------------------------------------------------------------
# Export: transform chain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_id, frame_index", [
    ("block-order-01", 1),    # before the block's first snapshot, no earlier block: untouched
    ("block-order-01", 3),
    ("block-order-01", 499),
    ("block-order-01", 501),
    ("block-order-02", 10),   # earlier block's last snapshot until this block's own
    ("block-order-02", 40),
    ("block-order-03", 7),    # block without snapshots: previous block's
])
def test_registration_matches_what_the_icp_stage_applies(block_id, frame_index):
    transforms = _transforms()
    stem = f"{SESSION}_semicontrolled_{block_id}".replace("block-order-", "block-order")
    df = transform_spatial_columns_scheduled(
        _row(P_CAM, frame_index), get_transform_schedule(transforms["transforms"], stem, 10_000))

    T, _ = export_step.registration_matrix_at(transforms, SESSION, block_id, frame_index)
    identity_pca = CalibrationResult(np.zeros(3), np.eye(3), np.zeros(2), np.eye(3))
    registered = export_step.transform_camera_point(P_CAM, T, identity_pca, None)["registered"]

    np.testing.assert_allclose(registered, df[STICKER_BLUE].to_numpy()[0], atol=1e-9)


def test_registration_is_identity_for_a_single_forearm_session():
    T, start = export_step.registration_matrix_at(None, SESSION, "block-order-01", 100)
    assert start is None
    np.testing.assert_array_equal(T, np.eye(4))


@pytest.mark.parametrize("rf_center", [[3.5, -12.0, 7.25], None])
def test_full_chain_matches_the_pipeline_csv_transforms(rf_center):
    transforms, calib = _transforms(), _calibration()
    T, _ = export_step.registration_matrix_at(transforms, SESSION, "block-order-01", 620)

    got = export_step.transform_camera_point(P_CAM, T, calib, rf_center)["rf_centered"]

    expected = _pipeline_point(P_CAM, "block-order-01", 620, transforms, calib, rf_center)
    np.testing.assert_allclose(got, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# Export: files
# ---------------------------------------------------------------------------

def _write_session(tmp_path, *, rf_center, with_transforms=True):
    """Label, transforms, PCA JSON, RF JSON and an RF-centred forearm PLY for one session."""
    config = SimpleNamespace(session_id=SESSION,
                             session_processed_output_dir=tmp_path / "2_processed" / "kinect" / SESSION)
    merged = tmp_path / "3_merged" / "1_kinect_and_nerve" / SESSION

    ManualLabelRefPointFileHandler.save(
        ManualLabelRefPointFileHandler.default_path(config.session_processed_output_dir, SESSION),
        {"label": "black_dot", "session_id": SESSION, "block_id": "block-order-01", "frame_index": 620,
         "xyz_camera_mm": P_CAM.tolist(), "pixel_uv": [950.0, 880.0],
         "pixel_method": "dark_blob_centroid", "xyz_method": "pixel_lookup"},
    )
    transforms = _transforms() if with_transforms else None
    if transforms is not None:
        path = config.session_processed_output_dir / "forearm_pointclouds" / f"{SESSION}_registration_transforms.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(transforms))

    calib = _calibration()
    pca_path = merged / "blocks_pca_calibrated" / "pca-xyz_transformation-matrices.json"
    pca_path.parent.mkdir(parents=True, exist_ok=True)
    pca_path.write_text(json.dumps(calib.to_dict()))
    rf_path = merged / "rf_center_origin.json"
    rf_path.write_text(json.dumps({"status": "ok" if rf_center else "no_cluster_found", "rf_center": rf_center}))

    expected = _pipeline_point(P_CAM, "block-order-01", 620, transforms, calib, rf_center)
    grid = np.stack(np.meshgrid(*[np.arange(-6.0, 7.0, 3.0)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    vertices = expected + grid + np.array([1.0, 0.0, 0.0])  # nearest vertex sits 1 mm from the point
    ply_path = merged / "forearm_rf_centered" / f"{SESSION}_forearm.ply"
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(ply_path), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices)))
    return config, merged, pca_path, rf_path, ply_path, expected


def test_export_writes_the_point_in_the_rf_centred_frame(tmp_path):
    config, merged, pca_path, rf_path, ply_path, expected = _write_session(tmp_path, rf_center=[3.5, -12.0, 7.25])
    out = merged / "manual_label_ref_point.json"

    export_step.export_manual_label_ref_point([config], pca_path, rf_path, ply_path, out)

    result = json.loads(out.read_text())
    assert (result["status"], result["coordinate_space"], result["units"]) == ("ok", "rf_centered", "mm")
    np.testing.assert_allclose(result["ref_point"], expected, atol=1e-6)
    np.testing.assert_allclose(result["points_mm"]["kinect_camera"], P_CAM)
    assert result["distance_to_forearm_mm"] == pytest.approx(1.0, abs=1e-3)
    assert result["registration"]["segment_start_frame"] == 500
    assert result["label_source"]["frame_index"] == 620

    # Up to date: a second run leaves the file alone.
    mtime = out.stat().st_mtime_ns
    export_step.export_manual_label_ref_point([config], pca_path, rf_path, ply_path, out)
    assert out.stat().st_mtime_ns == mtime


def test_export_stays_in_the_pca_frame_when_no_rf_was_found(tmp_path):
    config, merged, pca_path, rf_path, ply_path, expected = _write_session(tmp_path, rf_center=None)
    out = merged / "manual_label_ref_point.json"

    export_step.export_manual_label_ref_point([config], pca_path, rf_path, ply_path, out)

    result = json.loads(out.read_text())
    assert (result["coordinate_space"], result["rf_status"]) == ("pca_calibrated", "no_cluster_found")
    np.testing.assert_allclose(result["ref_point"], expected, atol=1e-6)
    np.testing.assert_allclose(result["ref_point"], result["points_mm"]["pca_calibrated"])


def test_export_without_registration_transforms_uses_the_camera_frame(tmp_path):
    config, merged, pca_path, rf_path, ply_path, expected = _write_session(
        tmp_path, rf_center=[3.5, -12.0, 7.25], with_transforms=False)
    out = merged / "manual_label_ref_point.json"

    export_step.export_manual_label_ref_point([config], pca_path, rf_path, ply_path, out)

    result = json.loads(out.read_text())
    assert result["registration"]["transforms_file"] is None
    np.testing.assert_allclose(result["points_mm"]["registered"], P_CAM)
    np.testing.assert_allclose(result["ref_point"], expected, atol=1e-6)


def test_export_fails_for_a_session_without_a_label(tmp_path):
    config = SimpleNamespace(session_id=SESSION, session_processed_output_dir=tmp_path / "2_processed")
    with pytest.raises(FileNotFoundError, match="define_manual_label_ref_point"):
        export_step.export_manual_label_ref_point(
            [config], tmp_path / "pca.json", tmp_path / "rf.json", tmp_path / "forearm.ply", tmp_path / "out.json")
