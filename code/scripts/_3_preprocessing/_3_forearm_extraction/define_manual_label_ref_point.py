"""Manual step: label the black reference dot on the touch receiver's forearm.

The afferent's receptive field is marked on the receiver's forearm with a black
ink dot. This step records where that dot is in 3D:

1. scrub the block's colour video to a frame where the dot is visible
   (the same frame selector as the hand-tracking ROI);
2. draw a square around the dot (the same square tool as the LED and
   hand-sticker ROIs);
3. the dot centre is the centroid of the dark blob inside the square, or the
   square centre when there is no clear dark blob;
4. its 3D position is read from that frame's depth point cloud with the
   single-pixel lookup the sticker 3D extraction uses (Kinect colour-camera
   frame, mm, parallax-corrected).

One label per session: the blocks' data and the forearm PLY are registered into
one frame per session, so the point is valid for every block. The
postprocessing step ``export_manual_label_ref_point`` carries it into the
RF-centred frame and writes ``<session_merged_output_dir>/manual_label_ref_point.json``.
"""

import logging
import tkinter as tk
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np

from preprocessing.common import FrameROISquare, KinectMKV, VideoFrameSelector
from preprocessing.forearm_extraction.data_access.manual_label_ref_point_filehandler import (
    ManualLabelRefPointFileHandler,
)
from preprocessing.stickers_analysis.xyz.core.xyz_extractor_centroid import CentroidPointCloudExtractor

logger = logging.getLogger(__name__)

# A dark blob only counts as the dot when it is clearly darker than the skin
# around it (grey levels, 0-255) and larger than a few pixels of noise.
MIN_DOT_CONTRAST = 25.0
MIN_DOT_AREA_PX = 4

# Sessions labelled during this run. The manual workflow loops over blocks, so
# with force_processing every block of the session would otherwise ask again.
_SESSIONS_LABELLED_THIS_RUN: Set[str] = set()


def refine_dot_center(frame_bgr: np.ndarray, roi: Dict[str, int]) -> Tuple[float, float, str]:
    """Pixel centre (u, v) of the black dot inside the drawn square.

    Otsu-thresholds the square and takes the centroid of the largest dark blob
    lying fully inside it. Falls back to the square centre when nothing inside
    is clearly darker than the skin, or when every dark region touches the
    square's edge (shadows, hair, or the square cutting through the dot).

    Args:
        frame_bgr: Full colour frame (BGR).
        roi: Square as returned by ``FrameROISquare``: ``x, y, width, height``.

    Returns:
        ``(u, v, method)`` in full-image pixels, with method
        ``"dark_blob_centroid"`` or ``"roi_center"``.
    """
    x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
    square_centre = (x + (w - 1) / 2.0, y + (h - 1) / 2.0, "roi_center")

    patch = frame_bgr[y:y + h, x:x + w]
    if patch.size == 0 or min(patch.shape[:2]) < 3:
        return square_centre

    grey = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, dark = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    is_dark = dark > 0
    if is_dark.all() or not is_dark.any():
        return square_centre
    if grey[~is_dark].mean() - grey[is_dark].mean() < MIN_DOT_CONTRAST:
        return square_centre

    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    patch_h, patch_w = grey.shape
    best = None
    for i in range(1, n_labels):
        left, top, comp_w, comp_h, area = stats[i]
        if area < MIN_DOT_AREA_PX:
            continue
        if left == 0 or top == 0 or left + comp_w >= patch_w or top + comp_h >= patch_h:
            continue
        if best is None or area > stats[best, cv2.CC_STAT_AREA]:
            best = i
    if best is None:
        return square_centre

    cx, cy = centroids[best]
    return x + float(cx), y + float(cy), "dark_blob_centroid"


def lookup_camera_xyz(
    point_cloud: np.ndarray, u: float, v: float, roi: Dict[str, int]
) -> Tuple[np.ndarray, str]:
    """3D position (mm, Kinect colour-camera frame) of pixel (u, v).

    Uses the single-pixel lookup of the sticker 3D extraction
    (``CentroidPointCloudExtractor.get_xyz_from_point_cloud``). Black ink can
    return no depth, so when that pixel is invalid the median of the valid
    points inside the square is used instead.

    Raises:
        ValueError: if the square holds no valid depth at all.
    """
    xyz = np.asarray(
        CentroidPointCloudExtractor.get_xyz_from_point_cloud(point_cloud, u, v), dtype=np.float64
    )
    if np.isfinite(xyz).all():
        return xyz, "pixel_lookup"

    x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
    patch = np.asarray(point_cloud[y:y + h, x:x + w], dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(patch).all(axis=1) & ~(patch == 0).all(axis=1)
    if not valid.any():
        raise ValueError(
            "No valid depth inside the square on this frame (the Kinect returned no 3D "
            "points there). Re-run the step and pick another frame."
        )
    return np.median(patch[valid], axis=0), "roi_median"


def read_kinect_frame(source_video_path: Path, frame_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """Colour image (BGR) and colour-aligned point cloud (mm) of one MKV capture.

    Sequential seek, so ``frame_index`` is the same capture index the colour
    MP4 and the sticker 3D extraction use.
    """
    with KinectMKV(source_video_path, seek_strategy="sequential") as mkv:
        frame = mkv[frame_index]
        color_bgr = frame.color
        point_cloud = frame.transformed_depth_point_cloud
        if color_bgr is None or point_cloud is None:
            raise ValueError(
                f"Frame {frame_index} of {Path(source_video_path).name} has no colour or no "
                f"depth. Re-run the step and pick another frame."
            )
        return np.array(color_bgr, copy=True), np.array(point_cloud, copy=True)


def _select_frame(rgb_video_path: Path, start_frame: int) -> Optional[int]:
    """Let the user scrub the colour video; returns the chosen frame or None."""
    cap = cv2.VideoCapture(str(rgb_video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {rgb_video_path}")
    root = tk.Tk()
    root.geometry("1x1+10000+10000")  # off-screen; withdraw() hides transient children on Windows
    try:
        selector = VideoFrameSelector(
            root, cap, current_frame_num=start_frame,
            title=f"Select a frame where the BLACK DOT on the receiver's forearm is visible — {rgb_video_path.name}",
        )
        return selector.select_frame()
    finally:
        root.destroy()
        cap.release()


def _draw_square(
    color_bgr: np.ndarray, window_title: str, predefined_roi: Optional[Dict[str, int]] = None
) -> Optional[Dict[str, int]]:
    """Square drawn by the user, or None when the window is closed without confirming."""
    roi_selector = FrameROISquare(
        color_bgr, is_rgb=False, window_title=window_title, predefined_roi=predefined_roi
    )
    roi_selector.run()
    if not roi_selector.roi_confirmed:
        return None
    rect = roi_selector.get_roi_data()
    if rect is None or rect["width"] < 1 or rect["height"] < 1:
        return None
    return {key: int(rect[key]) for key in ("x", "y", "width", "height")}


def define_manual_label_ref_point(
    rgb_video_path: Path,
    source_video_path: Path,
    session_id: str,
    block_id: str,
    output_path: Path,
    *,
    force_processing: bool = False,
) -> Optional[Path]:
    """Label the black reference dot once per session and save it.

    Saved JSON (``ManualLabelRefPointFileHandler``)::

        {
          "label": "black_dot",
          "session_id": str, "block_id": str, "frame_index": int,
          "roi": {"x", "y", "width", "height"},      -- the drawn square
          "pixel_uv": [u, v], "pixel_method": str,  -- dot centre, full-image pixels
          "xyz_camera_mm": [x, y, z], "xyz_method": str,
          "coordinate_space": "kinect_color_camera", "units": "mm", ...
        }

    Args:
        rgb_video_path: The block's colour MP4 (frame scrubbing).
        source_video_path: The block's MKV (colour and depth of the chosen frame).
        session_id: Session the label belongs to.
        block_id: Block whose video is used.
        output_path: Session-level label path
            (``ManualLabelRefPointFileHandler.default_path``).
        force_processing: Re-label even if the session already has a label.

    Returns:
        Path to the label, or None if the user cancelled (nothing is saved, so
        the next block of the session asks again).
    """
    if session_id in _SESSIONS_LABELLED_THIS_RUN or (output_path.exists() and not force_processing):
        logger.info(f"Skipping: reference point already labelled for {session_id} ({output_path}).")
        return output_path

    # Re-labelling on the same block starts from the previous frame and square.
    existing = None
    if output_path.exists():
        try:
            existing = ManualLabelRefPointFileHandler.load(output_path)
        except Exception:
            existing = None
    same_block = existing is not None and existing.get("block_id") == block_id
    start_frame = int(existing["frame_index"]) if same_block else 0

    frame_index = _select_frame(rgb_video_path, start_frame)
    if frame_index is None:
        logger.warning(f"Reference-point labelling cancelled for {session_id} ({block_id}) — no file saved.")
        return None

    print(f"[{session_id}] Reading frame {frame_index} from {source_video_path.name} "
          f"(sequential read, can take a minute)...")
    color_bgr, point_cloud = read_kinect_frame(source_video_path, frame_index)

    predefined = existing["roi"] if same_block and int(existing["frame_index"]) == frame_index else None
    roi = _draw_square(
        color_bgr,
        f"Draw a square around the BLACK DOT — {session_id} {block_id} frame {frame_index}  |  "
        f"left-click drag, then Proceed / Enter (ESC cancels)",
        predefined,
    )
    if roi is None:
        logger.warning(f"Reference-point labelling cancelled for {session_id} ({block_id}) — no file saved.")
        return None

    u, v, pixel_method = refine_dot_center(color_bgr, roi)
    xyz, xyz_method = lookup_camera_xyz(point_cloud, u, v, roi)

    label = {
        "label": "black_dot",
        "description": "Manually labelled reference point (black ink dot) on the touch receiver's forearm.",
        "session_id": session_id,
        "block_id": block_id,
        "frame_index": int(frame_index),
        "rgb_video_path": str(rgb_video_path),
        "source_video_path": str(source_video_path),
        "roi": roi,
        "pixel_uv": [float(u), float(v)],
        "pixel_method": pixel_method,
        "xyz_camera_mm": [float(c) for c in xyz],
        "xyz_method": xyz_method,
        "coordinate_space": "kinect_color_camera",
        "units": "mm",
        "parallax_correction": True,
        "labelled_at": datetime.now().isoformat(timespec="seconds"),
    }
    ManualLabelRefPointFileHandler.save(output_path, label)
    _SESSIONS_LABELLED_THIS_RUN.add(session_id)

    logger.info(
        f"Saved reference point for {session_id}: pixel ({u:.1f}, {v:.1f}) [{pixel_method}] -> "
        f"[{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}] mm [{xyz_method}] in {output_path}"
    )
    return output_path
