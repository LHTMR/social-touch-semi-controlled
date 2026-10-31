"""Persistence for the manually labelled forearm reference point (black ink dot).

One JSON per session, written by the manual step ``define_manual_label_ref_point``
and read by the postprocessing step ``export_manual_label_ref_point``::

    <session_processed_output_dir>/manual_label_ref_point/<session_id>_manual_label_ref_point.json
"""

import json
from pathlib import Path
from typing import Any, Dict, Union

LABEL_NAME = "manual_label_ref_point"


class ManualLabelRefPointFileHandler:
    """Saves and loads the per-session manual reference-point label."""

    @staticmethod
    def default_path(session_processed_output_dir: Union[str, Path], session_id: str) -> Path:
        """Canonical location of a session's label inside its ``2_processed`` folder."""
        return Path(session_processed_output_dir) / LABEL_NAME / f"{session_id}_{LABEL_NAME}.json"

    @staticmethod
    def save(file_path: Union[str, Path], label: Dict[str, Any]) -> None:
        """Write *label* as indented JSON, creating the parent folder if needed."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(label, f, indent=4)

    @staticmethod
    def load(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Read a label; raises ``FileNotFoundError`` when the session was never labelled."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Manual reference-point label not found: {file_path}. "
                f"Run the manual task 'define_manual_label_ref_point' for this session first."
            )
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
