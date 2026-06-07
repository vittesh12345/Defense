"""Detection-overlay rendering for demo figures.

Pure drawing helpers (no model dependency) so they're testable without weights.
Images are RGB uint8 HWC throughout.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from proving_ground.adapters.base import Detection


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Detection],
    color: tuple[int, int, int] = (0, 200, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Return a copy of the RGB image with boxes + class/score labels drawn."""
    out = image.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in d.box_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{d.class_name} {d.score:.2f}"
        cv2.putText(out, label, (x1, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def _titled(panel: np.ndarray, title: str, bar_h: int = 26) -> np.ndarray:
    """Add a black title bar above a panel."""
    h, w = panel.shape[:2]
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    cv2.putText(bar, title, (6, bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def side_by_side(
    left: np.ndarray, right: np.ndarray, left_title: str, right_title: str
) -> np.ndarray:
    """Two titled panels stacked horizontally with a thin separator."""
    left_t = _titled(left, left_title)
    right_t = _titled(right, right_title)
    sep = np.full((left_t.shape[0], 4, 3), 255, dtype=np.uint8)
    return np.hstack([left_t, sep, right_t])
