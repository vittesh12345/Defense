"""Run the engine over video frames.

Video footage is unlabelled, so instead of mAP this reports a GT-free
**detection-stability** proxy: how many detections survive the attack, summed
over evenly-sampled frames (`attacked / clean`). Lower = the attack removed more
of what the detector saw. Useful for domain footage (e.g. drone clips) where
hand ground truth isn't available; for a rigorous mAP, use labelled images.

Core logic (`run_on_frames`) is decoupled from video I/O (`frames_from_video`)
so it's testable without decoding a real clip.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from proving_ground.adapters.base import Detector


def frames_from_video(path: str, n_frames: int) -> list[tuple[int, np.ndarray]]:
    """Evenly sample ``n_frames`` RGB frames from a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        raise ValueError(f"video reports no frames: {path}")
    idxs = [round(i * (total - 1) / max(1, n_frames - 1)) for i in range(n_frames)] \
        if n_frames > 1 else [total // 2]
    out: list[tuple[int, np.ndarray]] = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if ok:
            out.append((fi, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return out


def run_on_frames(
    detector: Detector,
    attack,
    frames: Sequence[tuple[int, np.ndarray]],
) -> dict:
    """Clean vs attacked detection counts per frame + aggregate stability."""
    per_frame = []
    clean_total = attacked_total = 0
    for idx, frame in frames:
        clean = detector.predict(frame)
        adv = attack.apply(detector, frame, []) if attack is not None else frame
        attacked = detector.predict(adv)
        per_frame.append({"frame": idx, "clean_det": len(clean), "attacked_det": len(attacked)})
        clean_total += len(clean)
        attacked_total += len(attacked)
    return {
        "attack": getattr(attack, "name", None),
        "n_frames": len(frames),
        "clean_detections": clean_total,
        "attacked_detections": attacked_total,
        "detection_retained": (attacked_total / clean_total) if clean_total else None,
        "per_frame": per_frame,
    }


def run_video(detector: Detector, attack, path: str, n_frames: int = 8) -> dict:
    """Sample frames from a clip and run clean-vs-attacked detection over them."""
    result = run_on_frames(detector, attack, frames_from_video(path, n_frames))
    result["video"] = path
    return result
