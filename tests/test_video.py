"""Fast-tier tests for video frame-sampling + per-frame attack run (no weights)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from proving_ground.adapters.fake import FakeDetector
from proving_ground.attacks.degradation import DegradationAttack
from proving_ground.video import frames_from_video, run_on_frames


def _frame(i: int) -> np.ndarray:
    return np.random.default_rng(i).integers(0, 256, (48, 64, 3), dtype=np.uint8)


def test_run_on_frames_structure():
    frames = [(i, _frame(i)) for i in range(4)]
    res = run_on_frames(FakeDetector(), DegradationAttack("fog", 0.8), frames)
    assert res["n_frames"] == 4
    assert res["clean_detections"] == 4  # FakeDetector returns one box per frame
    assert len(res["per_frame"]) == 4
    assert res["attack"] == "degradation-fog"
    assert res["detection_retained"] is not None


def test_run_on_frames_no_attack_is_identity():
    res = run_on_frames(FakeDetector(), None, [(0, _frame(0))])
    assert res["attack"] is None
    assert res["detection_retained"] == 1.0


def test_frames_from_video_roundtrip(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("no MJPG VideoWriter available")
    for i in range(10):
        bgr = np.random.default_rng(i).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        writer.write(bgr)
    writer.release()

    frames = frames_from_video(path, 4)
    assert len(frames) == 4
    for _idx, f in frames:
        assert f.shape == (48, 64, 3) and f.dtype == np.uint8
