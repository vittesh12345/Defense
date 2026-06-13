"""Degraded-Visual-Environment (DVE) attacks: black-box image degradations.

These simulate real sensor/weather conditions (blur, noise, fog, low light, JPEG
artifacts) rather than gradient-crafted perturbations. They are fully black-box:
``apply`` ignores the detector and targets and just transforms the pixels, so any
``Detector`` works — no ``WhiteBox`` / gradients needed.

``severity`` is a float in ``[0, 1]``: 0 is (near-)identity, 1 is strong. Each
mode maps it to its own parameter. Stochastic modes (noise, low light) draw from
a seeded NumPy generator, so output is deterministic for a fixed seed.
"""

from __future__ import annotations

import cv2
import numpy as np

from proving_ground.adapters.base import Detection, Detector

MODES = (
    "gaussian_blur",
    "motion_blur",
    "gaussian_noise",
    "fog",
    "low_light",
    "jpeg_compression",
    "smoke",
    "dust",
)


def _u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


class DegradationAttack:
    def __init__(self, mode: str, severity: float = 0.5, seed: int = 0) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1]; got {severity}")
        self.mode = mode
        self.severity = severity
        self.seed = seed
        self.name = f"degradation-{mode}"

    def apply(self, detector: Detector, image: np.ndarray, targets: list[Detection]) -> np.ndarray:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected RGB uint8 HWC image, got {image.shape}/{image.dtype}")
        return getattr(self, f"_{self.mode}")(image)

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    # --- modes ------------------------------------------------------------

    def _gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        k = 1 + 2 * round(self.severity * 7)  # odd kernel, 1 (identity) .. 15
        if k <= 1:
            return image.copy()
        return cv2.GaussianBlur(image, (k, k), 0)

    def _motion_blur(self, image: np.ndarray) -> np.ndarray:
        length = max(1, round(self.severity * 20))  # 1 (identity) .. 20
        if length <= 1:
            return image.copy()
        kernel = np.eye(length, dtype=np.float32) / length  # 45-degree streak
        return cv2.filter2D(image, -1, kernel)

    def _gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        std = self.severity * 64.0
        if std <= 0:
            return image.copy()
        noise = self._rng().normal(0.0, std, size=image.shape)
        return _u8(image.astype(np.float32) + noise)

    def _fog(self, image: np.ndarray) -> np.ndarray:
        t = self.severity * 0.75  # veil opacity
        if t <= 0:
            return image.copy()
        veil = 235.0  # bright, low-contrast haze
        return _u8(image.astype(np.float32) * (1.0 - t) + veil * t)

    def _low_light(self, image: np.ndarray) -> np.ndarray:
        gamma = 1.0 + self.severity * 2.0  # >1 darkens
        std = self.severity * 8.0  # sensor noise in the dark
        if self.severity <= 0:
            return image.copy()
        dark = 255.0 * (image.astype(np.float32) / 255.0) ** gamma
        if std > 0:
            dark = dark + self._rng().normal(0.0, std, size=image.shape)
        return _u8(dark)

    def _jpeg_compression(self, image: np.ndarray) -> np.ndarray:
        quality = max(1, round(100 - self.severity * 95))  # 100 (near-identity) .. 5
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def _smoke(self, image: np.ndarray) -> np.ndarray:
        # Non-uniform dark-grey obscurant (battlefield smoke: grenades, fires,
        # vehicle exhaust). A low-resolution noise field, upsampled to image size,
        # supplies the billowing per-pixel opacity; the veil color is fixed.
        t = self.severity * 0.8
        if t <= 0:
            return image.copy()
        h, w = image.shape[:2]
        ch, cw = max(1, h // 8), max(1, w // 8)
        coarse = self._rng().standard_normal((ch, cw)).astype(np.float32)
        field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
        fmin, fmax = float(field.min()), float(field.max())
        field = (field - fmin) / (fmax - fmin) if fmax > fmin else np.zeros_like(field)
        alpha = (t * field)[..., None]
        veil = 80.0
        return _u8(image.astype(np.float32) * (1.0 - alpha) + veil * alpha)

    def _dust(self, image: np.ndarray) -> np.ndarray:
        # Warm-tan suspended-particle veil with fine granular noise (brownout,
        # sandstorms, vehicle disturbance). Uniform opacity plus per-pixel grain
        # distinguishes it from the bright-white, low-contrast `_fog` haze.
        t = self.severity * 0.7
        if t <= 0:
            return image.copy()
        veil = np.array([200.0, 175.0, 130.0], dtype=np.float32)
        grain = self._rng().normal(0.0, self.severity * 6.0, size=image.shape)
        return _u8(image.astype(np.float32) * (1.0 - t) + veil * t + grain)
