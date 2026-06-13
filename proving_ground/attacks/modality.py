"""Cross-modality domain-shift probes: simulated thermal-IR and SAR imagery.

HONEST SCOPE — read this first. We have no thermal/radar detector or imagery, so
these do **not** evaluate a real IR/SAR sensor. They are black-box *domain-shift*
transforms that render an RGB frame to *resemble* a thermal or SAR product, then
let us measure how far an optical (RGB-trained) detector's performance falls. So
this is a generalization / out-of-distribution robustness probe — "would this
model survive imagery from another modality?" — not a multi-modal capability
claim. They are kept out of the DVE degradation family and the headline suite for
exactly that reason: the threat model is different.

``thermal_ir`` — collapse to a white-hot luminance map (chroma gone) and smooth
fine detail (thermal optics are lower-resolution). For an RGB detector the
dominant OOD cue is the loss of colour and texture.

``sar`` — single-channel backscatter intensity corrupted by multiplicative
**speckle**, the defining SAR artifact, modeled as a Gamma(L, 1/L) gain (mean 1)
whose number of looks ``L`` falls as severity rises (fewer looks = heavier
speckle).

``severity`` in ``[0, 1]``: 0 is (near-)identity (blended back to the RGB frame),
1 is the full modality look. The ``sar`` speckle draws from a seeded generator,
so output is deterministic for a fixed seed. Fully black-box: ``apply`` ignores
the detector and targets.
"""

from __future__ import annotations

import cv2
import numpy as np

from proving_ground.adapters.base import Detection, Detector

MODES = ("thermal_ir", "sar")

# Rec. 601 luma weights for an RGB (not BGR) frame — our loader yields RGB.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


class ModalityShift:
    def __init__(self, mode: str, severity: float = 0.8, seed: int = 0) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown modality {mode!r}; choose from {MODES}")
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1]; got {severity}")
        self.mode = mode
        self.severity = severity
        self.seed = seed
        self.name = f"modality-{mode}"

    def apply(self, detector: Detector, image: np.ndarray, targets: list[Detection]) -> np.ndarray:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected RGB uint8 HWC image, got {image.shape}/{image.dtype}")
        return getattr(self, f"_{self.mode}")(image)

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def _luma(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) @ _LUMA  # H x W

    # --- modalities -------------------------------------------------------

    def _thermal_ir(self, image: np.ndarray) -> np.ndarray:
        s = self.severity
        if s <= 0:
            return image.copy()
        gray = self._luma(image)
        k = 1 + 2 * round(s * 3)  # lower-resolution thermal optics: blur up to 7px
        if k > 1:
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        # Smooth thermal tonal curve (white-hot: bright = hot).
        toned = 255.0 * (gray / 255.0) ** (1.0 + 0.5 * s)
        thermal = np.repeat(toned[..., None], 3, axis=2)
        return _u8((1.0 - s) * image.astype(np.float32) + s * thermal)

    def _sar(self, image: np.ndarray) -> np.ndarray:
        s = self.severity
        if s <= 0:
            return image.copy()
        gray = self._luma(image)
        looks = (1.0 - s) * 19.0 + 1.0  # s=0 -> 20 looks (mild), s=1 -> 1 look (heavy)
        speckle = self._rng().gamma(shape=looks, scale=1.0 / looks, size=gray.shape)
        sar = np.repeat((gray * speckle.astype(np.float32))[..., None], 3, axis=2)
        return _u8((1.0 - s) * image.astype(np.float32) + s * sar)
