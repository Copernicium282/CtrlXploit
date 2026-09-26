"""Robust (median / IQR) scaler, serialisable as plain lists."""
from __future__ import annotations

import numpy as np


class RobustScaler:
    def __init__(self, center=None, scale=None, clip: float = 10.0):
        self.center = None if center is None else np.asarray(center, dtype=np.float32)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float32)
        self.clip = clip

    def fit(self, x: np.ndarray) -> "RobustScaler":
        q1, med, q3 = np.nanpercentile(x, [25, 50, 75], axis=0)
        iqr = q3 - q1
        std = np.nanstd(x, axis=0)
        self.center = med.astype(np.float32)
        self.scale = np.where(iqr > 1e-6, iqr, np.where(std > 1e-6, std, 1.0)).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float32) - self.center) / self.scale
        return np.clip(np.nan_to_num(z), -self.clip, self.clip).astype(np.float32)

    def to_dict(self) -> dict:
        return {"center": self.center.tolist(), "scale": self.scale.tolist(), "clip": self.clip}

    @classmethod
    def from_dict(cls, d: dict) -> "RobustScaler":
        return cls(d["center"], d["scale"], d.get("clip", 10.0))
