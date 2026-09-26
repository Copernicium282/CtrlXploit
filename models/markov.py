"""Kill-chain Markov transition prior over MITRE stages (Dirichlet-smoothed)."""
from __future__ import annotations

import numpy as np

from ..constants import EXPLOIT_STAGES, N_STAGES


class KillChainMarkov:
    def __init__(self, alpha: float = 0.5, P=None):
        self.alpha = alpha
        self.P = None if P is None else np.asarray(P, dtype=np.float64)

    def fit(self, stages: np.ndarray, segments: np.ndarray) -> "KillChainMarkov":
        C = np.full((N_STAGES, N_STAGES), self.alpha)
        same = segments[1:] == segments[:-1]
        np.add.at(C, (stages[:-1][same], stages[1:][same]), 1.0)
        self.P = C / C.sum(1, keepdims=True)
        return self

    def propagate(self, belief: np.ndarray, steps: int) -> np.ndarray:
        """belief (N,C) -> per-step stage distributions (N,steps,C)."""
        out, b = [], belief
        for _ in range(steps):
            b = b @ self.P
            out.append(b)
        return np.stack(out, 1)

    def risk_within(self, belief: np.ndarray, steps: int) -> np.ndarray:
        """P(at least one exploitation-stage window in the next `steps` transitions).

        First-passage computation: propagate the belief, removing (absorbing)
        probability mass the moment it enters an exploitation stage."""
        v = belief.astype(np.float64).copy()
        for _ in range(steps):
            v = v @ self.P
            v[:, EXPLOIT_STAGES] = 0.0
        return np.clip(1.0 - v.sum(1), 0.0, 1.0)

    def to_dict(self):
        return {"alpha": self.alpha, "P": self.P.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["alpha"], d["P"])
