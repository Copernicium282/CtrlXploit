"""Windowed temporal matrices for training (context + future) and inference (context).

Sequences are represented as *index* matrices into the cell table and gathered
per mini-batch, so hundreds of thousands of overlapping sequences never have to
be materialised (93 features x 34 steps x 300k sequences would be ~4 GB)."""
from __future__ import annotations

import numpy as np


def segments(seg: np.ndarray):
    change = np.flatnonzero(np.diff(seg) != 0) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(seg)]
    return zip(starts, ends)


def training_index(seg, L: int, K: int, stride: int = 1, positive=None, benign_stride: int | None = None):
    """Return idx (M,L+K) and mask (M,L+K).

    The first L steps are context (posterior filtering); steps L..L+K-1 are the
    ground-truth future used to supervise K-step imagined rollouts. Positions
    outside the segment are clamped (idx) and masked (loss). Segments that never
    contain a positive cell are sampled with `benign_stride` (class balancing)."""
    idx_list, mask_list = [], []
    offs = np.arange(-L + 1, K + 1)
    for s, e in segments(seg):
        n = e - s
        st = stride
        if benign_stride and positive is not None and not positive[s:e].any():
            st = benign_stride
        for end in range(0, n, st):
            p = end + offs
            mask_list.append((p >= 0) & (p < n))
            idx_list.append(s + np.clip(p, 0, n - 1))
    return np.stack(idx_list), np.stack(mask_list).astype(np.float32)


def training_sequences(X, stage, y, seg, L: int, K: int, stride: int = 1):
    """Materialised variant (small data / tests)."""
    idx, mask = training_index(seg, L, K, stride)
    return X[idx], stage[idx], y[idx], mask


def context_index(seg, L: int) -> np.ndarray:
    idx = np.empty((len(seg), L), dtype=np.int64)
    offs = np.arange(-L + 1, 1)
    for s, e in segments(seg):
        t = np.arange(s, e)[:, None]
        idx[s:e] = np.maximum(t + offs, s)
    return idx


def context_windows(X, seg, L: int):
    """For every row t: the L most recent states ending at t (clamped to segment start)."""
    return X[context_index(seg, L)]


def stacked_features(X, seg, S: int) -> np.ndarray:
    """History-stacked design matrix [x_{t-S+1} .. x_t] for the sklearn baselines."""
    return X[context_index(seg, S)].reshape(len(X), -1)
