"""Explainability: attention over time + Expected-Gradients (GradientSHAP) feature attributions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..constants import EXPLOIT_STAGES
from ..features.states import SCALES
from .bundle import Bundle


def _risk_fn(bundle: Bundle, K: int):
    m = bundle.model
    w = bundle.config["engine"]["fusion"]
    wd, wr = w["direct"], w["rollout"]
    ex = torch.tensor(EXPLOIT_STAGES)

    def f(x):  # x (B,L,D) -> differentiable fused risk (direct + deterministic rollout)
        out = m.observe(x, sample=False)
        h, z = out["h"][:, -1], out["post_mu"][:, -1]
        direct = torch.sigmoid(m.heads(h, z)["risk_logit"])
        hi, zi = m.imagine(h, z, K, sample=False)
        roll = torch.softmax(m.heads(hi, zi)["stage_logits"], -1)[..., ex].sum(-1).max(-1).values
        return (wd * direct + wr * roll) / (wd + wr)
    return f


def gradient_shap(bundle: Bundle, ctx: np.ndarray, background: np.ndarray, n_samples: int = 64,
                  K: int | None = None, seed: int = 0) -> dict:
    """Expected Gradients (Erion et al. 2021; SHAP GradientExplainer): Shapley-value
    approximation that integrates gradients along paths from background samples."""
    K = K or int(bundle.config["engine"]["rollout_steps"])
    f = _risk_fn(bundle, K)
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(ctx[None].astype(np.float32))
    bg = torch.from_numpy(background[rng.integers(0, len(background), n_samples)].astype(np.float32))
    alpha = torch.from_numpy(rng.uniform(0, 1, (n_samples, 1, 1)).astype(np.float32))
    pts = (bg + alpha * (x - bg)).requires_grad_(True)
    was_training = bundle.model.training
    bundle.model.eval()
    with torch.backends.cudnn.flags(enabled=False):
        y = f(pts)
        (g,) = torch.autograd.grad(y.sum(), pts)
    bundle.model.train(was_training)
    attr = (g * (x - bg)).mean(0).detach().numpy()            # (L,D)
    with torch.no_grad():
        fx = float(f(x)[0])
        base = float(f(bg).mean())
    # Width comes from the bundle, not the current BASE_FEATURES, so a bundle trained on
    # an earlier feature set still explains itself with its own names.
    nb = len(bundle.feature_names) // len(SCALES)
    per_scale = attr.sum(0).reshape(len(SCALES), nb)           # (3,F)
    df = pd.DataFrame({"feature": bundle.feature_names[:nb], "attribution": per_scale.sum(0),
                       "instant": per_scale[0], "burst_vs_short_ema": per_scale[1], "long_trend": per_scale[2]})
    df = df.reindex(df["attribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return {"features": df, "per_time": np.abs(attr).sum(1), "risk": fx, "base_value": base}


def top_drivers(expl: dict, k: int = 5) -> list[str]:
    d = expl["features"].head(k)
    return [f'{r.feature} ({"+" if r.attribution > 0 else "-"}{abs(r.attribution):.3f})' for r in d.itertuples()]
