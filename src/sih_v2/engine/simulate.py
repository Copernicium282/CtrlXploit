"""K-step forward simulation with particle fusion.

For every window t:
  1. filter the posterior over the last L observed windows
  2. draw P particles z_t ~ q(z_t | history) and roll the learned prior K steps
  3. read the stage head at every imagined step -> P x K x C stage trajectories
  4. fuse three estimators of P(exploitation within K):
        direct   - forecast head on the filtered belief state
        rollout  - particle mean of max_k P(exploit stage at imagined step k)
        markov   - kill-chain transition prior applied to the current stage belief
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..constants import EXPLOIT_STAGES, STAGES
from ..features.sequences import context_index
from .bundle import Bundle


@dataclass
class ForecastResult:
    frame: pd.DataFrame          # one row per window
    rollout_mean: np.ndarray     # (N,K,C) particle-mean imagined stage distribution
    exploit_q: np.ndarray        # (N,K,3) 10/50/90% particle quantiles of P(exploit) per step
    markov_path: np.ndarray      # (N,K,C)
    attention: np.ndarray        # (N,L) attention of the latest window over its context
    stage_now: np.ndarray        # (N,C)
    X: np.ndarray                # (N,D) scaled state vectors (for explanations)


class ForecastEngine:
    def __init__(self, bundle: Bundle, particles: int | None = None, steps: int | None = None, seed: int = 0):
        self.b = bundle
        eng = bundle.config["engine"]
        self.P = int(particles or eng["particles"])
        self.K = int(steps or eng["rollout_steps"])
        self.L = int(bundle.config["forecast"]["context"])
        self.w = eng["fusion"]
        self.seed = seed
        self.exploit = torch.tensor(EXPLOIT_STAGES)

    def scale(self, df: pd.DataFrame) -> np.ndarray:
        return self.b.scaler.transform(df[self.b.feature_names].to_numpy(np.float32))

    @torch.no_grad()
    def _batch(self, ctx: np.ndarray):
        m = self.b.model
        x = torch.from_numpy(ctx)
        out = m.observe(x, sample=False)
        h, mu, std = out["h"][:, -1], out["post_mu"][:, -1], out["post_std"][:, -1]
        hd = m.heads(h, mu)
        # label-free surprise:
        #  obs  - error of the prior's one-step prediction of x_t made *before* x_t was seen
        #  kl   - KL(q(z_t|h_t,x_t) || p(z_t|h_t)): how much the observation moved the belief
        pred = m.decoder(torch.cat([h, out["prior_mu"][:, -1]], -1))
        s_obs = ((pred - x[:, -1]) ** 2).mean(-1)
        pm, ps = out["prior_mu"][:, -1], out["prior_std"][:, -1]
        s_kl = (torch.log(ps / std) + (std ** 2 + (mu - pm) ** 2) / (2 * ps ** 2) - 0.5).sum(-1)
        stage_now = torch.softmax(hd["stage_logits"], -1)
        direct = torch.sigmoid(hd["risk_logit"])
        B = x.shape[0]
        hp = h.repeat(self.P, 1)
        zp = (mu.repeat(self.P, 1) + std.repeat(self.P, 1) * torch.randn(self.P * B, mu.shape[1]))
        hi, zi = m.imagine(hp, zp, self.K, sample=True)
        probs = torch.softmax(m.heads(hi, zi)["stage_logits"], -1).reshape(self.P, B, self.K, -1)
        ex = probs[..., self.exploit].sum(-1)                        # (P,B,K)
        rollout = ex.max(-1).values.mean(0)                           # (B,)
        q = torch.quantile(ex, torch.tensor([0.1, 0.5, 0.9]), dim=0).permute(1, 2, 0)  # (B,K,3)
        hit = ex > 0.5
        first = torch.where(hit.any(-1), hit.float().argmax(-1) + 1.0, torch.full_like(ex[..., 0], float("nan")))
        eta = torch.nan_to_num(torch.nanmedian(first, dim=0).values, nan=-1.0)  # median ETA over hitting particles
        attn = out["attn"][:, -1] if out["attn"] is not None else torch.zeros(B, ctx.shape[1])
        return dict(stage_now=stage_now.numpy(), direct=direct.numpy(), rollout=rollout.numpy(),
                    rollout_mean=probs.mean(0).numpy(), q=q.numpy(), eta=eta.numpy(),
                    hit_frac=hit.any(-1).float().mean(0).numpy(), attn=attn.numpy(),
                    s_obs=s_obs.numpy(), s_kl=s_kl.numpy())

    def run(self, df: pd.DataFrame, batch_size: int = 512) -> ForecastResult:
        df = df.reset_index(drop=True)
        torch.manual_seed(self.seed)
        X = self.scale(df)
        ci = context_index(df["segment"].to_numpy(), self.L)
        parts = [self._batch(X[ci[i:i + batch_size]]) for i in range(0, len(ci), batch_size)]
        cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        markov_risk = self.b.markov.risk_within(cat["stage_now"], self.K)
        markov_path = self.b.markov.propagate(cat["stage_now"], self.K)
        w = self.w
        fused = (w["direct"] * cat["direct"] + w["rollout"] * cat["rollout"] + w["markov"] * markov_risk) \
            / (w["direct"] + w["rollout"] + w["markov"])

        out = df[[c for c in ("ts", "wid", "capture", "host", "segment", "stage", "y_future", "phase", "time_to_attack",
                              "exploit_now", "n_flows_raw", "split") if c in df]].copy()
        out["time"] = pd.to_datetime(out["ts"], unit="s")
        out["risk_direct"] = cat["direct"]
        out["risk_rollout"] = cat["rollout"]
        out["risk_markov"] = markov_risk
        out["risk"] = fused
        out["alert"] = (fused >= self.b.threshold).astype(np.int8)
        out["current_stage_pred"] = [STAGES[i] for i in cat["stage_now"].argmax(1)]
        fut = cat["rollout_mean"][..., EXPLOIT_STAGES].sum(1)
        out["forecast_stage"] = [STAGES[EXPLOIT_STAGES[i]] for i in fut.argmax(1)]
        out["eta_windows"] = cat["eta"]
        out["particle_hit_frac"] = cat["hit_frac"]
        out["surprise_obs"] = cat["s_obs"]
        out["surprise_kl"] = cat["s_kl"]
        sign = float(self.b.meta.get("surprise_sign", 1.0))
        grp = out["capture"] if "capture" in out else pd.Series(0, index=out.index)
        out["surprise_z"] = sign * robust_z(out["surprise_obs"], grp)
        out["anomaly"] = (out["surprise_z"] >= float(self.b.config["engine"].get("surprise_z", 3.0))).astype(np.int8)
        return ForecastResult(out, cat["rollout_mean"], cat["q"], markov_path, cat["attn"], cat["stage_now"], X)



def robust_z(v: pd.Series, groups: pd.Series) -> np.ndarray:
    """(v - median) / (1.4826 * MAD) within each capture. Median/MAD rather than
    mean/std because the handful of extreme hosts we hunt for would otherwise
    inflate the spread that hides them."""
    v = pd.Series(np.asarray(v, float), index=groups.index)
    med = v.groupby(groups).transform("median")
    mad = (v - med).abs().groupby(groups).transform("median")
    return ((v - med) / (1.4826 * mad + 1e-6)).to_numpy()
