"""World-model objective: reconstruction + KL (free nats) + stage + forecast + K-step overshooting."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from ..constants import EXPLOIT_STAGES

EXPLOIT = torch.tensor(EXPLOIT_STAGES)


def masked_mean(v, m):
    return (v * m).sum() / m.sum().clamp(min=1.0)


def world_model_loss(model, obs, stages, y, mask, L: int, K: int, cfg: dict, stage_weight=None):
    """obs (B,L+K,D) stages/y/mask (B,L+K). Returns (loss, logs)."""
    tc = cfg["train"]
    ctx = obs[:, :L]
    cm = mask[:, :L]
    out = model.observe(ctx, sample=True)
    h, z = out["h"], out["z"]
    hd = model.heads(h, z)

    recon = masked_mean(((hd["recon"] - ctx) ** 2).mean(-1), cm)
    kl = kl_divergence(Normal(out["post_mu"], out["post_std"]), Normal(out["prior_mu"], out["prior_std"])).sum(-1)
    kl_m = masked_mean(kl, cm)
    kl_loss = torch.clamp(kl_m, min=tc["free_nats"])  # free nats: do not over-compress the latent
    stage_ce = masked_mean(F.cross_entropy(hd["stage_logits"].transpose(1, 2), stages[:, :L],
                                           weight=stage_weight, reduction="none"), cm)
    # early-warning emphasis: positives that are NOT yet exploitation are the valuable ones
    yl = y[:, :L].float()
    pre = yl * (~torch.isin(stages[:, :L], EXPLOIT)).float()
    fw = 1.0 + (tc.get("w_prewarn", 1.0) - 1.0) * pre
    fc = masked_mean(F.binary_cross_entropy_with_logits(hd["risk_logit"], yl, reduction="none") * fw, cm)

    # K-step latent overshooting from every context position
    B, _, D = obs.shape
    hi, zi = model.imagine(h.reshape(B * L, -1), z.reshape(B * L, -1), K, sample=True)
    ih = model.heads(hi, zi)
    fut = torch.arange(L, device=obs.device)[:, None] + torch.arange(1, K + 1, device=obs.device)[None]  # (L,K)
    fut_stage = stages[:, fut].reshape(B * L, K)
    fut_mask = (mask[:, fut] * cm[:, :, None]).reshape(B * L, K)
    fut_obs = obs[:, fut].reshape(B * L, K, D)
    r_stage = masked_mean(F.cross_entropy(ih["stage_logits"].transpose(1, 2), fut_stage,
                                          weight=stage_weight, reduction="none"), fut_mask)
    r_recon = masked_mean(((ih["recon"] - fut_obs) ** 2).mean(-1), fut_mask)

    loss = (tc["w_recon"] * recon + tc["kl_beta"] * kl_loss + tc["w_stage"] * stage_ce
            + tc["w_forecast"] * fc + tc["w_rollout_stage"] * r_stage + tc["w_rollout_recon"] * r_recon)
    logs = {"loss": loss.item(), "recon": recon.item(), "kl": kl_m.item(), "stage": stage_ce.item(),
            "forecast": fc.item(), "roll_stage": r_stage.item(), "roll_recon": r_recon.item()}
    return loss, logs
