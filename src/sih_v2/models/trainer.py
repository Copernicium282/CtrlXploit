"""Training loops: NetWorldModel and the plain supervised LSTM baseline.

Both share the same data path (index-gathered mini-batches, benign-segment
sub-sampling, telemetry-jitter augmentation) and the same model-selection rule,
so the only difference between them is the world-model machinery itself
(latent dynamics, KL, reconstruction and K-step imagination).
"""
from __future__ import annotations

import copy
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from ..constants import EXPLOIT_STAGES, N_STAGES
from ..features.sequences import context_index, training_index
from .losses import world_model_loss
from .world_model import NetWorldModel

log = logging.getLogger(__name__)


def build_model(obs_dim: int, cfg: dict) -> NetWorldModel:
    m = cfg["model"]
    return NetWorldModel(obs_dim, N_STAGES, m["embed_dim"], m["lstm_hidden"], m["attn_heads"], m["attn_layers"],
                         m["deter_dim"], m["stoch_dim"], m["dropout"])


class SeqClassifier(nn.Module):
    """Plain supervised LSTM: x_{t-L+1..t} -> P(exploit in (t,t+H]) and current stage.
    Represents the LSTM-classifier family: temporal context, but no latent dynamics and no imagination."""

    def __init__(self, obs_dim: int, n_stages: int, hidden: int = 96, dropout: float = 0.2):
        super().__init__()
        self.hparams = dict(obs_dim=obs_dim, n_stages=n_stages, hidden=hidden, dropout=dropout)
        self.inp = nn.Sequential(nn.Linear(obs_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.risk = nn.Linear(hidden, 1)
        self.stage = nn.Linear(hidden, n_stages)

    def forward(self, x):
        h, _ = self.lstm(self.inp(x))
        return self.risk(h).squeeze(-1), self.stage(h)


def selection_score(y, s, stage, metric: str = "mix") -> float:
    """Validation model-selection score. 'ew' = early-warning PR-AUC (cells not yet
    under exploitation), 'overall' = PR-AUC, 'mix' = their mean (falls back to
    overall when the early-warning subset has a single class, as on CTU-13)."""
    def ap(yy, ss):
        return float(average_precision_score(yy, ss)) if len(yy) and yy.min() != yy.max() else float("nan")
    ew = ~np.isin(stage, EXPLOIT_STAGES)
    a_ew, a_all = ap(y[ew], s[ew]), ap(y, s)
    if metric == "ew":
        return a_ew if a_ew == a_ew else a_all
    if metric == "overall" or a_ew != a_ew:
        return a_all
    return 0.5 * (a_ew + a_all)


@torch.no_grad()
def direct_scores(model, X, seg, L, batch=2048):
    model.eval()
    ci = context_index(seg, L)
    out = []
    for i in range(0, len(ci), batch):
        x = torch.from_numpy(X[ci[i:i + batch]])
        if isinstance(model, SeqClassifier):
            out.append(torch.sigmoid(model(x)[0][:, -1]).numpy())
        else:
            o = model.observe(x, sample=False)
            out.append(torch.sigmoid(model.heads(o["h"][:, -1], o["post_mu"][:, -1])["risk_logit"]).numpy())
    return np.concatenate(out)


def _prep(train, cfg, seed):
    tc = cfg["train"]
    L, K = cfg["forecast"]["context"], cfg["engine"]["rollout_steps"]
    pos = (train["stage"] > 0) | (train["y"] > 0)
    idx, mk = training_index(train["seg"], L, K, 1, positive=pos, benign_stride=tc.get("benign_stride"))
    freq = np.bincount(train["stage"], minlength=N_STAGES).astype(float) + 1.0
    sw = torch.tensor((freq.sum() / freq) ** 0.5, dtype=torch.float32)
    log.info("training sequences: %d (benign-only segments sub-sampled every %s)", len(idx), tc.get("benign_stride"))
    return idx, torch.from_numpy(mk), sw / sw.mean(), np.random.default_rng(seed), L, K


VOLUME_FEATURES = ("log_flows", "log_bytes", "log_pkts")


def _benign_aug_setup(train, idx):
    """Benign flash-crowd augmentation.
    Sequences drawn from benign-only context get a sustained volume surge added to the instant and
    long-trend volume features in scaled units; labels stay benign. Flags, ports, TTL, timing and
    entropies are untouched, so genuine attack signatures keep their meaning."""
    names, scale = train.get("feature_names"), train.get("scale")
    if names is None or scale is None:
        return None
    cols = [i for i, n in enumerate(names) if n.split("__")[0] in VOLUME_FEATURES and "__dev_short" not in n]
    benign_seq = ((train["stage"][idx] == 0) & (train["y"][idx] == 0)).all(1)
    return torch.tensor(cols), torch.tensor(1.0 / np.asarray(scale)[cols], dtype=torch.float32), benign_seq


def _fit(model, step_fn, train, val, cfg, seed, idx, rng, L, name):
    tc = cfg["train"]
    X = torch.from_numpy(train["X"])
    aug_p = tc.get("benign_volume_aug", 0.0)
    aug = _benign_aug_setup(train, idx) if aug_p > 0 else None
    st = torch.from_numpy(train["stage"].astype(np.int64))
    yy = torch.from_numpy(train["y"].astype(np.int64))
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    best, best_state, wait, hist = -1.0, None, 0, []
    n = len(idx)
    cap = tc.get("max_seq_per_epoch") or n
    for ep in range(tc["epochs"]):
        model.train()
        t0 = time.time()
        perm = rng.permutation(n)[:cap]
        agg: dict[str, float] = {}
        nb = 0
        for i in range(0, len(perm), tc["batch_size"]):
            b = torch.from_numpy(idx[perm[i:i + tc["batch_size"]]])
            ob = X[b]
            if tc.get("input_noise", 0) > 0:  # telemetry jitter augmentation (regulariser)
                ob = ob + tc["input_noise"] * torch.randn_like(ob)
            if aug is not None:
                rows = perm[i:i + tc["batch_size"]]
                pick = torch.from_numpy(aug[2][rows] & (rng.random(len(rows)) < aug_p))
                if pick.any():
                    r = torch.from_numpy(rng.uniform(np.log(2), np.log(20), int(pick.sum())).astype(np.float32))
                    delta = torch.zeros(ob.shape[0], ob.shape[2])
                    delta[pick.nonzero().squeeze(1)[:, None], aug[0][None, :]] = r[:, None] * aug[1][None, :]
                    ob = (ob + delta[:, None, :]).clamp(-10, 10)
            loss, logs = step_fn(ob, st[b], yy[b], perm[i:i + tc["batch_size"]])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        sched.step()
        s = direct_scores(model, val["X"], val["seg"], L)
        score = selection_score(val["y"], s, val["stage"], tc.get("select_metric", "mix"))
        rec = {"epoch": ep + 1, **{k: v / nb for k, v in agg.items()}, "val_select": score, "sec": time.time() - t0}
        hist.append(rec)
        log.info("[%s seed %d] epoch %02d loss %.3f | val selection %.4f (%.1fs)", name, seed, ep + 1,
                 rec.get("loss", 0), score, rec["sec"])
        if score > best + 1e-4:
            best, best_state, wait = score, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= tc["patience"]:
                log.info("early stopping")
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


def train_world_model(train: dict, val: dict, cfg: dict, seed: int | None = None, progress=None):
    """train/val: dict(X, stage, y, seg) numpy arrays (X already scaled)."""
    seed = cfg["seed"] if seed is None else seed
    torch.manual_seed(seed)
    idx, mk, sw, rng, L, K = _prep(train, cfg, seed)
    model = build_model(train["X"].shape[1], cfg)

    def step(ob, st, yy, rows):
        return world_model_loss(model, ob, st, yy, mk[rows], L, K, cfg, sw)
    return _fit(model, step, train, val, cfg, seed, idx, rng, L, "world-model")


def train_seq_classifier(train: dict, val: dict, cfg: dict, seed: int | None = None):
    seed = cfg["seed"] if seed is None else seed
    torch.manual_seed(seed)
    idx, mk, sw, rng, L, K = _prep(train, cfg, seed)
    model = SeqClassifier(train["X"].shape[1], N_STAGES)
    ex = torch.tensor(EXPLOIT_STAGES)
    w_pre = cfg["train"].get("w_prewarn", 1.0)

    def step(ob, st, yy, rows):
        m = mk[rows][:, :L]
        risk, stage = model(ob[:, :L])
        yl = yy[:, :L].float()
        fw = 1.0 + (w_pre - 1.0) * yl * (~torch.isin(st[:, :L], ex)).float()
        bce = (F.binary_cross_entropy_with_logits(risk, yl, reduction="none") * fw * m).sum() / m.sum()
        ce = (F.cross_entropy(stage.transpose(1, 2), st[:, :L], weight=sw, reduction="none") * m).sum() / m.sum()
        loss = 2.0 * bce + ce
        return loss, {"loss": loss.item()}
    return _fit(model, step, train, val, cfg, seed, idx, rng, L, "lstm-baseline")
