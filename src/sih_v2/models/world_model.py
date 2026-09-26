"""NetWorldModel: a recurrent state-space world model of network telemetry.

    context encoder   e_t = MLP(x_t);  c_{1:t} = CausalTransformer(LSTM(e_{1:t}))
    deterministic     h_t = GRU(h_{t-1}, z_{t-1})
    prior (dynamics)  p(z_t | h_t)                -> P(S_{t+1} | S_t) used for imagination
    posterior         q(z_t | h_t, c_t)           -> filtering on observed telemetry
    decoder           p(x_t | h_t, z_t)           -> reconstructs the telemetry state vector
    stage head        p(stage_t | h_t, z_t)       -> kill-chain stage (MITRE tactic)
    forecast head     p(exploit in (t,t+H] | s_t) -> direct risk estimate

Forecasting = filter the posterior over the observed history, then roll the
learned prior K steps forward (many particles) and read the stage head at every
imagined step.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, h, o, dropout=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, o))


class CausalAttentionBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = mlp(d, 2 * d, d, dropout)

    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        a, w = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask,
                         need_weights=True, average_attn_weights=True)
        x = x + a
        return x + self.ff(self.ln2(x)), w


class NetWorldModel(nn.Module):
    def __init__(self, obs_dim: int, n_stages: int, embed_dim=64, lstm_hidden=64, attn_heads=4,
                 attn_layers=1, deter_dim=128, stoch_dim=16, dropout=0.1, max_len=512):
        super().__init__()
        self.hparams = dict(obs_dim=obs_dim, n_stages=n_stages, embed_dim=embed_dim, lstm_hidden=lstm_hidden,
                            attn_heads=attn_heads, attn_layers=attn_layers, deter_dim=deter_dim,
                            stoch_dim=stoch_dim, dropout=dropout, max_len=max_len)
        self.deter_dim, self.stoch_dim = deter_dim, stoch_dim
        self.encoder = mlp(obs_dim, embed_dim, embed_dim, dropout)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True)
        self.proj = nn.Linear(lstm_hidden, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.blocks = nn.ModuleList(CausalAttentionBlock(embed_dim, attn_heads, dropout) for _ in range(attn_layers))
        self.cell = nn.GRUCell(stoch_dim, deter_dim)
        self.prior_net = mlp(deter_dim, deter_dim, 2 * stoch_dim)
        self.post_net = mlp(deter_dim + embed_dim, deter_dim, 2 * stoch_dim)
        feat = deter_dim + stoch_dim
        self.decoder = mlp(feat, 2 * embed_dim, obs_dim)
        self.stage_head = mlp(feat, embed_dim, n_stages, dropout)
        self.forecast_head = mlp(feat, embed_dim, 1, dropout)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _dist(stats):
        mu, raw = stats.chunk(2, -1)
        return mu, F.softplus(raw) + 0.1

    def heads(self, h, z):
        s = torch.cat([h, z], -1)
        return {"stage_logits": self.stage_head(s), "risk_logit": self.forecast_head(s).squeeze(-1),
                "recon": self.decoder(s)}

    def context(self, x):
        e = self.encoder(x)
        c, _ = self.lstm(e)
        c = self.proj(c) + self.pos[:, : x.shape[1]]
        attn = None
        for blk in self.blocks:
            c, attn = blk(c)
        return c, attn

    # ------------------------------------------------------------------ filtering
    def observe(self, x, sample: bool = True):
        """x: (B,T,D) -> per-step h, z (posterior), prior/post statistics, attention."""
        B, T, _ = x.shape
        c, attn = self.context(x)
        h = x.new_zeros(B, self.deter_dim)
        z = x.new_zeros(B, self.stoch_dim)
        hs, zs, pm, ps, qm, qs = [], [], [], [], [], []
        for t in range(T):
            h = self.cell(z, h)
            p_mu, p_std = self._dist(self.prior_net(h))
            q_mu, q_std = self._dist(self.post_net(torch.cat([h, c[:, t]], -1)))
            z = q_mu + q_std * torch.randn_like(q_std) if sample else q_mu
            hs.append(h); zs.append(z); pm.append(p_mu); ps.append(p_std); qm.append(q_mu); qs.append(q_std)
        st = lambda v: torch.stack(v, 1)
        return {"h": st(hs), "z": st(zs), "prior_mu": st(pm), "prior_std": st(ps),
                "post_mu": st(qm), "post_std": st(qs), "attn": attn}

    # ------------------------------------------------------------------ imagination
    def imagine(self, h, z, steps: int, sample: bool = True):
        """Roll the learned prior dynamics forward: (N,deter),(N,stoch) -> (N,K,.)"""
        hs, zs = [], []
        for _ in range(steps):
            h = self.cell(z, h)
            mu, std = self._dist(self.prior_net(h))
            z = mu + std * torch.randn_like(std) if sample else mu
            hs.append(h); zs.append(z)
        return torch.stack(hs, 1), torch.stack(zs, 1)
