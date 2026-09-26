import numpy as np
import torch

from sih_v2.constants import EXPLOIT_STAGES, N_STAGES
from sih_v2.models.losses import world_model_loss
from sih_v2.models.markov import KillChainMarkov
from sih_v2.models.trainer import build_model


def test_world_model_shapes_and_backward(cfg):
    torch.manual_seed(0)
    m = build_model(20, cfg)
    x = torch.randn(3, 12, 20)
    m.eval()  # deterministic attention (no dropout)
    out = m.observe(x)
    assert out["h"].shape == (3, 12, cfg["model"]["deter_dim"]) and out["z"].shape == (3, 12, cfg["model"]["stoch_dim"])
    assert out["attn"].shape == (3, 12, 12)
    assert torch.allclose(out["attn"].sum(-1), torch.ones(3, 12), atol=1e-5)
    assert out["attn"][0, 0, 1:].abs().sum() == 0         # causal: no attention to the future
    hi, zi = m.imagine(out["h"][:, -1], out["z"][:, -1], 5)
    assert hi.shape == (3, 5, cfg["model"]["deter_dim"])
    L, K = 8, 4
    obs = torch.randn(4, L + K, 20)
    st = torch.randint(0, N_STAGES, (4, L + K))
    y = torch.randint(0, 2, (4, L + K))
    mask = torch.ones(4, L + K)
    m.train()
    loss, logs = world_model_loss(m, obs, st, y, mask, L, K, cfg)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert set(logs) >= {"recon", "kl", "stage", "forecast", "roll_stage", "roll_recon"}


def test_causality_of_filtering(cfg):
    """Changing future observations must not change the filtered state at time t."""
    m = build_model(10, cfg).eval()
    x = torch.randn(1, 10, 10)
    x2 = x.clone()
    x2[:, 6:] += 5.0
    a = m.observe(x, sample=False)["h"][:, :6]
    b = m.observe(x2, sample=False)["h"][:, :6]
    assert torch.allclose(a, b, atol=1e-5)


def test_markov_prior():
    stages = np.array([0, 0, 1, 1, 2, 4, 4, 5, 0, 0, 1, 2, 3])
    mk = KillChainMarkov(alpha=0.1).fit(stages, np.zeros_like(stages))
    assert np.allclose(mk.P.sum(1), 1)
    b = np.eye(N_STAGES)[[0, 1]]
    r1, r5 = mk.risk_within(b, 1), mk.risk_within(b, 5)
    assert (r5 >= r1 - 1e-9).all() and (r5 <= 1).all()
    assert r1[1] > r1[0]                                # recon is closer to exploitation than benign
    path = mk.propagate(b, 3)
    assert path.shape == (2, 3, N_STAGES) and np.allclose(path.sum(-1), 1)
    mk2 = KillChainMarkov.from_dict(mk.to_dict())
    assert np.allclose(mk2.P, mk.P)
    assert set(EXPLOIT_STAGES) == {2, 3, 4, 5, 6}
