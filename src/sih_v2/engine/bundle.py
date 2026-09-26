"""Portable model bundle (models/wm.pt): weights + scaler + Markov prior + thresholds + config."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..features.scaler import RobustScaler
from ..models.markov import KillChainMarkov
from ..models.world_model import NetWorldModel

FORMAT = "sih_v2.networldmodel/1"


@dataclass
class Bundle:
    model: NetWorldModel
    scaler: RobustScaler
    markov: KillChainMarkov
    feature_names: list
    thresholds: dict
    config: dict
    meta: dict = field(default_factory=dict)

    @property
    def threshold(self) -> float:
        return float(self.thresholds["fused"])


def save_bundle(path, b: Bundle) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": FORMAT, "hparams": b.model.hparams, "state_dict": b.model.state_dict(),
        "scaler": b.scaler.to_dict(), "markov": b.markov.to_dict(), "feature_names": list(b.feature_names),
        "thresholds": {k: float(v) for k, v in b.thresholds.items()}, "config": b.config, "meta": b.meta,
    }, path)


def load_bundle(path) -> Bundle:
    d = torch.load(path, map_location="cpu", weights_only=True)
    if d.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} bundle")
    model = NetWorldModel(**d["hparams"])
    model.load_state_dict(d["state_dict"])
    model.eval()
    return Bundle(model, RobustScaler.from_dict(d["scaler"]), KillChainMarkov.from_dict(d["markov"]),
                  d["feature_names"], d["thresholds"], d["config"], d.get("meta", {}))
