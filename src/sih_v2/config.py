"""Configuration loading, dataset/protocol resolution and project path handling"""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DATASETS = ("ctu13", "cic2018", "synthetic")
PROTOCOLS = ("temporal", "family")


def load_config(path: str | os.PathLike | None = None, dataset: str | None = None,
                protocol: str | None = None) -> dict[str, Any]:
    """Load config.yaml and resolve it for one dataset + evaluation protocol.

    `cfg["data"]` holds the dataset block; every `paths` entry has {dataset} and
    {protocol} substituted."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = copy.deepcopy(raw)
    cfg["dataset"] = dataset or raw.get("dataset", "ctu13")
    cfg["protocol"] = protocol or raw.get("protocol", "temporal")
    if cfg["dataset"] not in cfg.get("datasets", {}):
        raise ValueError(f"unknown dataset {cfg['dataset']!r}; known: {list(cfg.get('datasets', {}))}")
    cfg["data"] = cfg["datasets"][cfg["dataset"]]
    for section, vals in (cfg["data"].get("overrides") or {}).items():   # per-dataset hyper-parameters
        cfg.setdefault(section, {}).update(vals)
    cfg["paths_template"] = dict(raw["paths"])
    cfg["paths"] = {k: (v.format(dataset=cfg["dataset"], protocol=cfg["protocol"]) if isinstance(v, str) else v)
                    for k, v in raw["paths"].items()}
    cfg.setdefault("engine", {}).setdefault("rollout_steps", cfg["forecast"]["horizon"])
    return cfg


def seed_paths(cfg: dict, seed: int) -> dict:
    """Per-seed artefact paths (the primary bundle stays at paths.model_bundle)."""
    base = resolve(cfg["paths"]["model_dir"]) / "seeds" / f"seed{seed}"
    return {"model_bundle": base / "wm.pt", "dir": base}


def resolve(p: str | os.PathLike) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass
