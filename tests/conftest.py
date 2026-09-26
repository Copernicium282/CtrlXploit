import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sih_v2.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def small_states(cfg):
    """Windowed states from 4 synthetic episodes (fast)."""
    from sih_v2.engine.pipeline import windows_to_states
    from sih_v2.features.windows import build_windows_from_frame
    from sih_v2.ingest import synth
    from sih_v2.ingest.schema import normalize_frame

    rng = np.random.default_rng(0)
    sc = dict(cfg["synthetic"], windows_per_episode=80, campaign_prob=1.0, stealth_prob=0.0)
    eps = [synth.episode(rng, sc, 1.5e9 + i * 86400) for i in range(4)]
    raw = pd.concat(eps, ignore_index=True)
    raw["Timestamp"] = raw["ts"]
    flows = normalize_frame(raw[synth.CIC_COLUMNS])
    w = build_windows_from_frame(flows, cfg["features"]["window_seconds"])
    return windows_to_states(w, cfg)
