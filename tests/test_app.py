"""Headless smoke test of the Streamlit dashboard (streamlit.testing AppTest)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HAVE = any((ROOT / "models").glob("*/*/wm.pt"))
pytestmark = pytest.mark.skipif(not HAVE, reason="no trained bundle yet")


def _app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    assert not at.exception, at.exception
    return at


def test_every_model_renders_on_its_test_split():
    at = _app()
    models = at.sidebar.selectbox[0].options
    assert len(models) >= 1
    for m in models:
        at.sidebar.selectbox[0].set_value(m).run()
        assert not at.exception, (m, at.exception)
        assert len(at.tabs) == 7
        header = " ".join(x.value for x in at.markdown)
        assert "ThreatAhead" in header          # branded header of the current UI
        assert "OFFLINE" in header               # "runs fully offline" is a stated requirement


@pytest.mark.parametrize("source", ["Bundled sample - CSE-CIC-IDS2018-layout CSV (synthetic, held-out)",
                                    "Bundled sample - CTU-13-layout binetflow (synthetic)",
                                    "Bundled sample - raw PCAP capture (synthetic)"])
def test_samples_render(source):
    at = _app()
    at.sidebar.radio[0].set_value(source).run()
    assert not at.exception, at.exception
    at.slider[-1].set_value(0).run()          # move the live cursor
    assert not at.exception, at.exception


def test_ensemble_risk_mode():
    at = _app()
    radios = at.sidebar.radio
    score = next(r for r in radios if r.label == "Risk score")
    if len(score.options) > 1:
        score.set_value(score.options[-1]).run()
        assert not at.exception, at.exception
