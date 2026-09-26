"""Alert generation: debounced risk crossings -> SOC alerts with MITRE mapping."""
from __future__ import annotations

import pandas as pd

from ..constants import MITRE


def severity(risk: float) -> str:
    return "CRITICAL" if risk >= 0.9 else "HIGH" if risk >= 0.75 else "MEDIUM"


def generate_alerts(frame: pd.DataFrame, threshold: float, min_consecutive: int = 1,
                    window_seconds: float = 60.0) -> pd.DataFrame:
    rows = []
    for seg, part in frame.groupby("segment", sort=False):
        on = (part["risk"] >= threshold).to_numpy()
        run, start = 0, None
        for i in range(len(part) + 1):
            active = i < len(part) and on[i]
            if active:
                run += 1
                if run == min_consecutive:
                    start = i - min_consecutive + 1
            else:
                if start is not None:
                    blk = part.iloc[start:i]
                    first = blk.iloc[0]
                    stg = first["forecast_stage"]
                    m = MITRE[stg]
                    eta = first["eta_windows"]
                    rows.append({
                        "alert_id": f"A{len(rows) + 1:04d}", "segment": seg,
                        "start": first["time"], "end": blk.iloc[-1]["time"] + pd.Timedelta(seconds=window_seconds),
                        "duration_min": len(blk) * window_seconds / 60.0,
                        "peak_risk": float(blk["risk"].max()), "severity": severity(float(blk["risk"].max())),
                        "forecast_stage": stg, "mitre_tactic": f'{m["tactic"]} ({m["id"]})',
                        "techniques": "; ".join(m["techniques"]),
                        "eta_min": float(eta * window_seconds / 60.0) if eta > 0 else None,
                        "observed_stage_at_start": first.get("current_stage_pred"),
                        "true_phase_at_start": first.get("phase"),
                    })
                run, start = 0, None
    return pd.DataFrame(rows)
