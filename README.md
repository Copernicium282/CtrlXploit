# SIH 26153 · AI based Network Attack Forecasting (NTRO): NetWorldModel

Per-host latent world model of network traffic that forecasts attack progression, with K-step particle simulation,
MITRE ATT&CK mapping, a label-free surprise channel and leakage-checked evaluation on **real CTU-13 and CSE-CIC-IDS2018
traffic**. Runs 100% offline on CPU.

```bash
make setup && make fetch-data && make app           # venv + deps, prebuilt datasets, dashboard
make setup && make app                              # samples only - dashboard/tests work without the fetch
make all                                            # full rebuild from public data (~3 h CPU, <0.5 GB disk)
```
On Windows run `.\make.ps1 setup`, `.\make.ps1 fetch-data` and `.\make.ps1 app` in PowerShell (Windows PowerShell 5.1 or
PowerShell 7) instead of `make` - no `make` required either way: see [docs/README.md](docs/README.md) §1 for the three raw
commands.

**What ships:** trained bundles (`models/`), all reports and the bundled samples in `data/sample/` - the dashboard
runs offline out of the box. **What is fetched:** the processed datasets (`data/processed/`, 260 MB incl. the 123 MB
CTU-13 cells file) live on Hugging Face at [ir192m2Cn282/ThreatAhead](https://huggingface.co/datasets/ir192m2Cn282/ThreatAhead)
and are pulled by `make fetch-data`. **What is rebuilt:** `make data` regenerates them from the public sources
(~3 h CPU) if you prefer not to fetch.

| CTU-13 real traffic, per host, temporal split | F1 | PR-AUC |
|---|---|---|
| Random forest on stacked history (strongest baseline) | 0.867 | 0.861 |
| NetWorldModel (fused), 3 seeds | 0.732 ± 0.075 | 0.815 |
| **NetWorldModel ⊕ forest ensemble** (validation-fitted) | **0.871 ± 0.002** | 0.860 |

- **Stage forecasting:** the world model beats persistence at +10 min (0.669 vs 0.581). A stacked per-horizon classifier is better at +5 to +10 min.
- **Unseen malware families:** supervised methods fail (PR-AUC ≤ 0.16), but the label-free surprise channel still reaches host AUC 0.85.

Full guide and all results: [docs/README.md](docs/README.md) · Architecture: [docs/architecture.md](docs/architecture.md) · Credits: [docs/provenance.md](docs/provenance.md)
