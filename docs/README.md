# SIH 26153: AI based Network Attack Forecasting (NTRO) · NetWorldModel

An offline, CPU-only system that learns a latent **world model of network traffic dynamics** and uses it to forecast
attack progression. It is trained and evaluated on **real traffic** (CTU-13, CSE-CIC-IDS2018) under leakage-safe
protocols, against re-implementations of the approaches in every public SIH-26153 repository we could read.

---## 1. Quick start

```bash
make setup        # venv + pinned dependencies + this package, installed editable
make app          # serves the dashboard
```
On Windows (no `make`): `.\make.ps1 setup` then `.\make.ps1 app`. Without either helper:
`python -m venv .venv`, then `.venv/bin/python -m pip install -r requirements.txt` and
`.venv/bin/python -m pip install -e .` (`.venv\Scripts\python` on Windows), then
`.venv/bin/python -m streamlit run app.py`.

**What ships in the repo:** trained bundles (`models/`), every report (`reports/`, `eval/results/`) and the three
bundled sample files under `data/sample/`. The demo path below therefore runs fully offline with nothing to download
or train. **What does not ship:** the processed datasets themselves (`data/processed/`), because they are rebuilt from
the public sources by `make data` (CTU-13 streams 1.9 GB and CIC-IDS2018 nine days, ~0.3 GB kept, hours of CPU);
`data/raw/` is deleted file by file as the build proceeds. Sample-based tabs work without that rebuild;
the CTU-13/CIC-2018 model tabs need `make data` first, and say so in the dashboard when a table is missing.

**A 4-minute demo path:**
1. **Sidebar → Model:** *CTU-13 · real botnet traffic · per-host · temporal split*. The host picker opens on an infected machine.
2. **Live Monitor:**
   - Drag the **replay cursor**. Right of the cursor is the future the model hasn't seen.
   - Watch the risk curve, the model's stage strip against the true stage strip, and the label-free **surprise** track.
3. **Host Triage:** every host ranked by supervised **risk** and label-free **anomaly**, as two separate channels.
4. **Forward Simulation:** 32 imagined futures for the next 10 minutes, with the stage distribution and a particle fan chart, compared against what actually happened.
5. **MITRE ATT&CK:** current vs forecast tactic probabilities, the learned transition matrix, and a forecast heat-strip.
6. **Explainability:** temporal attention, GradientSHAP drivers, and the linear-SHAP baseline next to them.
7. **Benchmark & Integrity:**
   - every method with ± over seeds and 95% CIs;
   - stage forecasting per horizon;
   - six automatic cheating checks.
8. **Sidebar → Risk score → Ensemble:** the validation-fitted blend with the best tree model.
9. **Sidebar → Upload file:** any CIC-IDS2017/2018 CSV, CTU-13 `.binetflow`, UNSW-NB15, NetFlow CSV, Parquet or PCAP/PCAPNG.

## 2. Reproduce everything

```bash
make setup
make data        # streams CTU-13 (1.9 GB) and CIC-IDS2018 (9 days) one file at a time, deletes raw files; ~0.3 GB kept
make train       # 3 seeds per dataset/protocol + all baselines  (~2.5 h CPU)
make evaluate    # reports/<dataset>/<protocol>/evaluation.md
make eval        # eval/results/comparison.md + ablation
make test        # the full pytest suite
```
On Windows the same targets exist as `.\make.ps1 setup|data|train|evaluate|eval|benchmark|test|app` (Windows
PowerShell 5.1 or PowerShell 7; from cmd.exe: `powershell -File make.ps1 setup`), or
`.\make.ps1 ctu13|cic2018|synthetic` for one dataset end to end. Neither helper is required: the Makefile resolves
`.venv/bin/python`, `make.ps1` resolves `.venv\Scripts\python.exe`, and both fall back to the system `python3`/`py`
only to create the venv.

**Single commands:**
- `python -m sih_v2 --help`, then `{build_dataset|train|evaluate|benchmark} --dataset {ctu13|cic2018|synthetic} --protocol {temporal|family}`
- `train --calibrate-only` refits the validation-only ensemble without retraining.
- console scripts are installed too: `sih-build-dataset`, `sih-train`, `sih-evaluate`, `sih-benchmark`.

**If `python -m sih_v2` reports `No module named sih_v2`:** the editable install in `.venv` was made for a different
checkout (it stores an absolute path). Re-run the setup step above *in this checkout* - it reinstalls `-e .` here.

All settings live in `config.yaml`, including per-dataset overrides and the 7 GB disk budget, which is checked after every file.

## 4. How the forward simulation works

```
last 24 min of telemetry ─► encoder → LSTM → causal Transformer        (attention = which minutes mattered)
                         ─► posterior filter  h_t = GRU(h_{t-1}, z_{t-1});  z_t ~ q(z_t | h_t, c_t)
32 particles of z_t      ─► imagination       h_{t+k} = GRU(h, z);  z_{t+k} ~ p(z | h_{t+k})   k = 1..10, no observations
stage head per step      ─► P × K × 7 stage probabilities → forecast stage, ETA, 10/50/90 % fan
fusion                   ─► w_d·direct + w_r·rollout + w_m·Markov first-passage   (weights + threshold from validation)
surprise                 ─► ‖decoder(h_t, prior mean) − x_t‖² and KL(q‖p), robust-z within capture
```

## 5. Results

All numbers are measured by us on held-out data. Every method uses identical features, labels, splits and threshold rule
(F1-optimal subject to FPR ≤ 5% on validation, frozen before test). Deep models show the mean over 3 seeds.

### CTU-13, real botnet traffic, per host, temporal split (82k test cells)
| Method | F1 | F1 95% CI | PR-AUC |
|---|---|---|---|
| Persistence (oracle current label) | 0.347 | [0.11, 0.64] | 0.199 |
| Logistic regression, stacked 8 min (HowSuyash) | 0.339 | [0.08, 0.65] | 0.203 |
| Gradient boosting, stacked (Rijja) | 0.853 | [0.49, 0.96] | 0.851 |
| **Random forest, stacked (netsight)** | 0.867 | [0.50, 0.96] | **0.861** |
| LSTM classifier, no world model | 0.804 ± 0.030 | [0.42, 0.93] | 0.837 |
| **NetWorldModel (fused)** | 0.732 ± 0.075 | [0.44, 0.92] | 0.815 |
| **NetWorldModel ⊕ random forest (ensemble)** | **0.871 ± 0.002** | [0.52, 0.96] | 0.860 |

**Stage forecasting** (macro-F1, infected host-slots):

| | +1 | +2 | +5 | +10 min |
|---|---|---|---|---|
| **NetWorldModel rollout** | 0.818 | **0.763** | 0.648 | 0.669 |
| Persistence (oracle current stage) | **0.832** | 0.755 | 0.632 | 0.581 |
| Stage LR per horizon (stacked) | 0.789 | 0.759 | **0.760** | **0.695** |

### CTU-13, held-out malware families (test scenarios 5, 8, 13, never seen in training)
| Method | F1 | PR-AUC |
|---|---|---|
| Persistence (oracle) | 0.732 | 0.572 |
| Random forest, stacked | **0.159** | **0.161** |
| NetWorldModel (fused) | 0.080 ± 0.030 | 0.062 |

**Label-free surprise** (host-level AUC): **0.853**, vs 0.882 for supervised risk.

### CSE-CIC-IDS2018, network-wide, temporal split (1.8k test cells)
| Method | F1 | F1 95% CI | PR-AUC |
|---|---|---|---|
| Random forest, stacked | 0.420 | [0.07, 0.71] | **0.628** |
| LSTM classifier | 0.446 ± 0.023 | [0.19, 0.73] | 0.530 |
| **NetWorldModel (fused)** | **0.493 ± 0.054** | [0.15, 0.77] | 0.598 |

Schedule probe: CAUTION. Timing alone reproduces about 40% of the model's lift.

### Synthetic campaigns, controlled (the only data with true pre-infection periods)
Every learned method lands at F1 0.871–0.879 and PR-AUC 0.900–0.911, which is a tie.

Our rollout gives the best stage forecast from +3 minutes onward (+10 min: 0.423, vs 0.407 for stage LR and 0.206 for oracle persistence).

### What the evidence supports
1. **Binary attack forecasting:** on real CTU-13 traffic, **tree ensembles on 8 minutes of history are the strongest single models.**
   Our world model is competitive (PR-AUC 0.815 vs 0.861) but not better. The validation-fitted **ensemble matches the best model**, so the system is never worse than the strongest baseline.
2. **Stage forecasting:**
   - The world model's imagination beats persistence at longer horizons on CTU-13 (+10 min: 0.669 vs 0.581) and synthetic data.
   - It is the best non-oracle method at +1/+2 min.
   - A per-horizon stacked classifier is still better at +5 to +10 minutes on CTU-13.
3. **Generalisation to unseen malware families fails for every supervised method** (PR-AUC ≤ 0.16). There, the **label-free surprise channel is the most useful signal** (host AUC 0.85). On CTU-13's temporal test it is uninformative (AUC ≈ 0.5), matching HowSuyash's finding that botnet traffic can be less surprising than human traffic.
4. **Integrity:**
   - Every real-data run passes the split-integrity, permutation (p < 0.005), no-peeking and OOD-benign checks.
   - CIC-IDS2018 and our synthetic generator get a CAUTION on the schedule probe.

### Disclosures
- CIC-IDS2018 uses a smaller-model override that was chosen *after* a first test run. It was motivated by the 1.7k-cell training set, and the LSTM baseline shares it.
- CTU-13 uses `input_noise: 0.1`, chosen by a validation-only sweep.
- The CTU-13 family run used the earlier noise value (0.5).
- Numbers the public repos report are listed in `eval/results/comparison.md`. They are **not comparable** (different data, cells and splits).

## 6. Layout
```
app.py                         Streamlit SOC dashboard (7 tabs)
config.yaml                    datasets, protocols, overrides, disk budget
src/sih_v2/ingest/             schema aliases, chunked CSV/Parquet, pure-Python PCAP/PCAPNG, synthetic generator
src/sih_v2/features/           per-host / network cells, multi-scale states, splits, index-based sequences
src/sih_v2/models/             NetWorldModel, losses, Markov prior, baselines, trainer (+ LSTM baseline)
src/sih_v2/engine/             simulation + surprise, calibration + ensemble, alerts, explain, metrics, leakage checks
src/sih_v2/cli/                build_dataset · train · evaluate · benchmark
eval/                          compare_baselines.py · ablation.py · results/
reports/<dataset>/<protocol>/  evaluation.md · metrics.json · host_triage.csv · stage_horizon.csv · predictions
docs/                          README.md · architecture.md · presentation_slides.md · provenance.md
tests/                         36 tests
```
