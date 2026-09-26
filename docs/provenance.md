# Provenance: where every idea came from

We read the six publicly accessible SIH-26153 repositories on 2026-09-24 (shallow clones of their default branches).
**No code was copied.** Four of the six publish no licence (all rights reserved). PratikBorle/miniproject and
Rijja-explore are MIT-licensed, but we still re-implemented everything ourselves. What we took are *ideas*, credited below.
`GMinnu/SIH26153` and `BuzyU/NetForecast` return HTTP 404 and could not be read.

## Ideas adopted from the public repos

| Idea | Source repo | Where it lives here |
|---|---|---|
| Per-host state cells (one trajectory per source host, not one per network) | HowSuyash/AttackForecast; csxzor-devcs/26153 (per-host attack onset) | `features/windows.py` (host mode), `datasets.ctu13` |
| CTU-13 as primary real dataset, flags recovered from Argus `State` strings | HowSuyash/AttackForecast | `ingest/schema.py`, `cli/build_dataset.py` |
| Stage labels derived from CTU-13's own per-flow behaviour annotations | HowSuyash/AttackForecast (we map SPAM → Impact/T1496 rather than Exfiltration) | `constants._ctu_botnet_stage` |
| *Dominant*-stage window labelling (spam bots would otherwise "teleport" to the end of the kill chain) | HowSuyash/AttackForecast | `windows.stage_from_counts(rule="dominant")` |
| History-stacked logistic-regression baseline ("sees the same history the world model does") | HowSuyash/AttackForecast | `models/baselines.py` |
| Stage forecasting vs persistence, reported per horizon | HowSuyash/AttackForecast | `engine/metrics.stage_horizon_table` |
| Falsifiable no-peeking test (imagination-only loss ⇒ zero encoder gradient; wired version ⇒ non-zero) | HowSuyash/AttackForecast (`tests/prove_no_peeking.py`) | `engine/leakage.no_peeking`, `tests/test_rigor.py` |
| Family-held-out split on CTU-13 (train 1-4,6,7,9-11 · val 12 · test 5,8,13) | HowSuyash/AttackForecast | `config.yaml → datasets.ctu13.family_split` |
| Label-free "model surprise" channel, robust median/MAD z-scores, dual-channel host triage | HowSuyash/AttackForecast; ShadowCat ("NLL novelty"); PratikBorle (transition-reconstruction novelty) | `engine/simulate.py`, `engine/metrics.host_triage` |
| Pre-attack vs during-attack separation in every metric | csxzor-devcs/26153 (attack-semantics module) | `features/states.add_targets`, `engine/metrics.phase_metrics` |
| Persistence as the forecasting reference that must be beaten; claim-safety discipline | Rijja-explore/Network-Attack-Forecasting | baselines + honest reporting |
| Gradient-boosting (lagged XGBoost) baseline | Rijja-explore | `Gradient Boosting (stacked 8)` |
| Random-forest forecaster on rolling-window features | ayushshandilya-dev/netsight | `Random Forest (stacked 8)` |
| Multi-scale rolling features (w ∈ {3,6,12} means/std/slopes) | ayushshandilya-dev/netsight | our instant / short-EMA deviation / long-EMA trend state |
| Plain LSTM sequence classifier as the "no world model" ablation | PratikBorle/miniproject, csxzor-devcs/26153, ShadowCat | `models/trainer.SeqClassifier` |
| Robust scaling with a variance floor + clipping; regression test for unusual benign traffic; diverse-benign augmentation | PratikBorle/miniproject | `features/scaler.py`, `engine/leakage.ood_benign`, `benign_volume_aug` |
| Schedule-leakage probe (hour/day-only model vs traffic model) | muthukkumaranb/ShadowCat (Gate-0 report) | `engine/leakage.schedule_probe` |
| Purge gaps around chronological cuts; episode-grouped evaluation | ShadowCat (purge + embargo), López de Prado (2018) | `features/splits.py` |

## Ideas we deliberately did **not** adopt

* **Hash-chain / blockchain audit ledger** (ShadowCat). Tamper-evidence is useful operationally, but it doesn't change forecasting quality. Left as future work.
* **GNN fusion branch** (ShadowCat). Its own ablation kept it out of the primary path.
* **Live packet sniffing with root privileges** (netsight). Offline replay of recorded captures is safer for judging, and the PCAP path already covers raw packets.
* **Model-surprise as a risk score.** HowSuyash showed surprise *anti*-detects botnets on CTU-13, because machine traffic is more regular than human traffic. We therefore keep it as a separate channel and **choose its sign on validation**.

## Literature

Ha & Schmidhuber, *World Models* (2018) · Hafner et al., *PlaNet* (ICML 2019) and *Dreamer* (ICLR 2020): RSSM, latent overshooting, free nats ·
Kingma & Welling (2014); Kingma et al. (2016, free bits) · Hochreiter & Schmidhuber (1997); Cho et al. (2014); Vaswani et al. (2017) ·
Erion et al., *Expected Gradients* (Nat. Mach. Intell. 2021); Sundararajan et al. (2017); Lundberg & Lee (2017) ·
García et al., *An empirical comparison of botnet detection methods*, Computers & Security 2014 (CTU-13, CC BY 2.0) ·
Sharafaldin et al. (CSE-CIC-IDS2018) · Husák et al., *Survey of Attack Projection, Prediction, and Forecasting* (IEEE COMST 2019) ·
Hutchins et al. (2011, Cyber Kill Chain) · MITRE ATT&CK · López de Prado, *Advances in Financial Machine Learning* (2018, purging).
