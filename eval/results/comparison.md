# Comparison

## A. Measured - every approach re-run on identical data and protocol

### cic2018 · temporal (1,820 test cells, seeds [0, 1, 2])

| Method | Approach represented | Seeds | F1 | F1 ± | F1 95% CI | FPR | PR-AUC | PR-AUC ± | EW PR-AUC | Pre-attack recall | Mean lead (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Persistence (oracle current label) | Rijja-explore (its strongest validated forecaster) | 1 | 0.933 | 0.000 | [0.877, 0.970] | 0.007 | 0.910 | 0.000 | 0.051 | 0.000 | 0.000 |
| Logistic Regression (single window) | PS-mandated baseline (all repos) | 1 | 0.198 | 0.000 | [0.003, 0.452] | 0.029 | 0.458 | 0.000 | 0.124 | 0.152 | 1.667 |
| Logistic Regression (stacked 8) | HowSuyash/AttackForecast (stacked LR) | 1 | 0.162 | 0.000 | [0.017, 0.437] | 0.035 | 0.440 | 0.000 | 0.093 | 0.152 | 1.667 |
| Gradient Boosting (stacked 8) | Rijja-explore (lagged XGBoost) | 1 | 0.340 | 0.000 | [0.067, 0.577] | 0.103 | 0.526 | 0.000 | 0.097 | 0.258 | 1.167 |
| Random Forest (stacked 8) | ayushshandilya-dev/netsight (RF forecaster) | 1 | 0.420 | 0.000 | [0.065, 0.706] | 0.090 | 0.628 | 0.000 | 0.105 | 0.318 | 2.833 |
| LSTM classifier (no world model) | PratikBorle / csxzor-devcs / ShadowCat (LSTM family) | 3 | 0.446 | 0.023 | [0.192, 0.730] | 0.111 | 0.530 | 0.012 | 0.167 | 0.348 | 3.333 |
| NetWorldModel - direct head | ours (component) | 3 | 0.499 | 0.068 | [0.142, 0.793] | 0.106 | 0.590 | 0.051 | 0.081 | 0.177 | 1.889 |
| NetWorldModel - K-step rollout | ours (component) · HowSuyash-style imagination | 3 | 0.496 | 0.053 | [0.152, 0.772] | 0.107 | 0.598 | 0.045 | 0.082 | 0.177 | 1.722 |
| Markov kill-chain prior | ours (component) | 3 | 0.499 | 0.063 | [0.142, 0.787] | 0.103 | 0.599 | 0.047 | 0.086 | 0.202 | 2.000 |
| NetWorldModel - fused (ours) | ours | 3 | 0.493 | 0.054 | [0.152, 0.772] | 0.118 | 0.598 | 0.045 | 0.082 | 0.192 | 1.778 |
| NetWorldModel ⊕ tree ensemble (ours, ensemble) |  | 3 | 0.493 | 0.054 | [0.152, 0.772] | 0.118 | 0.598 | 0.045 | 0.082 | 0.192 | 1.778 |

Stage forecasting, macro-F1 per horizon (malicious host-slots):

| method | +1 | +2 | +3 | +5 | +7 | +10 |
|---|---|---|---|---|---|---|
| Markov kill-chain prior | 0.156 | 0.156 | 0.156 | 0.157 | 0.157 | 0.157 |
| NetWorldModel rollout (ours) | 0.156 | 0.156 | 0.156 | 0.157 | 0.157 | 0.157 |
| Persistence (inferred current stage) | 0.156 | 0.156 | 0.156 | 0.157 | 0.157 | 0.157 |
| Persistence (oracle current stage) | 0.986 | 0.979 | 0.969 | 0.951 | 0.930 | 0.900 |
| Stage LR per horizon (stacked) | 0.522 | 0.506 | 0.496 | 0.505 | 0.470 | 0.434 |

Host triage: 5/8 infected host-slots caught (0 only by the label-free channel); 0/3 benign host-slots flagged.

Integrity checks: 5 passed, flagged: schedule-only probe (hour, weekday, elapsed time) - CAUTION: timing carries part of the signal

### ctu13 · family (137,887 test cells, seeds [0, 1, 2])

| Method | Approach represented | Seeds | F1 | F1 ± | F1 95% CI | FPR | PR-AUC | PR-AUC ± | EW PR-AUC | Pre-attack recall | Mean lead (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Persistence (oracle current label) | Rijja-explore (its strongest validated forecaster) | 1 | 0.732 | 0.000 | [0.217, 0.990] | 0.006 | 0.572 | 0.000 | 0.112 | 0.832 | 4.857 |
| Logistic Regression (single window) | PS-mandated baseline (all repos) | 1 | 0.029 | 0.000 | [0.000, 0.072] | 0.061 | 0.054 | 0.000 | 0.028 | 0.317 | 4.714 |
| Logistic Regression (stacked 8) | HowSuyash/AttackForecast (stacked LR) | 1 | 0.032 | 0.000 | [0.000, 0.087] | 0.050 | 0.072 | 0.000 | 0.058 | 0.292 | 4.286 |
| Gradient Boosting (stacked 8) | Rijja-explore (lagged XGBoost) | 1 | 0.153 | 0.000 | [0.010, 0.246] | 0.026 | 0.125 | 0.000 | 0.072 | 0.590 | 5.000 |
| Random Forest (stacked 8) | ayushshandilya-dev/netsight (RF forecaster) | 1 | 0.159 | 0.000 | [0.009, 0.278] | 0.031 | 0.161 | 0.000 | 0.158 | 0.764 | 7.143 |
| LSTM classifier (no world model) | PratikBorle / csxzor-devcs / ShadowCat (LSTM family) | 3 | 0.091 | 0.026 | [0.000, 0.136] | 0.016 | 0.078 | 0.004 | 0.048 | 0.325 | 5.476 |
| NetWorldModel - direct head | ours (component) | 3 | 0.089 | 0.020 | [0.001, 0.138] | 0.017 | 0.085 | 0.010 | 0.067 | 0.317 | 5.143 |
| NetWorldModel - K-step rollout | ours (component) · HowSuyash-style imagination | 3 | 0.079 | 0.029 | [0.002, 0.084] | 0.032 | 0.057 | 0.016 | 0.050 | 0.284 | 4.667 |
| Markov kill-chain prior | ours (component) | 3 | 0.096 | 0.030 | [0.003, 0.119] | 0.026 | 0.076 | 0.015 | 0.051 | 0.317 | 5.095 |
| NetWorldModel - fused (ours) | ours | 3 | 0.080 | 0.030 | [0.002, 0.084] | 0.032 | 0.062 | 0.017 | 0.048 | 0.288 | 4.667 |
| NetWorldModel ⊕ tree ensemble (ours, ensemble) |  | 3 | 0.078 | 0.029 | [0.002, 0.084] | 0.032 | 0.068 | 0.011 | 0.059 | 0.290 | 4.667 |

Stage forecasting, macro-F1 per horizon (malicious host-slots):

| method | +1 | +2 | +3 | +5 | +7 | +10 |
|---|---|---|---|---|---|---|
| Markov kill-chain prior | 0.216 | 0.208 | 0.202 | 0.164 | 0.065 | 0.059 |
| NetWorldModel rollout (ours) | 0.208 | 0.207 | 0.203 | 0.199 | 0.190 | 0.186 |
| Persistence (inferred current stage) | 0.210 | 0.205 | 0.204 | 0.193 | 0.190 | 0.192 |
| Persistence (oracle current stage) | 0.725 | 0.620 | 0.554 | 0.510 | 0.512 | 0.500 |
| Stage LR per horizon (stacked) | 0.207 | 0.210 | 0.213 | 0.195 | 0.180 | 0.166 |

Host triage: 4/5 infected host-slots caught (0 only by the label-free channel); 294/1058 benign host-slots flagged.

Integrity checks: 6 passed

### ctu13 · temporal (82,117 test cells, seeds [0, 1, 2])

| Method | Approach represented | Seeds | F1 | F1 ± | F1 95% CI | FPR | PR-AUC | PR-AUC ± | EW PR-AUC | Pre-attack recall | Mean lead (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Persistence (oracle current label) | Rijja-explore (its strongest validated forecaster) | 1 | 0.347 | 0.000 | [0.114, 0.641] | 0.015 | 0.199 | 0.000 | 0.035 | 0.729 | 3.125 |
| Logistic Regression (single window) | PS-mandated baseline (all repos) | 1 | 0.370 | 0.000 | [0.106, 0.702] | 0.013 | 0.193 | 0.000 | 0.027 | 0.588 | 2.500 |
| Logistic Regression (stacked 8) | HowSuyash/AttackForecast (stacked LR) | 1 | 0.339 | 0.000 | [0.081, 0.646] | 0.014 | 0.203 | 0.000 | 0.038 | 0.576 | 3.000 |
| Gradient Boosting (stacked 8) | Rijja-explore (lagged XGBoost) | 1 | 0.853 | 0.000 | [0.485, 0.956] | 0.000 | 0.851 | 0.000 | 0.322 | 0.224 | 0.250 |
| Random Forest (stacked 8) | ayushshandilya-dev/netsight (RF forecaster) | 1 | 0.867 | 0.000 | [0.500, 0.963] | 0.000 | 0.861 | 0.000 | 0.370 | 0.294 | 1.375 |
| LSTM classifier (no world model) | PratikBorle / csxzor-devcs / ShadowCat (LSTM family) | 3 | 0.804 | 0.030 | [0.422, 0.930] | 0.001 | 0.837 | 0.005 | 0.275 | 0.384 | 1.292 |
| NetWorldModel - direct head | ours (component) | 3 | 0.750 | 0.075 | [0.443, 0.924] | 0.002 | 0.820 | 0.006 | 0.253 | 0.408 | 1.333 |
| NetWorldModel - K-step rollout | ours (component) · HowSuyash-style imagination | 3 | 0.707 | 0.094 | [0.410, 0.915] | 0.002 | 0.813 | 0.010 | 0.239 | 0.396 | 1.333 |
| Markov kill-chain prior | ours (component) | 3 | 0.715 | 0.097 | [0.411, 0.921] | 0.002 | 0.804 | 0.013 | 0.180 | 0.400 | 1.375 |
| NetWorldModel - fused (ours) | ours | 3 | 0.732 | 0.075 | [0.443, 0.924] | 0.002 | 0.815 | 0.013 | 0.217 | 0.396 | 1.333 |
| NetWorldModel ⊕ tree ensemble (ours, ensemble) |  | 3 | 0.871 | 0.002 | [0.517, 0.963] | 0.000 | 0.860 | 0.003 | 0.372 | 0.298 | 1.375 |

Stage forecasting, macro-F1 per horizon (malicious host-slots):

| method | +1 | +2 | +3 | +5 | +7 | +10 |
|---|---|---|---|---|---|---|
| Markov kill-chain prior | 0.806 | 0.707 | 0.635 | 0.643 | 0.654 | 0.383 |
| NetWorldModel rollout (ours) | 0.818 | 0.763 | 0.636 | 0.648 | 0.658 | 0.669 |
| Persistence (inferred current stage) | 0.796 | 0.741 | 0.692 | 0.630 | 0.632 | 0.635 |
| Persistence (oracle current stage) | 0.832 | 0.755 | 0.685 | 0.632 | 0.604 | 0.581 |
| Stage LR per horizon (stacked) | 0.789 | 0.759 | 0.657 | 0.760 | 0.753 | 0.695 |

Host triage: 7/46 infected host-slots caught (0 only by the label-free channel); 12/1694 benign host-slots flagged.

Integrity checks: 6 passed

### synthetic · temporal (3,200 test cells, seeds [0, 1, 2])

| Method | Approach represented | Seeds | F1 | F1 ± | F1 95% CI | FPR | PR-AUC | PR-AUC ± | EW PR-AUC | Pre-attack recall | Mean lead (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Persistence (oracle current label) | Rijja-explore (its strongest validated forecaster) | 1 | 0.860 | 0.000 | [0.833, 0.890] | 0.029 | 0.782 | 0.000 | 0.357 | 0.444 | 1.333 |
| Logistic Regression (single window) | PS-mandated baseline (all repos) | 1 | 0.879 | 0.000 | [0.842, 0.917] | 0.015 | 0.903 | 0.000 | 0.578 | 0.448 | 4.458 |
| Logistic Regression (stacked 8) | HowSuyash/AttackForecast (stacked LR) | 1 | 0.871 | 0.000 | [0.827, 0.911] | 0.013 | 0.899 | 0.000 | 0.565 | 0.414 | 4.125 |
| Gradient Boosting (stacked 8) | Rijja-explore (lagged XGBoost) | 1 | 0.879 | 0.000 | [0.844, 0.916] | 0.020 | 0.900 | 0.000 | 0.568 | 0.477 | 4.667 |
| Random Forest (stacked 8) | ayushshandilya-dev/netsight (RF forecaster) | 1 | 0.877 | 0.000 | [0.839, 0.916] | 0.014 | 0.902 | 0.000 | 0.571 | 0.431 | 4.292 |
| LSTM classifier (no world model) | PratikBorle / csxzor-devcs / ShadowCat (LSTM family) | 3 | 0.871 | 0.001 | [0.834, 0.911] | 0.017 | 0.911 | 0.003 | 0.602 | 0.464 | 4.625 |
| NetWorldModel - direct head | ours (component) | 3 | 0.877 | 0.002 | [0.840, 0.919] | 0.016 | 0.906 | 0.001 | 0.585 | 0.460 | 4.583 |
| NetWorldModel - K-step rollout | ours (component) · HowSuyash-style imagination | 3 | 0.838 | 0.005 | [0.800, 0.883] | 0.040 | 0.894 | 0.001 | 0.508 | 0.462 | 4.569 |
| Markov kill-chain prior | ours (component) | 3 | 0.846 | 0.008 | [0.811, 0.890] | 0.037 | 0.884 | 0.004 | 0.350 | 0.455 | 4.486 |
| NetWorldModel - fused (ours) | ours | 3 | 0.877 | 0.002 | [0.840, 0.919] | 0.016 | 0.906 | 0.001 | 0.585 | 0.460 | 4.583 |
| NetWorldModel ⊕ tree ensemble (ours, ensemble) |  | 3 | 0.877 | 0.001 | [0.841, 0.916] | 0.014 | 0.903 | 0.001 | 0.572 | 0.437 | 4.347 |

Stage forecasting, macro-F1 per horizon (malicious host-slots):

| method | +1 | +2 | +3 | +5 | +7 | +10 |
|---|---|---|---|---|---|---|
| Markov kill-chain prior | 0.501 | 0.462 | 0.399 | 0.329 | 0.239 | 0.148 |
| NetWorldModel rollout (ours) | 0.569 | 0.550 | 0.535 | 0.516 | 0.476 | 0.423 |
| Persistence (inferred current stage) | 0.575 | 0.522 | 0.468 | 0.368 | 0.273 | 0.194 |
| Persistence (oracle current stage) | 0.844 | 0.702 | 0.575 | 0.403 | 0.298 | 0.206 |
| Stage LR per horizon (stacked) | 0.763 | 0.623 | 0.565 | 0.523 | 0.471 | 0.407 |

Host triage: 15/15 infected host-slots caught (0 only by the label-free channel); 0/1 benign host-slots flagged.

Integrity checks: 5 passed, flagged: schedule-only probe (hour, weekday, elapsed time) - CAUTION: timing carries part of the signal

## B. Reported by the public repos (their own data/protocols - NOT comparable)

| Repo | Their setting | Their reported result |
|---|---|---|
| HowSuyash/AttackForecast | CTU-13, 13 scenarios, per-host 60 s cells, temporal 70/15/15 | RSSM world model F1 0.979, AP 0.995; stacked LR F1 0.977; single LR 0.963. Family holdout (5,8,13): WM F1 0.874 / AP 0.917 vs single LR F1 0.901. Stage forecast beats persistence at 9/10 horizons. Host triage 28/30 infected caught, 74/1500 false alarms. |
| PratikBorle/miniproject | CSE-CIC-IDS2018 Thursday-01-03 only, 10 s windows, 65/35 chronological | LSTM F1@5%FPR 0.664 vs LR 0.230; PR-AUC 0.683 vs 0.665; ROC-AUC 0.802 vs 0.813. |
| csxzor-devcs/26153 | bundled synthetic generator only | Real-data benchmark not run; its results file states synthetic results must not be cited and its test split had 0 pre-attack sequences. |
| Rijja-explore/Network-Attack-Forecasting | CTU-13-derived flow windows + CTU PCAP family states | XGBoost current-risk F1 0.941 (val); persistence forecasting F1 0.975 beats lagged XGBoost 0.919; explicitly claims no lead time and no world model. |
| ayushshandilya-dev/netsight | CIC-IDS2017, all 8 days, 500-flow windows, cross-day holdout | RF forecaster ROC-AUC 0.763, F1 0.375 @0.5; walk-forward ROC-AUC 0.48-0.85; LR cross-day ROC-AUC 0.539. |
| muthukkumaranb/ShadowCat | CSE-CIC-IDS2018, 1-min windows, LOEO / episode-grouped | Pre-onset forecasting F1 0.0 in LOEO; hour+day-only features beat traffic features on chronological holdout (F1 0.689 vs 0.044) - schedule leakage in CIC-IDS2018. |
| GMinnu/SIH26153, BuzyU/NetForecast | - | not publicly accessible (HTTP 404) |

The closest like-for-like reference is HowSuyash/AttackForecast (same dataset, per-host 60 s cells, the same held-out family scenarios 5/8/13). Its cell construction, label mapping and positives differ from ours (e.g. it maps SPAM to Exfiltration, we map it to Impact/T1496), so even there numbers are indicative only.
