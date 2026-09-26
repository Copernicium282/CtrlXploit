# Evaluation · ctu13 · temporal protocol

Test cells: 82,117. Target: exploitation-stage cell within the next 10 windows. Thresholds frozen on validation (F1-optimal s.t. FPR ≤ 0.05). Deep models: mean over seeds (± = std); 95 % CI = bootstrap over host-slots/segments for the primary seed.

## Binary forecasting

| Method | Seeds | F1 | F1 ± | F1 95% CI | Precision | Recall | FPR | ROC-AUC | PR-AUC | PR-AUC ± | EW PR-AUC | Pre-attack recall | During-attack recall | Incidents warned | Mean lead (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Persistence (oracle current label) | 1 | 0.347 | 0.000 | [0.114, 0.641] | 0.213 | 0.937 | 0.015 | 0.961 | 0.199 | 0.000 | 0.035 | 0.729 | 1.000 | 0.750 | 3.125 |
| Logistic Regression (single window) | 1 | 0.370 | 0.000 | [0.106, 0.702] | 0.235 | 0.871 | 0.013 | 0.944 | 0.193 | 0.000 | 0.027 | 0.588 | 0.942 | 0.375 | 2.500 |
| Logistic Regression (stacked 8) | 1 | 0.339 | 0.000 | [0.081, 0.646] | 0.212 | 0.857 | 0.014 | 0.939 | 0.203 | 0.000 | 0.038 | 0.576 | 0.931 | 0.500 | 3.000 |
| Gradient Boosting (stacked 8) | 1 | 0.853 | 0.000 | [0.485, 0.956] | 0.946 | 0.777 | 0.000 | 0.947 | 0.851 | 0.000 | 0.322 | 0.224 | 0.924 | 0.250 | 0.250 |
| Random Forest (stacked 8) | 1 | 0.867 | 0.000 | [0.500, 0.963] | 0.945 | 0.802 | 0.000 | 0.959 | 0.861 | 0.000 | 0.370 | 0.294 | 0.931 | 0.250 | 1.375 |
| LSTM classifier (no world model) | 3 | 0.804 | 0.030 | [0.422, 0.930] | 0.784 | 0.828 | 0.001 | 0.961 | 0.837 | 0.005 | 0.275 | 0.384 | 0.943 | 0.167 | 1.292 |
| NetWorldModel - direct head | 3 | 0.750 | 0.075 | [0.443, 0.924] | 0.689 | 0.838 | 0.002 | 0.953 | 0.820 | 0.006 | 0.253 | 0.408 | 0.950 | 0.250 | 1.333 |
| NetWorldModel - K-step rollout | 3 | 0.707 | 0.094 | [0.410, 0.915] | 0.635 | 0.825 | 0.002 | 0.955 | 0.813 | 0.010 | 0.239 | 0.396 | 0.936 | 0.250 | 1.333 |
| Markov kill-chain prior | 3 | 0.715 | 0.097 | [0.411, 0.921] | 0.647 | 0.826 | 0.002 | 0.962 | 0.804 | 0.013 | 0.180 | 0.400 | 0.937 | 0.292 | 1.375 |
| NetWorldModel - fused (ours) | 3 | 0.732 | 0.075 | [0.443, 0.924] | 0.670 | 0.825 | 0.002 | 0.951 | 0.815 | 0.013 | 0.217 | 0.396 | 0.936 | 0.250 | 1.333 |
| NetWorldModel ⊕ tree ensemble (ours, ensemble) | 3 | 0.871 | 0.002 | [0.517, 0.963] | 0.951 | 0.803 | 0.000 | 0.961 | 0.860 | 0.003 | 0.372 | 0.298 | 0.931 | 0.250 | 1.375 |

## Kill-chain stage forecasting (macro-F1 over stages present, malicious host-slots)

| method | +1 | +2 | +3 | +5 | +7 | +10 |
|---|---|---|---|---|---|---|
| Markov kill-chain prior | 0.806 | 0.707 | 0.635 | 0.643 | 0.654 | 0.383 |
| NetWorldModel rollout (ours) | 0.818 | 0.763 | 0.636 | 0.648 | 0.658 | 0.669 |
| Persistence (inferred current stage) | 0.796 | 0.741 | 0.692 | 0.630 | 0.632 | 0.635 |
| Persistence (oracle current stage) | 0.832 | 0.755 | 0.685 | 0.632 | 0.604 | 0.581 |
| Stage LR per horizon (stacked) | 0.789 | 0.759 | 0.657 | 0.760 | 0.753 | 0.695 |

Accuracy on cells whose stage *changes* by t+k (persistence scores 0 here by construction):

| method | +1 | +2 | +3 | +5 | +7 | +10 |
|---|---|---|---|---|---|---|
| Markov kill-chain prior | 0.358 | 0.289 | 0.382 | 0.438 | 0.483 | 0.549 |
| NetWorldModel rollout (ours) | 0.408 | 0.393 | 0.368 | 0.438 | 0.483 | 0.530 |
| Persistence (inferred current stage) | 0.350 | 0.311 | 0.319 | 0.325 | 0.351 | 0.390 |
| Persistence (oracle current stage) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Stage LR per horizon (stacked) | 0.383 | 0.363 | 0.431 | 0.615 | 0.632 | 0.530 |

## Host-level triage (two channels)

- infected host-slots caught: **7/46** (risk channel 7, anomaly channel only 0)
- false alarms: 12/1694 benign host-slots (0.7%); risk channel alone 9

## Label-free surprise channel

- direction chosen on validation: sign +1 (validation raw AUC 0.580)
- test window-level AUC (malicious vs benign cells): 0.496 (KL variant 0.317)
- test host-level AUC: surprise 0.352 vs supervised risk 0.474

## Integrity checks

- **split integrity (temporal)** - OK: no overlap  `{'issues': []}`
- **schedule-only probe (hour, weekday, elapsed time)** - OK: timing alone is far weaker than the traffic model  `{'schedule_ap': 0.0098, 'prevalence': 0.0044, 'model_ap': 0.8145, 'schedule_share_of_lift': 0.0066}`
- **capture-identity probe** - reported for context: which capture you are in carries this much signal  `{'identity_ap': 0.0143, 'prevalence': 0.0044}`
- **label-permutation test of test PR-AUC** - OK: p < 0.00498 (signal above label-permuted chance)  `{'observed_ap': 0.8113, 'null_mean': 0.0046, 'null_p99': 0.0054, 'p_value': 0.005}`
- **no-peeking gradient test** - OK: imagination never touches observations (and the check can fail)  `{'encoder_grad_detached': 0.0, 'encoder_grad_wired': 5241.6055}`
- **out-of-distribution benign (flash crowd x5; novel values in ['fin_ratio', 'psh_ratio', 'ack_ratio'])** - OK: risk channel stays quiet on unusual benign traffic  `{'cells': 1643, 'alert_rate_original': 0.0, 'alert_rate_flash_crowd': 0.0, 'alert_rate_novel_values': 0.0, 'anomaly_rate_flash_crowd': 0.0578, 'anomaly_rate_novel_values': 0.0201}`
