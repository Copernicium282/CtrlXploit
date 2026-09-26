# Evaluation · cic2018 · temporal protocol

Test cells: 1,820. Target: exploitation-stage cell within the next 10 windows. Thresholds frozen on validation (F1-optimal s.t. FPR ≤ 0.05). Deep models: mean over seeds (± = std); 95 % CI = bootstrap over host-slots/segments for the primary seed.

## Binary forecasting

| Method | Seeds | F1 | F1 ± | F1 95% CI | Precision | Recall | FPR | ROC-AUC | PR-AUC | PR-AUC ± | EW PR-AUC | Pre-attack recall | During-attack recall | Incidents warned | Mean lead (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Persistence (oracle current label) | 1 | 0.933 | 0.000 | [0.877, 0.970] | 0.983 | 0.889 | 0.007 | 0.941 | 0.910 | 0.000 | 0.051 | 0.000 | 1.000 | 0.000 | 0.000 |
| Logistic Regression (single window) | 1 | 0.198 | 0.000 | [0.003, 0.452] | 0.663 | 0.117 | 0.029 | 0.533 | 0.458 | 0.000 | 0.124 | 0.152 | 0.112 | 0.167 | 1.667 |
| Logistic Regression (stacked 8) | 1 | 0.162 | 0.000 | [0.017, 0.437] | 0.566 | 0.095 | 0.035 | 0.542 | 0.440 | 0.000 | 0.093 | 0.152 | 0.088 | 0.167 | 1.667 |
| Gradient Boosting (stacked 8) | 1 | 0.340 | 0.000 | [0.067, 0.577] | 0.538 | 0.248 | 0.103 | 0.676 | 0.526 | 0.000 | 0.097 | 0.258 | 0.249 | 0.333 | 1.167 |
| Random Forest (stacked 8) | 1 | 0.420 | 0.000 | [0.065, 0.706] | 0.628 | 0.316 | 0.090 | 0.734 | 0.628 | 0.000 | 0.105 | 0.318 | 0.314 | 0.333 | 2.833 |
| LSTM classifier (no world model) | 3 | 0.446 | 0.023 | [0.192, 0.730] | 0.617 | 0.355 | 0.111 | 0.542 | 0.530 | 0.012 | 0.167 | 0.348 | 0.355 | 0.333 | 3.333 |
| NetWorldModel - direct head | 3 | 0.499 | 0.068 | [0.142, 0.793] | 0.664 | 0.403 | 0.106 | 0.626 | 0.590 | 0.051 | 0.081 | 0.177 | 0.429 | 0.222 | 1.889 |
| NetWorldModel - K-step rollout | 3 | 0.496 | 0.053 | [0.152, 0.772] | 0.648 | 0.402 | 0.107 | 0.634 | 0.598 | 0.045 | 0.082 | 0.177 | 0.428 | 0.278 | 1.722 |
| Markov kill-chain prior | 3 | 0.499 | 0.063 | [0.142, 0.787] | 0.670 | 0.401 | 0.103 | 0.632 | 0.599 | 0.047 | 0.086 | 0.202 | 0.425 | 0.222 | 2.000 |
| NetWorldModel - fused (ours) | 3 | 0.493 | 0.054 | [0.152, 0.772] | 0.634 | 0.406 | 0.118 | 0.632 | 0.598 | 0.045 | 0.082 | 0.192 | 0.431 | 0.278 | 1.778 |
| NetWorldModel ⊕ tree ensemble (ours, ensemble) | 3 | 0.493 | 0.054 | [0.152, 0.772] | 0.634 | 0.406 | 0.118 | 0.632 | 0.598 | 0.045 | 0.082 | 0.192 | 0.431 | 0.278 | 1.778 |

## Kill-chain stage forecasting (macro-F1 over stages present, malicious host-slots)

| method | +1 | +2 | +3 | +5 | +7 | +10 |
|---|---|---|---|---|---|---|
| Markov kill-chain prior | 0.156 | 0.156 | 0.156 | 0.157 | 0.157 | 0.157 |
| NetWorldModel rollout (ours) | 0.156 | 0.156 | 0.156 | 0.157 | 0.157 | 0.157 |
| Persistence (inferred current stage) | 0.156 | 0.156 | 0.156 | 0.157 | 0.157 | 0.157 |
| Persistence (oracle current stage) | 0.986 | 0.979 | 0.969 | 0.951 | 0.930 | 0.900 |
| Stage LR per horizon (stacked) | 0.522 | 0.506 | 0.496 | 0.505 | 0.470 | 0.434 |

Accuracy on cells whose stage *changes* by t+k (persistence scores 0 here by construction):

| method | +1 | +2 | +3 | +5 | +7 | +10 |
|---|---|---|---|---|---|---|
| Markov kill-chain prior | 0.522 | 0.533 | 0.535 | 0.538 | 0.538 | 0.540 |
| NetWorldModel rollout (ours) | 0.522 | 0.533 | 0.535 | 0.538 | 0.538 | 0.540 |
| Persistence (inferred current stage) | 0.522 | 0.533 | 0.535 | 0.538 | 0.538 | 0.540 |
| Persistence (oracle current stage) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Stage LR per horizon (stacked) | 0.609 | 0.400 | 0.395 | 0.338 | 0.363 | 0.323 |

## Host-level triage (two channels)

- infected host-slots caught: **5/8** (risk channel 5, anomaly channel only 0)
- false alarms: 0/3 benign host-slots (0.0%); risk channel alone 0

## Label-free surprise channel

- direction chosen on validation: sign +1 (validation raw AUC 0.565)
- test window-level AUC (malicious vs benign cells): 0.628 (KL variant 0.624)
- test host-level AUC: surprise 0.500 vs supervised risk 0.833

## Integrity checks

- **split integrity (temporal)** - OK: no overlap  `{'issues': []}`
- **schedule-only probe (hour, weekday, elapsed time)** - CAUTION: timing carries part of the signal  `{'schedule_ap': 0.4334, 'prevalence': 0.3253, 'model_ap': 0.5978, 'schedule_share_of_lift': 0.3968}`
- **capture-identity probe** - reported for context: which capture you are in carries this much signal  `{'identity_ap': 0.448, 'prevalence': 0.3253}`
- **label-permutation test of test PR-AUC** - OK: p < 0.00498 (signal above label-permuted chance)  `{'observed_ap': 0.6499, 'null_mean': 0.3278, 'null_p99': 0.3595, 'p_value': 0.005}`
- **no-peeking gradient test** - OK: imagination never touches observations (and the check can fail)  `{'encoder_grad_detached': 0.0, 'encoder_grad_wired': 23.0868}`
- **out-of-distribution benign (flash crowd x5; novel values in ['fin_ratio', 'urg_ratio', 'syn_ratio'])** - OK: risk channel stays quiet on unusual benign traffic  `{'cells': 324, 'alert_rate_original': 0.0123, 'alert_rate_flash_crowd': 0.0031, 'alert_rate_novel_values': 0.0, 'anomaly_rate_flash_crowd': 0.0123, 'anomaly_rate_novel_values': 0.0185}`
