# Evaluation · synthetic · temporal protocol

Test cells: 3,200. Target: exploitation-stage cell within the next 10 windows. Thresholds frozen on validation (F1-optimal s.t. FPR ≤ 0.05). Deep models: mean over seeds (± = std); 95 % CI = bootstrap over host-slots/segments for the primary seed.

## Binary forecasting

| Method | Seeds | F1 | F1 ± | F1 95% CI | Precision | Recall | FPR | ROC-AUC | PR-AUC | PR-AUC ± | EW PR-AUC | Pre-attack recall | During-attack recall | Incidents warned | Mean lead (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Persistence (oracle current label) | 1 | 0.860 | 0.000 | [0.833, 0.890] | 0.898 | 0.825 | 0.029 | 0.898 | 0.782 | 0.000 | 0.357 | 0.444 | 1.000 | 0.208 | 1.333 |
| Logistic Regression (single window) | 1 | 0.879 | 0.000 | [0.842, 0.917] | 0.945 | 0.822 | 0.015 | 0.920 | 0.903 | 0.000 | 0.578 | 0.448 | 0.991 | 0.583 | 4.458 |
| Logistic Regression (stacked 8) | 1 | 0.871 | 0.000 | [0.827, 0.911] | 0.950 | 0.804 | 0.013 | 0.918 | 0.899 | 0.000 | 0.565 | 0.414 | 0.980 | 0.500 | 4.125 |
| Gradient Boosting (stacked 8) | 1 | 0.879 | 0.000 | [0.844, 0.916] | 0.927 | 0.835 | 0.020 | 0.916 | 0.900 | 0.000 | 0.568 | 0.477 | 0.998 | 0.542 | 4.667 |
| Random Forest (stacked 8) | 1 | 0.877 | 0.000 | [0.839, 0.916] | 0.949 | 0.814 | 0.014 | 0.923 | 0.902 | 0.000 | 0.571 | 0.431 | 0.987 | 0.500 | 4.292 |
| LSTM classifier (no world model) | 3 | 0.871 | 0.001 | [0.834, 0.911] | 0.938 | 0.813 | 0.017 | 0.931 | 0.911 | 0.003 | 0.602 | 0.464 | 0.971 | 0.556 | 4.625 |
| NetWorldModel - direct head | 3 | 0.877 | 0.002 | [0.840, 0.919] | 0.942 | 0.821 | 0.016 | 0.922 | 0.906 | 0.001 | 0.585 | 0.460 | 0.982 | 0.500 | 4.583 |
| NetWorldModel - K-step rollout | 3 | 0.838 | 0.005 | [0.800, 0.883] | 0.864 | 0.814 | 0.040 | 0.919 | 0.894 | 0.001 | 0.508 | 0.462 | 0.973 | 0.500 | 4.569 |
| Markov kill-chain prior | 3 | 0.846 | 0.008 | [0.811, 0.890] | 0.874 | 0.820 | 0.037 | 0.918 | 0.884 | 0.004 | 0.350 | 0.455 | 0.985 | 0.528 | 4.486 |
| NetWorldModel - fused (ours) | 3 | 0.877 | 0.002 | [0.840, 0.919] | 0.942 | 0.821 | 0.016 | 0.922 | 0.906 | 0.001 | 0.585 | 0.460 | 0.982 | 0.500 | 4.583 |
| NetWorldModel ⊕ tree ensemble (ours, ensemble) | 3 | 0.877 | 0.001 | [0.841, 0.916] | 0.948 | 0.816 | 0.014 | 0.924 | 0.903 | 0.001 | 0.572 | 0.437 | 0.988 | 0.500 | 4.347 |

## Kill-chain stage forecasting (macro-F1 over stages present, malicious host-slots)

| method | +1 | +2 | +3 | +5 | +7 | +10 |
|---|---|---|---|---|---|---|
| Markov kill-chain prior | 0.501 | 0.462 | 0.399 | 0.329 | 0.239 | 0.148 |
| NetWorldModel rollout (ours) | 0.569 | 0.550 | 0.535 | 0.516 | 0.476 | 0.423 |
| Persistence (inferred current stage) | 0.575 | 0.522 | 0.468 | 0.368 | 0.273 | 0.194 |
| Persistence (oracle current stage) | 0.844 | 0.702 | 0.575 | 0.403 | 0.298 | 0.206 |
| Stage LR per horizon (stacked) | 0.763 | 0.623 | 0.565 | 0.523 | 0.471 | 0.407 |

Accuracy on cells whose stage *changes* by t+k (persistence scores 0 here by construction):

| method | +1 | +2 | +3 | +5 | +7 | +10 |
|---|---|---|---|---|---|---|
| Markov kill-chain prior | 0.271 | 0.234 | 0.195 | 0.222 | 0.318 | 0.290 |
| NetWorldModel rollout (ours) | 0.316 | 0.336 | 0.358 | 0.404 | 0.408 | 0.434 |
| Persistence (inferred current stage) | 0.226 | 0.182 | 0.156 | 0.115 | 0.078 | 0.050 |
| Persistence (oracle current stage) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Stage LR per horizon (stacked) | 0.258 | 0.343 | 0.402 | 0.462 | 0.466 | 0.441 |

## Host-level triage (two channels)

- infected host-slots caught: **15/15** (risk channel 15, anomaly channel only 0)
- false alarms: 0/1 benign host-slots (0.0%); risk channel alone 0

## Label-free surprise channel

- direction chosen on validation: sign +1 (validation raw AUC 0.840)
- test window-level AUC (malicious vs benign cells): 0.835 (KL variant 0.807)
- test host-level AUC: surprise 1.000 vs supervised risk 1.000

## Integrity checks

- **split integrity (temporal)** - OK: no overlap  `{'issues': []}`
- **schedule-only probe (hour, weekday, elapsed time)** - CAUTION: timing carries part of the signal  `{'schedule_ap': 0.51, 'prevalence': 0.2372, 'model_ap': 0.9063, 'schedule_share_of_lift': 0.4077}`
- **capture-identity probe** - reported for context: which capture you are in carries this much signal  `{'identity_ap': 0.2383, 'prevalence': 0.2372}`
- **label-permutation test of test PR-AUC** - OK: p < 0.00498 (signal above label-permuted chance)  `{'observed_ap': 0.9075, 'null_mean': 0.2385, 'null_p99': 0.253, 'p_value': 0.005}`
- **no-peeking gradient test** - OK: imagination never touches observations (and the check can fail)  `{'encoder_grad_detached': 0.0, 'encoder_grad_wired': 10826.7998}`
- **out-of-distribution benign (flash crowd x5; novel values in ['rst_ratio', 'retrans_rate', 'psh_ratio'])** - OK: risk channel stays quiet on unusual benign traffic  `{'cells': 200, 'alert_rate_original': 0.0, 'alert_rate_flash_crowd': 0.0, 'alert_rate_novel_values': 0.0, 'anomaly_rate_flash_crowd': 0.085, 'anomaly_rate_novel_values': 0.075}`
