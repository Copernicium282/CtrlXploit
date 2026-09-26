# Evaluation · ctu13 · family protocol

Test cells: 137,887. Target: exploitation-stage cell within the next 10 windows. Thresholds frozen on validation (F1-optimal s.t. FPR ≤ 0.05). Deep models: mean over seeds (± = std); 95 % CI = bootstrap over host-slots/segments for the primary seed.

## Binary forecasting

| Method | Seeds | F1 | F1 ± | F1 95% CI | Precision | Recall | FPR | ROC-AUC | PR-AUC | PR-AUC ± | EW PR-AUC | Pre-attack recall | During-attack recall | Incidents warned | Mean lead (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Persistence (oracle current label) | 1 | 0.732 | 0.000 | [0.217, 0.990] | 0.584 | 0.978 | 0.006 | 0.986 | 0.572 | 0.000 | 0.112 | 0.832 | 1.000 | 0.857 | 4.857 |
| Logistic Regression (single window) | 1 | 0.029 | 0.000 | [0.000, 0.072] | 0.016 | 0.111 | 0.061 | 0.121 | 0.054 | 0.000 | 0.028 | 0.317 | 0.084 | 0.714 | 4.714 |
| Logistic Regression (stacked 8) | 1 | 0.032 | 0.000 | [0.000, 0.087] | 0.019 | 0.106 | 0.050 | 0.139 | 0.072 | 0.000 | 0.058 | 0.292 | 0.080 | 0.571 | 4.286 |
| Gradient Boosting (stacked 8) | 1 | 0.153 | 0.000 | [0.010, 0.246] | 0.101 | 0.320 | 0.026 | 0.826 | 0.125 | 0.000 | 0.072 | 0.590 | 0.284 | 1.000 | 5.000 |
| Random Forest (stacked 8) | 1 | 0.159 | 0.000 | [0.009, 0.278] | 0.101 | 0.379 | 0.031 | 0.814 | 0.161 | 0.000 | 0.158 | 0.764 | 0.325 | 1.000 | 7.143 |
| LSTM classifier (no world model) | 3 | 0.091 | 0.026 | [0.000, 0.136] | 0.085 | 0.113 | 0.016 | 0.223 | 0.078 | 0.004 | 0.048 | 0.325 | 0.084 | 0.667 | 5.476 |
| NetWorldModel - direct head | 3 | 0.089 | 0.020 | [0.001, 0.138] | 0.076 | 0.124 | 0.017 | 0.446 | 0.085 | 0.010 | 0.067 | 0.317 | 0.098 | 0.619 | 5.143 |
| NetWorldModel - K-step rollout | 3 | 0.079 | 0.029 | [0.002, 0.084] | 0.070 | 0.130 | 0.032 | 0.274 | 0.057 | 0.016 | 0.050 | 0.284 | 0.109 | 0.571 | 4.667 |
| Markov kill-chain prior | 3 | 0.096 | 0.030 | [0.003, 0.119] | 0.093 | 0.149 | 0.026 | 0.521 | 0.076 | 0.015 | 0.051 | 0.317 | 0.126 | 0.667 | 5.095 |
| NetWorldModel - fused (ours) | 3 | 0.080 | 0.030 | [0.002, 0.084] | 0.072 | 0.131 | 0.032 | 0.277 | 0.062 | 0.017 | 0.048 | 0.288 | 0.110 | 0.571 | 4.667 |
| NetWorldModel ⊕ tree ensemble (ours, ensemble) | 3 | 0.078 | 0.029 | [0.002, 0.084] | 0.068 | 0.131 | 0.032 | 0.327 | 0.068 | 0.011 | 0.059 | 0.290 | 0.110 | 0.571 | 4.667 |

## Kill-chain stage forecasting (macro-F1 over stages present, malicious host-slots)

| method | +1 | +2 | +3 | +5 | +7 | +10 |
|---|---|---|---|---|---|---|
| Markov kill-chain prior | 0.216 | 0.208 | 0.202 | 0.164 | 0.065 | 0.059 |
| NetWorldModel rollout (ours) | 0.208 | 0.207 | 0.203 | 0.199 | 0.190 | 0.186 |
| Persistence (inferred current stage) | 0.210 | 0.205 | 0.204 | 0.193 | 0.190 | 0.192 |
| Persistence (oracle current stage) | 0.725 | 0.620 | 0.554 | 0.510 | 0.512 | 0.500 |
| Stage LR per horizon (stacked) | 0.207 | 0.210 | 0.213 | 0.195 | 0.180 | 0.166 |

Accuracy on cells whose stage *changes* by t+k (persistence scores 0 here by construction):

| method | +1 | +2 | +3 | +5 | +7 | +10 |
|---|---|---|---|---|---|---|
| Markov kill-chain prior | 0.175 | 0.137 | 0.115 | 0.094 | 0.105 | 0.103 |
| NetWorldModel rollout (ours) | 0.160 | 0.137 | 0.115 | 0.089 | 0.087 | 0.084 |
| Persistence (inferred current stage) | 0.165 | 0.130 | 0.112 | 0.084 | 0.087 | 0.081 |
| Persistence (oracle current stage) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Stage LR per horizon (stacked) | 0.145 | 0.130 | 0.148 | 0.053 | 0.066 | 0.062 |

## Host-level triage (two channels)

- infected host-slots caught: **4/5** (risk channel 4, anomaly channel only 0)
- false alarms: 294/1058 benign host-slots (27.8%); risk channel alone 285

## Label-free surprise channel

- direction chosen on validation: sign +1 (validation raw AUC 0.579)
- test window-level AUC (malicious vs benign cells): 0.795 (KL variant 0.578)
- test host-level AUC: surprise 0.853 vs supervised risk 0.882

## Integrity checks

- **split integrity (family)** - OK: no overlap  `{'issues': []}`
- **schedule-only probe (hour, weekday, elapsed time)** - OK: timing alone is far weaker than the traffic model  `{'schedule_ap': 0.0103, 'prevalence': 0.0091, 'model_ap': 0.0618, 'schedule_share_of_lift': 0.0227}`
- **capture-identity probe** - N/A: test captures unseen in training (family protocol)  `{}`
- **label-permutation test of test PR-AUC** - OK: p < 0.00498 (signal above label-permuted chance)  `{'observed_ap': 0.0391, 'null_mean': 0.0091, 'null_p99': 0.0099, 'p_value': 0.005}`
- **no-peeking gradient test** - OK: imagination never touches observations (and the check can fail)  `{'encoder_grad_detached': 0.0, 'encoder_grad_wired': 3349.3083}`
- **out-of-distribution benign (flash crowd x5; novel values in ['fin_ratio', 'psh_ratio', 'ack_ratio'])** - OK: risk channel stays quiet on unusual benign traffic  `{'cells': 7090, 'alert_rate_original': 0.0147, 'alert_rate_flash_crowd': 0.0118, 'alert_rate_novel_values': 0.0, 'anomaly_rate_flash_crowd': 0.0934, 'anomaly_rate_novel_values': 0.0371}`
