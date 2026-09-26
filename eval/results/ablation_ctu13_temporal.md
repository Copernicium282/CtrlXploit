# Ablations · ctu13 · temporal

Evaluated on every malicious host-slot plus 150 sampled benign host-slots (8,590 test cells); thresholds re-calibrated on the same kind of validation subset.

## Fusion components

| variant | F1 | FPR | PR-AUC | EW PR-AUC | Mean lead (min) |
|---|---|---|---|---|---|
| fusion: direct | 0.651 | 0.033 | 0.857 | 0.393 | 1.500 |
| fusion: rollout | 0.613 | 0.041 | 0.866 | 0.406 | 1.500 |
| fusion: markov | 0.617 | 0.039 | 0.865 | 0.395 | 1.500 |
| fusion: direct+rollout | 0.617 | 0.040 | 0.867 | 0.403 | 1.500 |
| fusion: direct+markov | 0.643 | 0.034 | 0.867 | 0.398 | 1.500 |
| fusion: all (0.5/0.35/0.15) | 0.620 | 0.039 | 0.867 | 0.402 | 1.500 |

## Imagination branch: particles x rollout horizon

| variant | particles | K | sec | F1 | FPR | PR-AUC | EW PR-AUC | Mean lead (min) |
|---|---|---|---|---|---|---|---|---|
| rollout | 1 | 5 | 1.500 | 0.583 | 0.051 | 0.867 | 0.412 | 1.750 |
| rollout | 1 | 10 | 1.600 | 0.635 | 0.037 | 0.861 | 0.364 | 1.500 |
| rollout | 1 | 15 | 1.800 | 0.604 | 0.044 | 0.864 | 0.382 | 1.500 |
| rollout | 8 | 5 | 2.300 | 0.610 | 0.042 | 0.864 | 0.407 | 1.500 |
| rollout | 8 | 10 | 3.300 | 0.613 | 0.041 | 0.866 | 0.406 | 1.500 |
| rollout | 8 | 15 | 4.200 | 0.616 | 0.041 | 0.867 | 0.407 | 1.500 |
| rollout | 32 | 5 | 5.200 | 0.612 | 0.042 | 0.867 | 0.405 | 1.500 |
| rollout | 32 | 10 | 9.300 | 0.610 | 0.042 | 0.867 | 0.404 | 1.500 |
| rollout | 32 | 15 | 13.300 | 0.607 | 0.042 | 0.867 | 0.405 | 1.500 |
