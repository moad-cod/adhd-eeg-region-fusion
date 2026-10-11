# Three-region experiment

This experiment asks whether a subset of three anatomical EEG regions can
outperform the global five-region representation. It evaluates all ten unique
sets of three regions from frontal (F), central (C), parietal (P), temporal
(T), and occipital (O).

## Fair comparison controls

- The regional experiment reuses the global 95-feature cache.
- It changes only the selected channel-band columns.
- The sampling window, labels, models, folds, and random seed remain fixed.
- `StratifiedGroupKFold` keeps every child's windows in one fold.
- The command rejects global results produced with different folds or seeds.

Feature counts vary because regions contain different numbers of channels. For
example, F-C-P contains 13 channels and 65 bandpower features, while C-P-O
contains 8 channels and 40 features. Every output records both counts.

## Run

```bash
make regions \
  MODEL="logistic linear_svm naive_bayes rbf_svm random_forest extra_trees xgboost knn mlp_neural_net"
```

Use `make regions-smoke` only to validate the pipeline. Smoke-test scores use a
small subset of subjects and must not be reported as research results.

The full benchmark trains 450 fold-specific estimators (10 combinations × 9
models × 5 folds), so it can take several minutes. Progress is checkpointed
after every model. If the process is interrupted, run the same command again to
resume. Pass `EXTRA_ARGS=--restart` only when all checkpoints should be ignored
and recomputed.

## Interpretation

The comparison reports `delta_roc_auc = three-region AUC - global AUC` and the
equivalent balanced-accuracy difference. A positive observed delta means the
subset scored higher in these folds; it does not alone establish a statistically
reliable improvement. Selecting the best of ten combinations introduces
selection optimism. A final confirmatory analysis should use nested
cross-validation or paired subject-level bootstrap confidence intervals.
