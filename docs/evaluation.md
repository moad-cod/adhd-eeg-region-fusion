# Evaluation artifacts

All reported predictions are out-of-fold and aggregated to one row per
subject. Window probabilities are averaged; a threshold of 0.5 produces the
subject diagnosis.

The experiment writes:

- `metrics.json` with configuration, fold metrics, and overall metrics;
- `subject_predictions.csv` with every out-of-fold subject prediction;
- `benchmark_summary.csv` ranked by subject-level ROC-AUC;
- individual confusion matrices and ROC curves in PNG and SVG;
- a combined subject-level ROC comparison in PNG and SVG;
- Top-15 window-level EEG feature charts for interpretable models.

Balanced accuracy is the primary threshold metric because it gives sensitivity
and specificity equal weight. ROC-AUC describes ranking across all thresholds.
Neither metric makes these research models suitable for clinical diagnosis.
