# ADHD EEG region fusion

This project studies ADHD classification from the 19-channel EEG dataset used in
Chauhan and Choi (2023). The first implemented milestone is a **global baseline**:
all 19 channels are used together before individual regions or region fusion are
compared.

## Current status

The global baseline is implemented and verified on the complete local dataset:

| Model | Balanced accuracy | Sensitivity | Specificity | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 60.3% | 63.9% | 56.7% | 65.4% |
| Linear SVM | 60.3% | 65.6% | 55.0% | 64.6% |
| Gaussian Naive Bayes | 64.3% | 83.6% | 45.0% | 66.9% |

These are out-of-fold results for 121 subjects and 2057 windows. They are a
research baseline, not a clinical diagnostic result.

## Dataset contract

The input is one CSV row per time sample. It must contain `Class`, `ID`, and the
19 channels `Fp1`, `Fp2`, `F3`, `F4`, `C3`, `C4`, `P3`, `P4`, `O1`, `O2`,
`F7`, `F8`, `T7`, `T8`, `P7`, `P8`, `Fz`, `Cz`, and `Pz`. `Class` must be
either `ADHD` or `Control`, and every `ID` must occupy one contiguous block.

The loader reads the large CSV in chunks, validates labels and numeric values,
and yields one subject at a time instead of retaining the entire file in RAM.

## Method

1. Load recordings and keep every subject separate.
2. Split each recording into non-overlapping 8-second windows (1024 samples at
   128 Hz); incomplete trailing samples are discarded.
3. Compute log band power for delta, theta, alpha, beta, and lower-gamma on each
   channel (95 features per window).
4. Evaluate logistic regression, linear SVM, and Gaussian Naive Bayes with
   stratified, subject-grouped cross-validation.
5. Average window probabilities per subject and report subject-level balanced
   accuracy, sensitivity, specificity, F1, and ROC-AUC.

For the linear SVM, its decision score is passed through a logistic transform so
that it has the same 0–1 representation and 0.5 decision threshold. Treat this
as a confidence score rather than a clinically calibrated probability.

The train/test boundary is always at subject level. A child's windows can never
appear in both sets. Feature scaling is fitted inside each training fold.

The distributed CSV has already undergone artifact removal, ICA and filtering.
The pipeline therefore does not apply a second signal filter.

## Run

Create an environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the global experiment:

```bash
adhd-eeg-global --data "EEG Dataset for ADHD/adhdata.csv"
```

During development, a smaller smoke run is available:

```bash
adhd-eeg-global --data "EEG Dataset for ADHD/adhdata.csv" \
  --max-subjects-per-class 5 --folds 2 --models logistic
```

Generated artifacts go to `outputs/global/`:

- `features.npz`: cached window features and subject metadata
- `metrics.json`: fold and overall subject-level metrics
- `subject_predictions.csv`: out-of-fold predictions
- one confusion matrix and ROC curve per model

Delete the feature cache or pass `--rebuild-cache` after changing feature or
segmentation settings.

## Tests

```bash
pytest
```

## Next research steps

Keep the evaluation procedure fixed and change only the channel selection:
first compare the five individual regions, then combinations of three regions,
and finally train a leakage-safe late-fusion model from regional predictions.
