# ADHD EEG region fusion

This project studies ADHD classification from the 19-channel EEG dataset used in
Chauhan and Choi (2023). The first implemented milestone is a **global baseline**:
all 19 channels are used together before individual regions or region fusion are
compared.

The study is organized around three questions: the global five-region baseline,
the paper's sets of three regions, and the contribution of each individual
region. See the [experimental roadmap](docs/experiment.md) for the complete
design and interpretation plan.

## Experiment status

| Research question | Representation | Status | Main visual evidence |
| --- | --- | --- | --- |
| 1. Global baseline | Five regions, 19 channels, 95 features | Complete | Nine-model ROC comparison and per-model diagnostics |
| 2. Three-region subsets | Ten unique three-region combinations | Complete | Combination heatmap and global-versus-best comparison |
| 3. Individual regions | Five separate anatomical regions | Not yet implemented | No result figures yet |

The figures shown below are generated artifacts from the corresponding
experiments, not manually entered illustrations. The complete experimental
logic is documented in the [experimental roadmap](docs/experiment.md).

## Question 1 — Global five-region benchmark

The global experiment is implemented for nine classifiers. Results below are
out-of-fold, subject-level estimates from 121 children and 2,057 EEG windows.
Models are ranked by ROC-AUC.

| Rank | Model | Balanced accuracy | Sensitivity | Specificity | F1 | ROC-AUC |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Random Forest | 66.0% | 78.7% | 53.3% | 70.1% | **73.7%** |
| 2 | RBF SVM | **67.7%** | 80.3% | 55.0% | 71.5% | 73.3% |
| 3 | Extra Trees | 65.1% | 86.9% | 43.3% | **71.6%** | 73.2% |
| 4 | XGBoost | 65.2% | 82.0% | 48.3% | 70.4% | 71.9% |
| 5 | MLP Neural Net | 66.1% | 73.8% | **58.3%** | 68.7% | 71.2% |
| 6 | KNN | 61.8% | **88.5%** | 35.0% | 70.1% | 69.4% |
| 7 | Gaussian Naive Bayes | 64.3% | 83.6% | 45.0% | 70.3% | 66.9% |
| 8 | Logistic Regression | 60.3% | 63.9% | 56.7% | 61.9% | 65.4% |
| 9 | Linear SVM | 60.3% | 65.6% | 55.0% | 62.5% | 64.6% |

Random Forest provides the strongest ranking performance, while RBF SVM has the
highest balanced accuracy. KNN is most sensitive but has low specificity,
illustrating why no single metric should determine model selection. These are
research results, not clinical diagnostic performance claims.

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
4. Evaluate selected classical or neural-network models with stratified,
   subject-grouped cross-validation.
5. Average window probabilities per subject and report subject-level balanced
   accuracy, sensitivity, specificity, F1, and ROC-AUC.

For the linear SVM, its decision score is passed through a logistic transform so
that it has the same 0–1 representation and 0.5 decision threshold. Treat this
as a confidence score rather than a clinically calibrated probability.

The train/test boundary is always at subject level. A child's windows can never
appear in both sets. Feature scaling is fitted inside each training fold.

The distributed CSV has already undergone artifact removal, ICA and filtering.
The pipeline therefore does not apply a second signal filter.

## Project structure

```text
src/adhd_eeg/
├── config.py       # sampling, channels, regions, and EEG bands
├── data.py         # validation, subject loading, and segmentation
├── features.py     # FFT-based log-bandpower extraction
├── evaluation.py   # subject aggregation, metrics, and plots
├── models/         # one model factory per classification algorithm
└── experiment.py   # cross-validation and command-line workflow
tests/              # focused unit tests for the core transformations
outputs/global/     # generated cache, predictions, metrics, and figures
outputs/all_models/ # structured nine-model benchmark and comparison
outputs/three_region_benchmark/ # regional results and comparison figures
```

The notebook remains exploratory. The package is the reproducible source of
truth for experiments.

## Feature representation

Every window starts as `1024 samples × 19 channels`. A Hann-windowed Fourier
transform estimates power in delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz),
beta (13–30 Hz), and lower-gamma (30–45 Hz). Integrating each band and applying
`log10` produces `5 bands × 19 channels = 95` features.

Feature names use `<band>_<channel>`, for example `theta_Fp1`. Keeping this
representation explicit makes later regional selection a channel-filtering
operation rather than a new preprocessing pipeline.

## Scientific results and visual diagnostics

### Exploratory signal characterization

![Representative multichannel EEG traces for control and ADHD recordings](outputs/Signals-Vis.png)

**Figure 1. Representative 3-second multichannel EEG traces.** The stacked
traces provide a qualitative check of channel continuity, amplitude scale, and
temporal structure. A representative trace is not a population estimate and
must not be interpreted as proof of a diagnostic difference.

![Regional EEG power spectral density](outputs/PSD-Vis.png)

**Figure 2. Regional power spectral density illustration.** Spectra are grouped
into frontal, temporal, central, parietal, and occipital regions, with theta
(4–8 Hz) and beta (13–30 Hz) highlighted. The visible low-frequency contrast
motivates frequency-band features; statistical conclusions require the
subject-level cross-validation reported below.

### Nine-model ROC benchmark

![Publication-ready comparison of all subject-level ROC curves](outputs/all_models/comparison/all_models_subject_roc.png)

**Figure 3. Subject-level ROC comparison.** Curves are ordered by ROC-AUC; the
four strongest models are emphasized and the inset expands the clinically
important low-FPR region. Random Forest ranks first (AUC 0.737), closely
followed by RBF SVM (0.733) and Extra Trees (0.732). The modest separation and
overlapping curves argue for cautious interpretation and external validation.

### Per-model diagnostics

Confusion matrices show threshold-dependent errors at a 0.5 subject-level
decision threshold. ROC curves show discrimination across all thresholds.
Feature charts report absolute standardized coefficients, native tree
importance, or mean absolute first-layer MLP weights; their magnitudes are
interpretable within a model, not directly comparable between model families.

<details open>
<summary><strong>Random Forest — strongest ROC-AUC (0.737)</strong></summary>

| Subject-level confusion matrix | Subject-level ROC |
| --- | --- |
| ![Random Forest confusion matrix](outputs/all_models/models/random_forest/random_forest_confusion_matrix.png) | ![Random Forest ROC](outputs/all_models/models/random_forest/random_forest_roc.png) |

![Random Forest Top-15 EEG features](outputs/all_models/models/random_forest/random_forest_top_features.png)

The model emphasizes a distributed spectral signature, led by lower-gamma C3,
alpha Cz, beta F4, and additional central/parietal features. These are
predictive associations, not causal biomarkers.
</details>

<details>
<summary><strong>RBF SVM — highest balanced accuracy (0.677)</strong></summary>

| Subject-level confusion matrix | Subject-level ROC |
| --- | --- |
| ![RBF SVM confusion matrix](outputs/all_models/models/rbf_svm/rbf_svm_confusion_matrix.png) | ![RBF SVM ROC](outputs/all_models/models/rbf_svm/rbf_svm_roc.png) |

The nonlinear kernel has no native per-feature importance, so interpretation is
limited to its subject-level discrimination and error profile.
</details>

<details>
<summary><strong>Extra Trees — ROC-AUC 0.732</strong></summary>

| Subject-level confusion matrix | Subject-level ROC |
| --- | --- |
| ![Extra Trees confusion matrix](outputs/all_models/models/extra_trees/extra_trees_confusion_matrix.png) | ![Extra Trees ROC](outputs/all_models/models/extra_trees/extra_trees_roc.png) |

![Extra Trees Top-15 EEG features](outputs/all_models/models/extra_trees/extra_trees_top_features.png)

The randomized tree ensemble offers a complementary nonlinear importance
profile. Its high sensitivity is accompanied by lower specificity.
</details>

<details>
<summary><strong>XGBoost — ROC-AUC 0.719</strong></summary>

| Subject-level confusion matrix | Subject-level ROC |
| --- | --- |
| ![XGBoost confusion matrix](outputs/all_models/models/xgboost/xgboost_confusion_matrix.png) | ![XGBoost ROC](outputs/all_models/models/xgboost/xgboost_roc.png) |

![XGBoost Top-15 EEG features](outputs/all_models/models/xgboost/xgboost_top_features.png)

Boosted-tree importance identifies the features most useful for sequential
error correction; importance does not encode direction of association.
</details>

<details>
<summary><strong>MLP Neural Net — ROC-AUC 0.712</strong></summary>

| Subject-level confusion matrix | Subject-level ROC |
| --- | --- |
| ![MLP confusion matrix](outputs/all_models/models/mlp_neural_net/mlp_neural_net_confusion_matrix.png) | ![MLP ROC](outputs/all_models/models/mlp_neural_net/mlp_neural_net_roc.png) |

![MLP Top-15 input features](outputs/all_models/models/mlp_neural_net/mlp_neural_net_top_features.png)

The feature chart summarizes mean absolute input-to-hidden-layer weights. It is
a diagnostic of model reliance, not a complete neural-network explanation.
</details>

<details>
<summary><strong>KNN — highest sensitivity (0.885), ROC-AUC 0.694</strong></summary>

| Subject-level confusion matrix | Subject-level ROC |
| --- | --- |
| ![KNN confusion matrix](outputs/all_models/models/knn/knn_confusion_matrix.png) | ![KNN ROC](outputs/all_models/models/knn/knn_roc.png) |

KNN detects many ADHD subjects but produces more false positives, reflected by
35.0% specificity. Distance-based KNN has no native feature-importance vector.
</details>

<details>
<summary><strong>Gaussian Naive Bayes — ROC-AUC 0.669</strong></summary>

| Subject-level confusion matrix | Subject-level ROC |
| --- | --- |
| ![Naive Bayes confusion matrix](outputs/all_models/models/naive_bayes/naive_bayes_confusion_matrix.png) | ![Naive Bayes ROC](outputs/all_models/models/naive_bayes/naive_bayes_roc.png) |

The model is sensitive (83.6%) but less specific (45.0%), consistent with
over-prediction of ADHD at the fixed threshold.
</details>

<details>
<summary><strong>Logistic Regression — interpretable linear baseline</strong></summary>

| Subject-level confusion matrix | Subject-level ROC |
| --- | --- |
| ![Logistic Regression confusion matrix](outputs/all_models/models/logistic/logistic_confusion_matrix.png) | ![Logistic Regression ROC](outputs/all_models/models/logistic/logistic_roc.png) |

![Logistic Regression Top-15 coefficients](outputs/all_models/models/logistic/logistic_top_features.png)

Absolute standardized coefficients identify the features contributing most to
the linear decision function. Absolute values show magnitude, not direction.
</details>

<details>
<summary><strong>Linear SVM — linear margin baseline</strong></summary>

| Subject-level confusion matrix | Subject-level ROC |
| --- | --- |
| ![Linear SVM confusion matrix](outputs/all_models/models/linear_svm/linear_svm_confusion_matrix.png) | ![Linear SVM ROC](outputs/all_models/models/linear_svm/linear_svm_roc.png) |

![Linear SVM Top-15 coefficients](outputs/all_models/models/linear_svm/linear_svm_top_features.png)

The standardized linear-margin coefficients provide a sparse diagnostic of the
features defining separation, although overall discrimination is modest.
</details>

### Original three-model baseline figures

These retained figures document the initial global-baseline milestone before
the experiment was expanded to nine classifiers.

<details>
<summary><strong>Open original baseline gallery</strong></summary>

| Model | Confusion matrix | ROC curve |
| --- | --- | --- |
| Logistic Regression | ![Original logistic confusion matrix](outputs/global/logistic_confusion_matrix.png) | ![Original logistic ROC](outputs/global/logistic_roc.png) |
| Linear SVM | ![Original linear SVM confusion matrix](outputs/global/linear_svm_confusion_matrix.png) | ![Original linear SVM ROC](outputs/global/linear_svm_roc.png) |
| Gaussian Naive Bayes | ![Original Naive Bayes confusion matrix](outputs/global/naive_bayes_confusion_matrix.png) | ![Original Naive Bayes ROC](outputs/global/naive_bayes_roc.png) |
</details>

## Question 2 — Three-region experiment

The regional benchmark evaluates all ten unique selections of three regions
with all nine classifiers, producing 90 subject-level results. It reuses the
same folds, random seed, feature extraction, and subject aggregation as the
global experiment so that channel selection is the principal changed factor.

![ROC-AUC heatmap for every model and three-region combination](outputs/three_region_benchmark/comparison/region_combination_heatmap.png)

**Figure 4. Subject-level ROC-AUC across the three-region search space.** Each
cell represents one classifier and one regional subset. The heatmap should be
used to assess whether a result is consistent across several models, rather
than selecting a region solely from one unusually high cell. The highest
observed ROC-AUC is 0.778 for RBF SVM with Central-Parietal-Occipital (C-P-O).

![Global five-region models compared with their best three-region subsets](outputs/three_region_benchmark/comparison/global_vs_best_three_regions.png)

**Figure 5. Best three-region result versus the global baseline for each
model.** All nine classifiers obtain a higher observed ROC-AUC from
their best three-region subset. The largest increase is for Linear SVM with
C-P-O (0.646 to 0.729, ΔAUC = +0.083). RBF SVM with C-P-O provides the
strongest absolute regional result (AUC 0.778 versus 0.733 globally). Because
the best subset is selected from ten candidates on the same cross-validation
results, these differences are exploratory and require confidence intervals or
nested validation before being described as generalizable improvements.

The machine-readable regional evidence is available in
[`three_region_summary.csv`](outputs/three_region_benchmark/comparison/three_region_summary.csv),
[`three_region_summary.json`](outputs/three_region_benchmark/comparison/three_region_summary.json),
and the
[`global_vs_best_three_regions.csv`](outputs/three_region_benchmark/comparison/global_vs_best_three_regions.csv)
comparison. Each folder under
[`outputs/three_region_benchmark/combinations/`](outputs/three_region_benchmark/combinations/)
contains the combined ROC plot plus the confusion matrix, ROC curve, and
available feature-importance figure for every model in that combination.

## Question 3 — Individual-region experiment

The single-region benchmark has not yet been implemented or run. It will
evaluate Frontal, Central, Parietal, Temporal, and Occipital features
independently under the same validation design. No one-region image is shown in
this README until that experiment produces real results.

## Run

Create an environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Short Make commands

Use the root `Makefile` to avoid repeating the dataset path and CLI options:

```bash
make run MODEL=random_forest
make run MODEL=random_forest RUN_NAME=rf_final
make smoke MODEL=rbf_svm
```

The run name becomes the output directory, such as `outputs/rf_final/`. Compare
multiple models in one run with:

```bash
make run MODEL="logistic random_forest extra_trees" RUN_NAME=tree_comparison
```

Folds and random seed are configurable:

```bash
make run MODEL=random_forest RUN_NAME=rf_10fold FOLDS=10 SEED=123
```

Run `make help` to see the common forms. Additional CLI flags can be passed via
`EXTRA_ARGS`, for example `EXTRA_ARGS=--rebuild-cache`.

### Three-region experiment

The paper's main regional experiment evaluates all ten sets of three regions
while preserving the same subject folds used by the global baseline:

```bash
make regions \
  MODEL="logistic linear_svm naive_bayes rbf_svm random_forest extra_trees xgboost knn mlp_neural_net"
```

Run a fast implementation check on two combinations:

```bash
make regions-smoke MODEL=logistic COMBINATIONS="F-C-P F-T-O"
```

Regional outputs are written under `outputs/three_region_benchmark/`, including
a model-by-combination ROC-AUC heatmap and global-versus-best-subset CSV, JSON,
and PNG comparisons. The command validates that global and regional folds and
random seeds match before comparing scores. See
[the three-region experiment guide](docs/three-region-experiment.md) for the
combination design and interpretation limits.

The full run checkpoints every completed model. If it is interrupted, execute
the same `make regions ...` command to resume instead of repeating finished
work.

Run the global experiment:

```bash
adhd-eeg-global --data "EEG Dataset for ADHD/adhdata.csv"
```

The default run evaluates `logistic`, `linear_svm`, and `naive_bayes`. The
additional available model names are `rbf_svm`, `random_forest`, `extra_trees`,
`xgboost`, `knn`, and `mlp_neural_net`.

Run all newly added scikit-learn models:

```bash
adhd-eeg-global --data "EEG Dataset for ADHD/adhdata.csv" \
  --models rbf_svm random_forest extra_trees knn mlp_neural_net
```

Run one model by itself, for example the RBF SVM:

```bash
adhd-eeg-global --data "EEG Dataset for ADHD/adhdata.csv" \
  --models rbf_svm --output outputs/rbf_svm
```

XGBoost is optional. Install and run it with:

```bash
pip install -e ".[xgboost]"
adhd-eeg-global --data "EEG Dataset for ADHD/adhdata.csv" \
  --models xgboost --output outputs/xgboost
```

Use a distinct `--output` directory for separate runs; otherwise the newest run
replaces `metrics.json` and `subject_predictions.csv` in the previous directory.

To evaluate every registered model after installing XGBoost:

```bash
adhd-eeg-global --data "EEG Dataset for ADHD/adhdata.csv" \
  --models logistic linear_svm naive_bayes rbf_svm random_forest \
  extra_trees xgboost knn mlp_neural_net
```

During development, a smaller smoke run is available:

```bash
adhd-eeg-global --data "EEG Dataset for ADHD/adhdata.csv" \
  --max-subjects-per-class 5 --folds 2 --models logistic
```

Generated artifacts use a structured run directory such as
`outputs/all_models/`:

- `features.npz`: cached window features and subject metadata
- `metrics.json`: fold and overall subject-level metrics
- `subject_predictions.csv`: out-of-fold predictions
- `comparison/benchmark_summary.csv`: model ranking for spreadsheets
- `comparison/benchmark_summary.json`: the same ranking as structured JSON
- `comparison/all_models_subject_roc.png`: consolidated ROC comparison
- `models/<algorithm>/`: that model's confusion matrix, ROC curve, and optional
  Top-15 feature-importance plot

Example:

```text
outputs/all_models/
├── comparison/
│   ├── all_models_subject_roc.png
│   ├── benchmark_summary.csv
│   └── benchmark_summary.json
├── models/
│   ├── logistic/
│   ├── random_forest/
│   └── ...
├── metrics.json
└── subject_predictions.csv
```

Delete the feature cache or pass `--rebuild-cache` after changing feature or
segmentation settings.

## Reproducibility

The default experiment uses five folds and random seed 42. Record any changed
arguments alongside reported results. To rebuild features and select models
explicitly, run:

```bash
adhd-eeg-global --data "EEG Dataset for ADHD/adhdata.csv" \
  --rebuild-cache --folds 5 --random-state 42 \
  --models logistic linear_svm naive_bayes
```

Scaling is learned only from a training fold, and every window from one subject
stays in the same fold. `metrics.json` stores the effective configuration and
fold-level results.

## Reading the metrics

- **Sensitivity** is the fraction of ADHD subjects correctly detected.
- **Specificity** is the fraction of control subjects correctly rejected.
- **Balanced accuracy** averages sensitivity and specificity so one class does
  not dominate the headline score.
- **ROC-AUC** measures ranking quality over every possible decision threshold.

RBF SVM currently has the highest balanced accuracy, while Random Forest has
the highest ROC-AUC. Several models achieve high sensitivity by sacrificing
specificity, so model selection must consider the confusion matrix and
subject-level predictions rather than one headline number.

## Tests

```bash
pytest
```

## Scope and limitations

- The dataset contains 121 children, so the number of independent observations
  is 121—not the number of generated windows.
- Bandpower is an interpretable baseline but does not represent phase,
  connectivity, or temporal evolution within a window.
- Results come from one dataset and need external validation before they can be
  generalized.
- The models support research and education only. They are not medical devices
  and must not be used to diagnose ADHD.
- The paper's scores are not directly comparable because preprocessing,
  features, and validation details differ.

## Next research steps

Keep the evaluation procedure fixed and change only the channel selection. The
global and three-region benchmarks are complete; the next direct experiment is
the individual-region benchmark.

Suggested milestone order:

1. Implement and report each individual region against the global baseline.
2. Estimate uncertainty for global-versus-regional differences.
3. Use nested validation if selecting and reporting the best regional subset.
4. Build late fusion from out-of-fold regional predictions.
5. Interpret which bands and channels drive the best classical model.

## Research reference

N. Chauhan and B.-J. Choi, “Regional Contribution in
Electrophysiological-Based Classifications of Attention Deficit Hyperactive
Disorder (ADHD) Using Machine Learning,” *Computation*, vol. 11, no. 9, 2023,
doi: [10.3390/computation11090180](https://doi.org/10.3390/computation11090180).
