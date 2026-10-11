# Experimental roadmap

This project is organized around three related research questions. The
experiments must be completed in order because each stage provides the baseline
or explanation needed by the next stage.

## Shared experimental controls

The following conditions remain fixed in all three experiments:

- 121 subjects: 61 ADHD and 60 control;
- 128 Hz EEG sampling frequency;
- non-overlapping 8-second windows (1024 samples);
- five frequency bands: delta, theta, alpha, beta, and lower gamma;
- subject-grouped five-fold cross-validation;
- random seed 42;
- identical classifier configurations;
- window predictions averaged into one prediction per subject;
- balanced accuracy, sensitivity, specificity, F1-score, and ROC-AUC;
- preprocessing and scaling learned without using test-subject information.

Keeping these conditions fixed means that changes in performance can be
attributed primarily to the selected EEG regions rather than to a different
evaluation procedure.

## Question 1 — Global five-region baseline

> How accurately can the classifiers distinguish ADHD and control subjects
> when all available EEG regions are used?

The global representation includes all five regions and all 19 channels:

```text
Frontal + Central + Parietal + Temporal + Occipital
19 channels × 5 frequency bands = 95 features
```

This experiment compares all supported algorithms under the most complete
feature representation. Its purpose is to establish the reference performance
that every regional experiment must be compared against.

Primary outputs:

- one subject-level result for every classifier;
- combined ROC comparison;
- balanced accuracy, sensitivity, specificity, F1, and ROC-AUC;
- confusion matrices and interpretable feature plots.

This experiment is complete. Random Forest currently has the highest global
ROC-AUC, while RBF SVM has the highest global balanced accuracy.

## Question 2 — Sets of three regions

> Can a subset of three brain regions outperform the global five-region model?

This is the main experiment derived from the selected research paper. It tests
all ten unique combinations:

```text
F-C-P   F-C-T   F-C-O   F-P-T   F-P-O
F-T-O   C-P-T   C-P-O   C-T-O   P-T-O
```

Where:

```text
F = Frontal    C = Central    P = Parietal
T = Temporal   O = Occipital
```

Every classifier is evaluated on every combination using the same subject
folds as the global experiment. The principal comparison is:

```text
delta ROC-AUC = best three-region ROC-AUC - global ROC-AUC
```

A positive observed difference means that the subset scored higher in these
folds. It does not by itself prove a statistically reliable improvement because
the best result is selected from ten candidates. A final claim should include
paired subject-level confidence intervals or nested cross-validation.

Primary outputs:

- model-by-combination ROC-AUC heatmap;
- ranked CSV and JSON results;
- global-versus-best-three-region comparison;
- per-combination predictions, metrics, ROC curves, and confusion matrices.

## Question 3 — Individual-region contribution

> How much discriminative information does each brain region provide when used
> alone?

This experiment evaluates five independent feature subsets:

| Region | Channels | Features |
| --- | ---: | ---: |
| Frontal | 7 | 35 |
| Central | 3 | 15 |
| Parietal | 3 | 15 |
| Temporal | 4 | 20 |
| Occipital | 2 | 10 |

The purpose is not only to find the strongest isolated region. It also helps
explain the three-region results. A strong combination may contain individually
strong regions, or it may reveal complementary information from regions that
are weaker alone.

Primary outputs:

- model-by-region performance heatmap;
- ranking of the five individual regions;
- comparison with the global baseline;
- comparison between individual regions and the best three-region subset.

## Final interpretation

The three questions provide different levels of evidence:

```text
Global experiment
    establishes the full-information baseline

Three-region experiment
    tests whether fewer regions preserve or improve performance

One-region experiment
    explains the contribution of each anatomical region
```

The final conclusion should report whether the best three-region subset exceeds
the global baseline, which individual regions contribute most, and whether the
performance gain is large and stable enough to justify using fewer electrodes.

