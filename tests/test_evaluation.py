import numpy as np

from adhd_eeg.evaluation import aggregate_subject_predictions, classification_metrics


def test_subject_probabilities_are_averaged():
    result = aggregate_subject_predictions(
        np.array(["a", "a", "b", "b"]),
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.3, 0.8, 0.6]),
    )
    np.testing.assert_allclose(result["probability"], [0.2, 0.7])
    metrics = classification_metrics(result)
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
