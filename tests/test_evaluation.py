import numpy as np
import pandas as pd
import pytest

from adhd_eeg.evaluation import (
    aggregate_subject_predictions,
    benchmark_summary,
    classification_metrics,
    save_multi_model_roc,
    save_top_feature_importance,
)


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


def _subject_predictions(probabilities):
    labels = np.array([0, 0, 1, 1])
    return pd.DataFrame(
        {
            "subject_id": ["a", "b", "c", "d"],
            "label": labels,
            "probability": probabilities,
            "prediction": (np.asarray(probabilities) >= 0.5).astype(int),
            "windows": 1,
        }
    )


def test_benchmark_summary_is_sorted_by_subject_auc():
    predictions = {
        "weaker": _subject_predictions([0.1, 0.7, 0.6, 0.8]),
        "stronger": _subject_predictions([0.1, 0.2, 0.8, 0.9]),
    }
    summary = benchmark_summary(predictions)
    assert list(summary["model"]) == ["stronger", "weaker"]
    assert list(summary.columns) == [
        "model",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
        "roc_auc",
    ]


def test_benchmark_and_feature_plots_save_png_and_svg(tmp_path):
    predictions = {"model_a": _subject_predictions([0.1, 0.2, 0.8, 0.9])}
    save_multi_model_roc(predictions, tmp_path)
    save_top_feature_importance(
        np.array([0.2, -0.9, 0.4]),
        ["delta_Fp1", "theta_Fp1", "beta_Pz"],
        "model_a",
        tmp_path,
        top_n=2,
    )
    assert (tmp_path / "all_models_subject_roc.png").is_file()
    assert (tmp_path / "all_models_subject_roc.svg").is_file()
    assert (tmp_path / "model_a_top_features.png").is_file()
    assert (tmp_path / "model_a_top_features.svg").is_file()


def test_feature_plot_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError, match="same length"):
        save_top_feature_importance([0.2], ["delta_Fp1", "theta_Fp1"], "bad", tmp_path)
