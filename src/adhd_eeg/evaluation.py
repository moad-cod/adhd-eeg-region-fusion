"""Subject-level metrics, benchmark plots, and model interpretability."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)

PLOT_DPI = 220
PLOT_FORMATS = ("png", "svg")


def aggregate_subject_predictions(
    groups: np.ndarray, labels: np.ndarray, probabilities: np.ndarray
) -> pd.DataFrame:
    frame = pd.DataFrame({"subject_id": groups, "label": labels, "probability": probabilities})
    label_counts = frame.groupby("subject_id")["label"].nunique()
    if (label_counts != 1).any():
        raise ValueError("A subject has conflicting labels")
    return (
        frame.groupby("subject_id", as_index=False, sort=False)
        .agg(label=("label", "first"), probability=("probability", "mean"), windows=("label", "size"))
        .assign(prediction=lambda x: (x["probability"] >= 0.5).astype(int))
    )


def classification_metrics(subjects: pd.DataFrame) -> dict[str, float | int]:
    y = subjects["label"].to_numpy()
    pred = subjects["prediction"].to_numpy()
    prob = subjects["probability"].to_numpy()
    return {
        "subjects": int(len(subjects)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
    }


def benchmark_summary(model_predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one subject-level clinical-metric row per evaluated model."""
    rows = []
    for model_name, subjects in model_predictions.items():
        metrics = classification_metrics(subjects)
        rows.append(
            {
                "model": model_name,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "f1_score": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )
    columns = [
        "model",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
        "roc_auc",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        "roc_auc", ascending=False, ignore_index=True
    )


def _save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in PLOT_FORMATS:
        fig.savefig(
            output_dir / f"{stem}.{extension}",
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )


def save_plots(subjects: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    """Save polished subject-level confusion-matrix and ROC diagnostics."""
    y = subjects["label"].to_numpy()
    pred = subjects["prediction"].to_numpy()
    prob = subjects["probability"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y,
        pred,
        display_labels=["Control", "ADHD"],
        ax=ax,
        colorbar=False,
        cmap="Blues",
    )
    ax.set_title(f"{model_name}: Subject-Level Confusion Matrix")
    ax.set_xlabel("Predicted subject diagnosis")
    ax.set_ylabel("True subject diagnosis")
    fig.tight_layout()
    _save_figure(fig, output_dir, f"{model_name}_confusion_matrix")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2.2, label=f"{model_name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="0.45", label="Random chance (AUC = 0.50)")
    ax.set_title(f"{model_name}: Subject-Level ROC")
    ax.set_xlabel("Subject-level false positive rate")
    ax.set_ylabel("Subject-level true positive rate")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_figure(fig, output_dir, f"{model_name}_roc")
    plt.close(fig)


def save_multi_model_roc(
    model_predictions: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Overlay all models' out-of-fold subject-level ROC curves."""
    if not model_predictions:
        raise ValueError("model_predictions must contain at least one model")

    colors = plt.colormaps["tab10"].resampled(len(model_predictions))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for index, (model_name, subjects) in enumerate(model_predictions.items()):
        y = subjects["label"].to_numpy()
        probability = subjects["probability"].to_numpy()
        fpr, tpr, _ = roc_curve(y, probability)
        auc = roc_auc_score(y, probability)
        ax.plot(
            fpr,
            tpr,
            color=colors(index),
            linewidth=2.2,
            label=f"{model_name} (AUC = {auc:.3f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color="0.35",
        linewidth=1.5,
        label="Random chance (AUC = 0.50)",
    )
    ax.set_title("Global EEG Benchmark: Subject-Level ROC Comparison")
    ax.set_xlabel("Subject-level false positive rate")
    ax.set_ylabel("Subject-level true positive rate")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_dir, "all_models_subject_roc")
    plt.close(fig)


def extract_feature_importance(fitted_model) -> np.ndarray | None:
    """Extract native input importance from an estimator or fitted pipeline."""
    estimator = fitted_model.steps[-1][1] if hasattr(fitted_model, "steps") else fitted_model
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float)
    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        return np.mean(np.abs(np.atleast_2d(coefficients)), axis=0)
    if hasattr(estimator, "coefs_") and estimator.coefs_:
        # Mean absolute weights from each input feature into the first hidden layer.
        return np.mean(np.abs(np.asarray(estimator.coefs_[0], dtype=float)), axis=1)
    return None


def save_top_feature_importance(
    importances: np.ndarray,
    feature_names: list[str] | np.ndarray,
    model_name: str,
    output_dir: Path,
    top_n: int = 15,
) -> None:
    """Save a Top-N horizontal chart of channel-band feature importance."""
    values = np.asarray(importances, dtype=float).reshape(-1)
    names = np.asarray(feature_names, dtype=str).reshape(-1)
    if len(values) != len(names):
        raise ValueError("importances and feature_names must have the same length")
    if not np.isfinite(values).all():
        raise ValueError("importances must contain only finite values")
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    values = np.abs(values)
    selected = np.argsort(values)[-min(top_n, len(values)) :]
    selected = selected[np.argsort(values[selected])]
    fig_height = max(5.5, 0.36 * len(selected))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    palette = plt.colormaps["viridis"](np.linspace(0.25, 0.9, len(selected)))
    ax.barh(names[selected], values[selected], color=palette)
    ax.set_title(f"{model_name}: Top {len(selected)} Window-Level EEG Features")
    ax.set_xlabel("Absolute coefficient or native feature importance")
    ax.set_ylabel("EEG frequency-band and channel feature")
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    fig.tight_layout()
    _save_figure(fig, output_dir, f"{model_name}_top_features")
    plt.close(fig)
