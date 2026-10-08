"""Subject aggregation, metrics, and compact research artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)


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


def save_plots(subjects: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    y = subjects["label"].to_numpy()
    pred = subjects["prediction"].to_numpy()
    prob = subjects["probability"].to_numpy()

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["Control", "ADHD"], ax=ax, colorbar=False)
    ax.set_title(f"{model_name}: subject-level confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y, prob, name=model_name, ax=ax)
    ax.set_title(f"{model_name}: subject-level ROC")
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_roc.png", dpi=160)
    plt.close(fig)

