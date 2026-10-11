"""Subject-level metrics, benchmark plots, and model interpretability."""

import json

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)

PLOT_DPI = 300
PLOT_FORMAT = "png"


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


def save_benchmark_summary(
    model_predictions: dict[str, pd.DataFrame], output_dir: Path
) -> pd.DataFrame:
    """Save the ranked benchmark in both tabular CSV and structured JSON."""
    summary = benchmark_summary(model_predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "benchmark_summary.csv", index=False)
    with (output_dir / "benchmark_summary.json").open("w") as file:
        json.dump(
            {"models_ranked_by_roc_auc": summary.to_dict(orient="records")},
            file,
            indent=2,
        )
    return summary


def _save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{stem}.{PLOT_FORMAT}",
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
    model_predictions: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    highlighted_models: int = 4,
    include_inset: bool = True,
) -> None:
    """Save a publication-ready, AUC-ranked subject-level ROC comparison."""
    if not model_predictions:
        raise ValueError("model_predictions must contain at least one model")
    if highlighted_models < 0:
        raise ValueError("highlighted_models cannot be negative")

    curves = []
    for model_name, subjects in model_predictions.items():
        y = subjects["label"].to_numpy()
        probability = subjects["probability"].to_numpy()
        fpr, tpr, _ = roc_curve(y, probability)
        curves.append(
            {
                "name": model_name,
                "label": _display_model_name(model_name),
                "fpr": fpr,
                "tpr": tpr,
                "auc": float(roc_auc_score(y, probability)),
            }
        )
    curves.sort(key=lambda curve: curve["auc"], reverse=True)

    top_count = min(highlighted_models, len(curves))
    strong_colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]
    if top_count > len(strong_colors):
        strong_colors = sns.color_palette("colorblind", n_colors=top_count)
    muted_count = len(curves) - top_count
    muted_colors = sns.color_palette("muted", n_colors=max(muted_count, 1))
    baseline_styles = ["--", "-.", ":"]

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(11.5, 7.2))
        inset = None
        if include_inset:
            inset = inset_axes(ax, width="42%", height="42%", loc="lower right", borderpad=1.4)

        for index, curve in enumerate(curves):
            is_highlighted = index < top_count
            color = strong_colors[index] if is_highlighted else muted_colors[index - top_count]
            line_style = "-" if is_highlighted else baseline_styles[(index - top_count) % 3]
            line_width = 3.0 if is_highlighted else 1.5
            alpha = 1.0 if is_highlighted else 0.58
            label = f"{curve['label']}  (AUC = {curve['auc']:.3f})"
            ax.step(
                curve["fpr"],
                curve["tpr"],
                where="post",
                color=color,
                linestyle=line_style,
                linewidth=line_width,
                alpha=alpha,
                label=label,
                zorder=3 if is_highlighted else 2,
            )
            if inset is not None:
                inset.step(
                    curve["fpr"],
                    curve["tpr"],
                    where="post",
                    color=color,
                    linestyle=line_style,
                    linewidth=2.0 if is_highlighted else 1.0,
                    alpha=alpha,
                )

        chance_style = {
            "color": "#666666",
            "linestyle": "--",
            "linewidth": 1.5,
            "alpha": 0.7,
        }
        ax.plot([0, 1], [0, 1], label="Random chance  (AUC = 0.500)", **chance_style)
        ax.set(
            xlim=(-0.01, 1.01),
            ylim=(-0.01, 1.01),
            xlabel="Subject-level false positive rate",
            ylabel="Subject-level true positive rate",
            title="Global EEG Benchmark: Subject-Level ROC Comparison",
        )
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
        ax.set_title(ax.get_title(), fontsize=17, fontweight="bold", pad=14)
        ax.tick_params(labelsize=10)
        ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.65)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            frameon=False,
            fontsize=9.5,
            title="Models ranked by ROC-AUC",
            title_fontsize=10,
        )

        if inset is not None:
            inset.plot([0, 0.3], [0, 0.3], **chance_style)
            inset.set_xlim(0.0, 0.3)
            inset_tpr_max = max(
                np.max(curve["tpr"][curve["fpr"] <= 0.3]) for curve in curves
            )
            inset.set_ylim(0.0, min(1.0, inset_tpr_max + 0.08))
            inset.set_title("Low-FPR region", fontsize=9, pad=4)
            inset.set_xlabel("FPR", fontsize=8)
            inset.set_ylabel("TPR", fontsize=8)
            inset.tick_params(labelsize=7)
            inset.grid(True, color="#E2E2E2", linewidth=0.6, alpha=0.7)
            mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="#777777", alpha=0.55)

        fig.subplots_adjust(left=0.09, bottom=0.11, right=0.72, top=0.90)
        _save_figure(fig, output_dir, "all_models_subject_roc")
        plt.close(fig)


def _display_model_name(model_name: str) -> str:
    """Convert internal CLI names into publication-friendly labels."""
    labels = {
        "logistic": "Logistic Regression",
        "linear_svm": "Linear SVM",
        "naive_bayes": "Naive Bayes",
        "rbf_svm": "RBF SVM",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "xgboost": "XGBoost",
        "knn": "KNN",
        "mlp_neural_net": "MLP Neural Net",
    }
    return labels.get(model_name, model_name.replace("_", " ").title())


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
