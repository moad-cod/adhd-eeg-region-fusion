"""Cross-combination summaries for the three-region experiment."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .evaluation import PLOT_DPI
from .regions import THREE_REGION_COMBINATIONS


def save_regional_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = summary.sort_values("roc_auc", ascending=False, ignore_index=True)
    ranked.to_csv(output_dir / "three_region_summary.csv", index=False)
    with (output_dir / "three_region_summary.json").open("w") as file:
        json.dump({"results_ranked_by_roc_auc": ranked.to_dict("records")}, file, indent=2)


def save_region_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    order = [name for name in THREE_REGION_COMBINATIONS if name in set(summary["combination"])]
    matrix = summary.pivot(index="model", columns="combination", values="roc_auc")
    matrix = matrix.reindex(columns=order)
    fig_width = max(10, 0.9 * len(order) + 3)
    fig_height = max(5, 0.55 * len(matrix) + 2)
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            vmin=0.5,
            vmax=max(0.8, float(matrix.max().max())),
            linewidths=0.5,
            cbar_kws={"label": "Subject-level ROC-AUC"},
            ax=ax,
        )
        ax.set_title("Three-Region EEG Benchmark by Model", fontsize=15, fontweight="bold")
        ax.set_xlabel("Three-region combination")
        ax.set_ylabel("Classifier")
        fig.tight_layout()
        fig.savefig(output_dir / "region_combination_heatmap.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)


def compare_with_global(
    regional_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Compare each model's best three-region result with its global baseline."""
    best_indices = regional_summary.groupby("model")["roc_auc"].idxmax()
    best = regional_summary.loc[best_indices].copy()
    best = best.rename(
        columns={
            "combination": "best_three_regions",
            "roc_auc": "three_region_roc_auc",
            "balanced_accuracy": "three_region_balanced_accuracy",
        }
    )
    global_columns = global_summary[["model", "roc_auc", "balanced_accuracy"]].rename(
        columns={
            "roc_auc": "global_roc_auc",
            "balanced_accuracy": "global_balanced_accuracy",
        }
    )
    comparison = best.merge(global_columns, on="model", how="inner")
    comparison["delta_roc_auc"] = (
        comparison["three_region_roc_auc"] - comparison["global_roc_auc"]
    )
    comparison["delta_balanced_accuracy"] = (
        comparison["three_region_balanced_accuracy"]
        - comparison["global_balanced_accuracy"]
    )
    comparison = comparison.sort_values("delta_roc_auc", ascending=False, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "global_vs_best_three_regions.csv", index=False)
    with (output_dir / "global_vs_best_three_regions.json").open("w") as file:
        json.dump({"comparison": comparison.to_dict("records")}, file, indent=2)

    plot_data = comparison.melt(
        id_vars=["model", "best_three_regions"],
        value_vars=["global_roc_auc", "three_region_roc_auc"],
        var_name="feature_scope",
        value_name="roc_auc",
    )
    labels = {"global_roc_auc": "Global (5 regions)", "three_region_roc_auc": "Best 3 regions"}
    plot_data["feature_scope"] = plot_data["feature_scope"].map(labels)
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=plot_data,
            x="model",
            y="roc_auc",
            hue="feature_scope",
            palette=["#7A7A7A", "#0072B2"],
            ax=ax,
        )
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
        ax.set_ylim(0.45, max(0.8, float(plot_data["roc_auc"].max()) + 0.04))
        ax.set_title("Global Five Regions vs. Best Three-Region Subset", fontweight="bold")
        ax.set_xlabel("Classifier")
        ax.set_ylabel("Subject-level ROC-AUC")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="Feature scope", frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "global_vs_best_three_regions.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
    return comparison

