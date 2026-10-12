"""Scientific summaries for the standalone single-region experiment."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .evaluation import PLOT_DPI
from .regions import SINGLE_REGIONS


def save_single_region_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = summary.sort_values("roc_auc", ascending=False, ignore_index=True)
    ranked.to_csv(output_dir / "single_region_summary.csv", index=False)
    with (output_dir / "single_region_summary.json").open("w") as file:
        json.dump({"results_ranked_by_roc_auc": ranked.to_dict("records")}, file, indent=2)


def save_single_region_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    order = [region for region in SINGLE_REGIONS if region in set(summary["region"])]
    matrix = summary.pivot(index="model", columns="region", values="roc_auc")
    matrix = matrix.reindex(columns=order)
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(10, 7))
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
        ax.set_title("Standalone Predictive Value by EEG Region", fontsize=15, fontweight="bold")
        ax.set_xlabel("Single EEG region")
        ax.set_ylabel("Classifier")
        fig.tight_layout()
        fig.savefig(output_dir / "single_region_heatmap.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)


def compare_feature_scopes(
    single_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
    three_region_summary: pd.DataFrame | None,
    output_dir: Path,
) -> pd.DataFrame:
    """Compare each model's best single region with broader feature scopes."""
    best_indices = single_summary.groupby("model")["roc_auc"].idxmax()
    best = single_summary.loc[best_indices, ["model", "region", "roc_auc"]].rename(
        columns={"region": "best_single_region", "roc_auc": "single_region_roc_auc"}
    )
    global_scores = global_summary[["model", "roc_auc"]].rename(
        columns={"roc_auc": "global_roc_auc"}
    )
    comparison = best.merge(global_scores, on="model", how="inner")
    if three_region_summary is not None:
        indices = three_region_summary.groupby("model")["roc_auc"].idxmax()
        three = three_region_summary.loc[
            indices, ["model", "combination", "roc_auc"]
        ].rename(
            columns={
                "combination": "best_three_regions",
                "roc_auc": "three_region_roc_auc",
            }
        )
        comparison = comparison.merge(three, on="model", how="left")
    comparison["single_minus_global_auc"] = (
        comparison["single_region_roc_auc"] - comparison["global_roc_auc"]
    )
    comparison = comparison.sort_values(
        "single_region_roc_auc", ascending=False, ignore_index=True
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "feature_scope_comparison.csv", index=False)
    with (output_dir / "feature_scope_comparison.json").open("w") as file:
        json.dump({"comparison": comparison.to_dict("records")}, file, indent=2)

    value_columns = ["global_roc_auc", "single_region_roc_auc"]
    labels = {
        "global_roc_auc": "Global (5 regions)",
        "single_region_roc_auc": "Best single region",
    }
    if "three_region_roc_auc" in comparison:
        value_columns.insert(1, "three_region_roc_auc")
        labels["three_region_roc_auc"] = "Best 3 regions"
    plot_data = comparison.melt(
        id_vars="model",
        value_vars=value_columns,
        var_name="feature_scope",
        value_name="roc_auc",
    )
    plot_data["feature_scope"] = plot_data["feature_scope"].map(labels)
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(
            data=plot_data,
            x="model",
            y="roc_auc",
            hue="feature_scope",
            palette=["#7A7A7A", "#0072B2", "#D55E00"][: len(value_columns)],
            ax=ax,
        )
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
        ax.set_ylim(0.45, max(0.8, float(plot_data["roc_auc"].max()) + 0.04))
        ax.set_title("Standalone EEG Region vs. Multi-Region Benchmarks", fontweight="bold")
        ax.set_xlabel("Classifier")
        ax.set_ylabel("Subject-level ROC-AUC")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="Feature scope", frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "feature_scope_comparison.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
    return comparison
