"""Evaluate the standalone predictive value of each EEG region."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import (
    classification_metrics,
    save_benchmark_summary,
    save_multi_model_roc,
    save_plots,
    save_top_feature_importance,
)
from .experiment import cross_validate_model
from .features import build_feature_dataset
from .models import DEFAULT_MODELS, MODEL_NAMES, model_registry
from .regional_experiment import (
    _load_checkpoint,
    _validate_global_configuration,
    _write_metrics_checkpoint,
)
from .regions import SINGLE_REGIONS, select_region_features
from .single_region_visualization import (
    compare_feature_scopes,
    save_single_region_heatmap,
    save_single_region_summary,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure the standalone predictive value of each EEG region"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/single_region_benchmark"))
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("outputs/global_five_region_benchmark/features.npz"),
    )
    parser.add_argument(
        "--comparison-root", type=Path, default=Path("outputs/comparisons")
    )
    parser.add_argument(
        "--global-summary",
        type=Path,
        default=Path("outputs/comparisons/global_models/benchmark_summary.csv"),
    )
    parser.add_argument(
        "--global-metrics",
        type=Path,
        default=Path("outputs/global_five_region_benchmark/metrics.json"),
    )
    parser.add_argument(
        "--three-region-summary",
        type=Path,
        default=Path("outputs/comparisons/three_region_models/three_region_summary.csv"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(DEFAULT_MODELS))
    parser.add_argument("--regions", nargs="+", choices=SINGLE_REGIONS, default=list(SINGLE_REGIONS))
    parser.add_argument("--max-subjects-per-class", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore compatible per-model checkpoints and recompute results",
    )
    return parser.parse_args(argv)


def _load_dataset(args) -> dict[str, np.ndarray]:
    use_shared_cache = args.max_subjects_per_class is None
    if use_shared_cache and args.feature_cache.exists() and not args.rebuild_cache:
        loaded = np.load(args.feature_cache)
        print(f"Loaded global feature cache from {args.feature_cache}")
        return {name: loaded[name] for name in loaded.files}
    dataset = build_feature_dataset(
        args.data, max_subjects_per_class=args.max_subjects_per_class
    )
    if use_shared_cache:
        args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.feature_cache, **dataset)
        print(f"Saved global feature cache to {args.feature_cache}")
    return dataset


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(args)
    X, y, groups = dataset["X"], dataset["y"], dataset["groups"]
    feature_names = dataset["feature_names"]
    subject_labels = pd.DataFrame({"subject": groups, "label": y}).drop_duplicates()
    if args.folds > int(subject_labels.groupby("label").size().min()):
        raise ValueError("--folds exceeds the smallest subject class")

    rows = []
    for region in args.regions:
        regional_X, regional_names, channels = select_region_features(
            X, feature_names, region
        )
        region_dir = args.output / "regions" / region
        print(
            f"Evaluating {region}: {len(channels)} channels, "
            f"{regional_X.shape[1]} features"
        )
        predictions = {}
        prediction_frames = []
        configuration = {
            "region": region,
            "channels": list(channels),
            "feature_count": int(regional_X.shape[1]),
            "folds": args.folds,
            "random_state": args.random_state,
        }
        metrics_path = region_dir / "metrics.json"
        region_metrics = _load_checkpoint(metrics_path, configuration, args.restart)
        _write_metrics_checkpoint(metrics_path, region_metrics)

        for model_name, model in model_registry(args.random_state, args.models).items():
            model_dir = region_dir / "models" / model_name
            predictions_path = model_dir / "subject_predictions.csv"
            saved = region_metrics["models"].get(model_name)
            if not args.restart and saved is not None and predictions_path.exists():
                subjects = pd.read_csv(predictions_path)
                fold_metrics = saved["folds"]
                overall = saved["overall"]
                importance = None
                print(
                    f"  {model_name}: resumed checkpoint "
                    f"(BA={overall['balanced_accuracy']:.3f}, AUC={overall['roc_auc']:.3f})"
                )
            else:
                try:
                    subjects, fold_metrics, importance = cross_validate_model(
                        model, regional_X, y, groups, args.folds, args.random_state
                    )
                except KeyboardInterrupt as error:
                    print(
                        "\nInterrupted safely. Completed model checkpoints were saved; "
                        "run the same command to resume."
                    )
                    raise SystemExit(130) from error
                overall = classification_metrics(subjects)
                model_dir.mkdir(parents=True, exist_ok=True)
                subjects.to_csv(predictions_path, index=False)
                region_metrics["models"][model_name] = {
                    "overall": overall,
                    "folds": fold_metrics,
                }
                save_plots(subjects, model_name, model_dir)
                if importance is not None:
                    save_top_feature_importance(
                        importance, regional_names, model_name, model_dir
                    )
                _write_metrics_checkpoint(metrics_path, region_metrics)
                print(
                    f"  {model_name}: BA={overall['balanced_accuracy']:.3f}, "
                    f"AUC={overall['roc_auc']:.3f} [checkpoint saved]"
                )
            predictions[model_name] = subjects
            prediction_frames.append(subjects.assign(model=model_name))
            rows.append(
                {
                    "region": region,
                    "model": model_name,
                    "region_count": 1,
                    "channel_count": len(channels),
                    "feature_count": int(regional_X.shape[1]),
                    **{key: value for key, value in overall.items() if key != "subjects"},
                }
            )

        comparison_dir = region_dir / "model_comparison"
        save_multi_model_roc(predictions, comparison_dir)
        save_benchmark_summary(predictions, comparison_dir)
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            region_dir / "subject_predictions.csv", index=False
        )

    summary = pd.DataFrame(rows)
    single_comparison_dir = args.comparison_root / "single_region_models"
    save_single_region_summary(summary, single_comparison_dir)
    save_single_region_heatmap(summary, single_comparison_dir)
    if args.global_summary.exists() and args.max_subjects_per_class is None:
        _validate_global_configuration(args)
        global_summary = pd.read_csv(args.global_summary)
        missing = set(args.models).difference(global_summary["model"])
        if missing:
            raise ValueError("Global summary is missing model(s): " + ", ".join(sorted(missing)))
        three_summary = (
            pd.read_csv(args.three_region_summary)
            if args.three_region_summary.exists()
            else None
        )
        compare_feature_scopes(
            summary,
            global_summary[global_summary["model"].isin(args.models)],
            three_summary,
            args.comparison_root / "feature_scopes",
        )
    else:
        print("Skipped feature-scope comparison: matching global results were unavailable.")
    print(f"Single-region results written to {args.output}")


if __name__ == "__main__":
    main()
