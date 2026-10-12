"""Evaluate all sets of three EEG regions against the global baseline."""

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
from .regional_visualization import compare_with_global, save_region_heatmap, save_regional_summary
from .regions import THREE_REGION_COMBINATIONS, select_combination_features


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare all three-region ADHD EEG subsets")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/three_region_benchmark"))
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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--combinations",
        nargs="+",
        choices=tuple(THREE_REGION_COMBINATIONS),
        default=list(THREE_REGION_COMBINATIONS),
    )
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


def _validate_global_configuration(args) -> None:
    if not args.global_metrics.exists():
        return
    configuration = json.loads(args.global_metrics.read_text()).get("configuration", {})
    if configuration.get("folds") != args.folds or configuration.get("random_state") != args.random_state:
        raise ValueError(
            "Global and regional folds/random state differ. Re-run the global benchmark "
            "with matching --folds and --random-state before comparing results."
        )


def _write_metrics_checkpoint(path: Path, metrics: dict) -> None:
    """Atomically persist progress so an interrupted run can be resumed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as file:
        json.dump(metrics, file, indent=2)
    temporary.replace(path)


def _load_checkpoint(
    path: Path, expected_configuration: dict, restart: bool
) -> dict:
    if restart or not path.exists():
        return {"configuration": expected_configuration, "models": {}}
    checkpoint = json.loads(path.read_text())
    if checkpoint.get("configuration") != expected_configuration:
        raise ValueError(
            f"Checkpoint configuration differs at {path}. "
            "Use --restart or select a new output directory."
        )
    return checkpoint


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
    for combination in args.combinations:
        regional_X, regional_names, channels = select_combination_features(
            X, feature_names, combination
        )
        combination_dir = args.output / "combinations" / combination
        print(
            f"Evaluating {combination}: {len(channels)} channels, "
            f"{regional_X.shape[1]} features"
        )
        models = model_registry(args.random_state, args.models)
        predictions = {}
        prediction_frames = []
        configuration = {
            "combination": combination,
            "channels": list(channels),
            "feature_count": int(regional_X.shape[1]),
            "folds": args.folds,
            "random_state": args.random_state,
        }
        metrics_path = combination_dir / "metrics.json"
        combination_metrics = _load_checkpoint(
            metrics_path, configuration, args.restart
        )
        _write_metrics_checkpoint(metrics_path, combination_metrics)
        for model_name, model in models.items():
            model_dir = combination_dir / "models" / model_name
            model_predictions_path = model_dir / "subject_predictions.csv"
            saved_model_metrics = combination_metrics["models"].get(model_name)
            if (
                not args.restart
                and saved_model_metrics is not None
                and model_predictions_path.exists()
            ):
                subjects = pd.read_csv(model_predictions_path)
                fold_metrics = saved_model_metrics["folds"]
                overall = saved_model_metrics["overall"]
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
                subjects.to_csv(model_predictions_path, index=False)
                combination_metrics["models"][model_name] = {
                    "overall": overall,
                    "folds": fold_metrics,
                }
                save_plots(subjects, model_name, model_dir)
                if importance is not None:
                    save_top_feature_importance(
                        importance, regional_names, model_name, model_dir
                    )
                _write_metrics_checkpoint(metrics_path, combination_metrics)
                print(
                    f"  {model_name}: BA={overall['balanced_accuracy']:.3f}, "
                    f"AUC={overall['roc_auc']:.3f} [checkpoint saved]"
                )
            predictions[model_name] = subjects
            prediction_frames.append(subjects.assign(model=model_name))
            rows.append(
                {
                    "combination": combination,
                    "model": model_name,
                    "region_count": 3,
                    "channel_count": len(channels),
                    "feature_count": int(regional_X.shape[1]),
                    **{key: value for key, value in overall.items() if key != "subjects"},
                }
            )

        comparison_dir = combination_dir / "model_comparison"
        save_multi_model_roc(predictions, comparison_dir)
        save_benchmark_summary(predictions, comparison_dir)
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            combination_dir / "subject_predictions.csv", index=False
        )
        _write_metrics_checkpoint(metrics_path, combination_metrics)

    summary = pd.DataFrame(rows)
    regional_comparison_dir = args.comparison_root / "three_region_models"
    save_regional_summary(summary, regional_comparison_dir)
    save_region_heatmap(summary, regional_comparison_dir)

    if args.global_summary.exists() and args.max_subjects_per_class is None:
        _validate_global_configuration(args)
        global_summary = pd.read_csv(args.global_summary)
        missing_models = set(args.models).difference(global_summary["model"])
        if missing_models:
            raise ValueError(
                "Global summary is missing requested model(s): "
                + ", ".join(sorted(missing_models))
            )
        global_summary = global_summary[global_summary["model"].isin(args.models)]
        compare_with_global(
            summary, global_summary, args.comparison_root / "feature_scopes"
        )
    else:
        print("Skipped global comparison: matching full global results were not available.")
    print(f"Regional results written to {args.output}")


if __name__ == "__main__":
    main()

