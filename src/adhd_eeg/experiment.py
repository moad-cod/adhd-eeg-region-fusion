"""Command-line global all-channel experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from .evaluation import (
    aggregate_subject_predictions,
    benchmark_summary,
    classification_metrics,
    extract_feature_importance,
    save_multi_model_roc,
    save_plots,
    save_top_feature_importance,
)
from .features import build_feature_dataset
from .models import DEFAULT_MODELS, MODEL_NAMES, model_registry


def cross_validate_model(model, X, y, groups, folds: int, random_state: int):
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
    probabilities = np.full(len(y), np.nan, dtype=float)
    fold_rows = []
    fold_importances = []
    for fold, (train, test) in enumerate(splitter.split(X, y, groups), start=1):
        fitted = clone(model).fit(X[train], y[train])
        if hasattr(fitted, "predict_proba"):
            probabilities[test] = fitted.predict_proba(X[test])[:, 1]
        else:
            # The logistic transform preserves the SVM decision boundary at 0.5
            # without relying on SVC's deprecated probability=True mode.
            decision = np.clip(fitted.decision_function(X[test]), -30, 30)
            probabilities[test] = 1.0 / (1.0 + np.exp(-decision))
        importance = extract_feature_importance(fitted)
        if importance is not None:
            fold_importances.append(importance)
        fold_subjects = aggregate_subject_predictions(groups[test], y[test], probabilities[test])
        fold_rows.append({"fold": fold, **classification_metrics(fold_subjects)})
    if np.isnan(probabilities).any():
        raise RuntimeError("Some samples did not receive an out-of-fold prediction")
    subjects = aggregate_subject_predictions(groups, y, probabilities)
    mean_importance = np.mean(fold_importances, axis=0) if fold_importances else None
    return subjects, fold_rows, mean_importance


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the global 19-channel ADHD EEG baseline")
    parser.add_argument("--data", type=Path, required=True, help="Path to adhdata.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs/global"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(DEFAULT_MODELS),
        help="Models to evaluate (default: logistic linear_svm naive_bayes)",
    )
    parser.add_argument("--max-subjects-per-class", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "features.npz"
    # Reduced smoke runs deliberately do not reuse or overwrite the full cache.
    use_cache = args.max_subjects_per_class is None
    if use_cache and cache.exists() and not args.rebuild_cache:
        loaded = np.load(cache)
        dataset = {name: loaded[name] for name in loaded.files}
        print(f"Loaded cached features from {cache}")
    else:
        dataset = build_feature_dataset(
            args.data, max_subjects_per_class=args.max_subjects_per_class
        )
        if use_cache:
            np.savez_compressed(cache, **dataset)
            print(f"Saved feature cache to {cache}")

    X, y, groups = dataset["X"], dataset["y"], dataset["groups"]
    subject_labels = pd.DataFrame({"subject": groups, "label": y}).drop_duplicates()
    min_class = int(subject_labels.groupby("label").size().min())
    if args.folds > min_class:
        raise ValueError(f"--folds={args.folds} exceeds the smallest subject class ({min_class})")
    print(f"Dataset: {len(X)} windows, {len(subject_labels)} subjects, {X.shape[1]} features")

    all_metrics = {
        "configuration": {
            "folds": args.folds,
            "random_state": args.random_state,
            "models": args.models,
            "feature_count": int(X.shape[1]),
            "window_count": int(len(X)),
            "subject_count": int(len(subject_labels)),
        },
        "models": {},
    }
    prediction_frames = []
    model_predictions = {}
    models = model_registry(args.random_state, args.models)
    for name in args.models:
        print(f"Evaluating {name}...")
        subjects, fold_metrics, feature_importance = cross_validate_model(
            models[name], X, y, groups, args.folds, args.random_state
        )
        overall = classification_metrics(subjects)
        all_metrics["models"][name] = {"overall": overall, "folds": fold_metrics}
        prediction_frames.append(subjects.assign(model=name))
        model_predictions[name] = subjects
        save_plots(subjects, name, args.output)
        if feature_importance is not None:
            save_top_feature_importance(
                feature_importance,
                dataset["feature_names"],
                name,
                args.output,
            )
        print(f"  balanced accuracy={overall['balanced_accuracy']:.3f}, AUC={overall['roc_auc']:.3f}")

    save_multi_model_roc(model_predictions, args.output)
    benchmark_summary(model_predictions).to_csv(
        args.output / "benchmark_summary.csv", index=False
    )
    with (args.output / "metrics.json").open("w") as file:
        json.dump(all_metrics, file, indent=2)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        args.output / "subject_predictions.csv", index=False
    )
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
