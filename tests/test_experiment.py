from adhd_eeg.experiment import parse_args
from adhd_eeg.models import DEFAULT_MODELS, MODEL_NAMES, model_registry


def test_requested_sklearn_models_are_registered_without_xgboost():
    requested = [
        "rbf_svm",
        "random_forest",
        "extra_trees",
        "knn",
        "mlp_neural_net",
    ]
    assert list(model_registry(42, requested)) == requested


def test_cli_defaults_remain_the_original_baselines():
    args = parse_args(["--data", "dataset.csv"])
    assert args.models == list(DEFAULT_MODELS)
    assert "xgboost" in MODEL_NAMES
