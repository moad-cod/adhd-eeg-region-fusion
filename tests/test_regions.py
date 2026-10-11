import numpy as np
import pytest

from adhd_eeg.features import feature_names
from adhd_eeg.regional_experiment import _load_checkpoint, _write_metrics_checkpoint
from adhd_eeg.regions import (
    THREE_REGION_COMBINATIONS,
    channels_for_combination,
    select_combination_features,
)


def test_all_ten_three_region_combinations_are_defined():
    assert len(THREE_REGION_COMBINATIONS) == 10
    assert len(set(THREE_REGION_COMBINATIONS)) == 10


@pytest.mark.parametrize(
    ("combination", "channel_count", "feature_count"),
    [("F-C-P", 13, 65), ("F-T-O", 13, 65), ("C-P-O", 8, 40)],
)
def test_combination_selects_expected_channels_and_features(
    combination, channel_count, feature_count
):
    names = np.asarray(feature_names())
    X = np.arange(2 * len(names)).reshape(2, len(names))
    selected, selected_names, channels = select_combination_features(
        X, names, combination
    )
    assert len(channels) == channel_count
    assert selected.shape == (2, feature_count)
    assert len(selected_names) == feature_count


def test_unknown_combination_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        channels_for_combination("F-C-X")


def test_regional_checkpoint_round_trip_and_configuration_guard(tmp_path):
    path = tmp_path / "metrics.json"
    configuration = {"combination": "F-C-P", "folds": 5, "random_state": 42}
    metrics = {"configuration": configuration, "models": {"logistic": {}}}
    _write_metrics_checkpoint(path, metrics)
    assert _load_checkpoint(path, configuration, restart=False) == metrics
    with pytest.raises(ValueError, match="Checkpoint configuration differs"):
        _load_checkpoint(path, {**configuration, "folds": 2}, restart=False)

