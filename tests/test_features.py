import numpy as np

from adhd_eeg.config import CHANNELS, FS
from adhd_eeg.features import bandpower_features, feature_names


def test_theta_signal_has_more_theta_than_alpha_power():
    time = np.arange(FS * 8) / FS
    signal = np.sin(2 * np.pi * 6 * time)
    windows = np.tile(signal[None, :, None], (1, 1, len(CHANNELS)))
    features = bandpower_features(windows)
    theta_first_channel = features[0, len(CHANNELS)]
    alpha_first_channel = features[0, 2 * len(CHANNELS)]
    assert theta_first_channel > alpha_first_channel + 3


def test_feature_names_match_output_width():
    windows = np.zeros((2, FS * 8, len(CHANNELS)))
    assert bandpower_features(windows).shape == (2, len(feature_names()))

