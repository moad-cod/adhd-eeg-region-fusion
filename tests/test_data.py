import numpy as np

from adhd_eeg.data import segment_signal


def test_segmentation_drops_incomplete_tail():
    samples = np.arange(22).reshape(11, 2)
    windows = segment_signal(samples, window_samples=4)
    assert windows.shape == (2, 4, 2)
    np.testing.assert_array_equal(windows[1], samples[4:8])


def test_segmentation_returns_empty_when_recording_is_short():
    windows = segment_signal(np.zeros((3, 19)), window_samples=4)
    assert windows.shape == (0, 4, 19)

