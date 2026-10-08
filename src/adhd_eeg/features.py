"""Interpretable frequency-domain features for the global baseline."""

from pathlib import Path

import numpy as np

from .config import BANDS, CHANNELS, FS, WINDOW_SAMPLES
from .data import load_subjects, segment_signal


def bandpower_features(windows: np.ndarray, fs: int = FS) -> np.ndarray:
    """Calculate log10 band power for each window, channel, and EEG band."""
    if windows.ndim != 3:
        raise ValueError("windows must have shape (windows, time, channels)")
    n_time = windows.shape[1]
    taper = np.hanning(n_time).astype(np.float32)
    centered = windows - windows.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered * taper[None, :, None], axis=1)
    psd = (np.abs(spectrum) ** 2) / (fs * np.square(taper).sum())
    frequencies = np.fft.rfftfreq(n_time, d=1 / fs)

    result = []
    for low, high in BANDS.values():
        # Adjacent bands share their boundary frequency, matching common EEG
        # band definitions and making the calculation easy to audit.
        mask = (frequencies >= low) & (frequencies <= high)
        power = np.trapezoid(psd[:, mask, :], frequencies[mask], axis=1)
        result.append(np.log10(np.maximum(power, 1e-12)))
    # Band-major order: delta/all channels, theta/all channels, ...
    return np.concatenate(result, axis=1).astype(np.float32)


def feature_names() -> list[str]:
    return [f"{band}_{channel}" for band in BANDS for channel in CHANNELS]


def build_feature_dataset(
    csv_path: str | Path,
    *,
    max_subjects_per_class: int | None = None,
) -> dict[str, np.ndarray]:
    """Build window features plus labels and subject IDs from the source CSV."""
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    used = {0: 0, 1: 0}

    for subject_id, label, samples in load_subjects(csv_path):
        if max_subjects_per_class is not None and used[label] >= max_subjects_per_class:
            continue
        windows = segment_signal(samples, WINDOW_SAMPLES)
        if len(windows) == 0:
            continue
        subject_features = bandpower_features(windows)
        features.append(subject_features)
        labels.append(np.full(len(windows), label, dtype=np.int8))
        groups.append(np.full(len(windows), subject_id, dtype=f"U{max(16, len(subject_id))}"))
        used[label] += 1

    if not features:
        raise ValueError("No complete EEG windows were produced")
    return {
        "X": np.concatenate(features),
        "y": np.concatenate(labels),
        "groups": np.concatenate(groups),
        "feature_names": np.asarray(feature_names()),
    }

