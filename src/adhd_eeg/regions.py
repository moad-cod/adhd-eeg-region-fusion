"""Brain-region definitions and feature selection for regional experiments."""

from collections.abc import Iterable

import numpy as np

from .config import BANDS, REGIONS

REGION_CODES = {
    "F": "frontal",
    "C": "central",
    "P": "parietal",
    "T": "temporal",
    "O": "occipital",
}

SINGLE_REGIONS = tuple(REGION_CODES.values())

THREE_REGION_COMBINATIONS = {
    "F-C-P": ("F", "C", "P"),
    "F-C-T": ("F", "C", "T"),
    "F-C-O": ("F", "C", "O"),
    "F-P-T": ("F", "P", "T"),
    "F-P-O": ("F", "P", "O"),
    "F-T-O": ("F", "T", "O"),
    "C-P-T": ("C", "P", "T"),
    "C-P-O": ("C", "P", "O"),
    "C-T-O": ("C", "T", "O"),
    "P-T-O": ("P", "T", "O"),
}


def channels_for_combination(combination: str) -> tuple[str, ...]:
    """Return unique channels for one named three-region combination."""
    try:
        codes = THREE_REGION_COMBINATIONS[combination]
    except KeyError as error:
        raise ValueError(f"Unknown three-region combination: {combination}") from error
    channels = []
    for code in codes:
        channels.extend(REGIONS[REGION_CODES[code]])
    if len(channels) != len(set(channels)):
        raise ValueError(f"Duplicate channel in combination {combination}")
    return tuple(channels)


def select_combination_features(
    X: np.ndarray,
    feature_names: Iterable[str],
    combination: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Select all bandpower columns belonging to a three-region combination."""
    names = np.asarray(list(feature_names), dtype=str)
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError("X columns and feature_names must have the same length")
    channels = channels_for_combination(combination)
    channel_set = set(channels)
    mask = np.asarray([name.rsplit("_", 1)[-1] in channel_set for name in names])
    expected = len(channels) * len(BANDS)
    if int(mask.sum()) != expected:
        raise ValueError(
            f"{combination} selected {int(mask.sum())} features; expected {expected}"
        )
    return X[:, mask], names[mask], channels


def select_region_features(
    X: np.ndarray,
    feature_names: Iterable[str],
    region: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Select all bandpower columns belonging to one anatomical region."""
    if region not in SINGLE_REGIONS:
        raise ValueError(f"Unknown EEG region: {region}")
    names = np.asarray(list(feature_names), dtype=str)
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError("X columns and feature_names must have the same length")
    channels = tuple(REGIONS[region])
    channel_set = set(channels)
    mask = np.asarray([name.rsplit("_", 1)[-1] in channel_set for name in names])
    expected = len(channels) * len(BANDS)
    if int(mask.sum()) != expected:
        raise ValueError(
            f"{region} selected {int(mask.sum())} features; expected {expected}"
        )
    return X[:, mask], names[mask], channels

