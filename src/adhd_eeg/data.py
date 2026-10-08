"""Dataset validation and subject-level loading."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CHANNELS

LABEL_MAP = {"Control": 0, "ADHD": 1}


def load_subjects(csv_path: str | Path) -> Iterator[tuple[str, int, np.ndarray]]:
    """Yield ``(subject_id, label, samples_x_channels)`` in file order.

    The source file stores each subject in one contiguous block. Loading only one
    subject at a time keeps memory use modest despite the two-million-row CSV.
    """
    path = Path(csv_path)
    required = [*CHANNELS, "Class", "ID"]
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(required) - set(header.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    seen: set[str] = set()
    carry: pd.DataFrame | None = None
    for chunk in pd.read_csv(path, usecols=required, chunksize=100_000):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last_id = str(chunk["ID"].iloc[-1])
        carry = chunk.loc[chunk["ID"].astype(str) == last_id]
        complete = chunk.loc[chunk["ID"].astype(str) != last_id]
        for item in _frames_to_subjects(complete, seen):
            yield item

    if carry is not None and not carry.empty:
        yield from _frames_to_subjects(carry, seen)


def _frames_to_subjects(
    frame: pd.DataFrame, seen: set[str]
) -> Iterator[tuple[str, int, np.ndarray]]:
    for subject_id, group in frame.groupby("ID", sort=False):
        subject_id = str(subject_id)
        if subject_id in seen:
            raise ValueError(f"Subject {subject_id!r} occurs in non-contiguous blocks")
        seen.add(subject_id)
        labels = group["Class"].unique()
        if len(labels) != 1 or labels[0] not in LABEL_MAP:
            raise ValueError(f"Invalid label(s) for subject {subject_id}: {labels}")
        samples = group.loc[:, CHANNELS].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(samples).all():
            raise ValueError(f"Non-finite EEG value found for subject {subject_id}")
        yield subject_id, LABEL_MAP[str(labels[0])], samples


def segment_signal(samples: np.ndarray, window_samples: int) -> np.ndarray:
    """Return complete non-overlapping windows without crossing a subject."""
    if samples.ndim != 2:
        raise ValueError("samples must have shape (time, channels)")
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    count = len(samples) // window_samples
    if count == 0:
        return np.empty((0, window_samples, samples.shape[1]), dtype=samples.dtype)
    return samples[: count * window_samples].reshape(
        count, window_samples, samples.shape[1]
    )

