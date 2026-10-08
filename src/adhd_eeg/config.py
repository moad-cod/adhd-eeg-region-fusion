"""Constants shared by loading, feature extraction, and experiments."""

from collections import OrderedDict

FS = 128
WINDOW_SECONDS = 8
WINDOW_SAMPLES = FS * WINDOW_SECONDS

CHANNELS = (
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
)

# T7/T8/P7/P8 are the modern names for the paper's T3/T4/T5/T6.
REGIONS = {
    "frontal": ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"),
    "central": ("C3", "C4", "Cz"),
    "parietal": ("P3", "P4", "Pz"),
    "temporal": ("T7", "T8", "P7", "P8"),
    "occipital": ("O1", "O2"),
}

BANDS = OrderedDict(
    delta=(0.5, 4.0),
    theta=(4.0, 8.0),
    alpha=(8.0, 13.0),
    beta=(13.0, 30.0),
    lower_gamma=(30.0, 45.0),
)

