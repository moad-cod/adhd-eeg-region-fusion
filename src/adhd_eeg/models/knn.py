"""Distance-weighted k-nearest-neighbors classifier."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def create_model(random_state: int):
    del random_state  # KNN has no stochastic training step.
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1),
    )
