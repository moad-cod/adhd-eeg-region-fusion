"""Extremely randomized trees classifier."""

from sklearn.ensemble import ExtraTreesClassifier


def create_model(random_state: int):
    return ExtraTreesClassifier(
        n_estimators=500,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )

