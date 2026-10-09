"""Random-forest classifier."""

from sklearn.ensemble import RandomForestClassifier


def create_model(random_state: int):
    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )

