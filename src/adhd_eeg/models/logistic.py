"""Logistic-regression baseline."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def create_model(random_state: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=random_state,
        ),
    )
