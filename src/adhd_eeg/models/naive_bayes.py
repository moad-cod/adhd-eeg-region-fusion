"""Gaussian Naive Bayes baseline."""

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def create_model(random_state: int):
    del random_state  # GaussianNB has no stochastic training step.
    return make_pipeline(StandardScaler(), GaussianNB())

