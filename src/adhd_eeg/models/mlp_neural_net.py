"""Small multilayer-perceptron neural network."""

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def create_model(random_state: int):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            early_stopping=True,
            max_iter=500,
            random_state=random_state,
        ),
    )
