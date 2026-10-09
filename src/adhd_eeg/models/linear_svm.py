"""Linear support-vector machine."""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def create_model(random_state: int):
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", class_weight="balanced", random_state=random_state),
    )
