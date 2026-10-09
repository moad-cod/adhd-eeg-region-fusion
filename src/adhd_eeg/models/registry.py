"""Central mapping from CLI model names to model factories."""

from collections.abc import Iterable

from .extra_trees import create_model as create_extra_trees
from .knn import create_model as create_knn
from .linear_svm import create_model as create_linear_svm
from .logistic import create_model as create_logistic
from .mlp_neural_net import create_model as create_mlp_neural_net
from .naive_bayes import create_model as create_naive_bayes
from .random_forest import create_model as create_random_forest
from .rbf_svm import create_model as create_rbf_svm
from .xgboost import create_model as create_xgboost

MODEL_FACTORIES = {
    "logistic": create_logistic,
    "linear_svm": create_linear_svm,
    "naive_bayes": create_naive_bayes,
    "rbf_svm": create_rbf_svm,
    "random_forest": create_random_forest,
    "extra_trees": create_extra_trees,
    "xgboost": create_xgboost,
    "knn": create_knn,
    "mlp_neural_net": create_mlp_neural_net,
}
MODEL_NAMES = tuple(MODEL_FACTORIES)
DEFAULT_MODELS = ("logistic", "linear_svm", "naive_bayes")


def model_registry(
    random_state: int, selected_models: Iterable[str] | None = None
) -> dict[str, object]:
    """Construct only the selected models in stable CLI order."""
    requested = set(DEFAULT_MODELS if selected_models is None else selected_models)
    unknown = requested.difference(MODEL_FACTORIES)
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(sorted(unknown))}")
    return {
        name: factory(random_state)
        for name, factory in MODEL_FACTORIES.items()
        if name in requested
    }
