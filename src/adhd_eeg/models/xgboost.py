"""Optional XGBoost classifier."""


def create_model(random_state: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise ImportError(
            "The xgboost model requires the optional dependency. "
            "Install it with: pip install -e '.[xgboost]'"
        ) from error

    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random_state,
    )

