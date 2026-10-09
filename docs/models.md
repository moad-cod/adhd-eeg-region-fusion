# Model catalog

Every classifier lives in `src/adhd_eeg/models/` and exposes a
`create_model(random_state)` factory. The central registry maps CLI names to
these factories without putting algorithm configuration in the experiment
runner.

| CLI name | Estimator | Scaling | Native interpretation |
| --- | --- | --- | --- |
| `logistic` | Logistic regression | Yes | Absolute coefficients |
| `linear_svm` | Linear SVM | Yes | Absolute coefficients |
| `naive_bayes` | Gaussian Naive Bayes | Yes | None used |
| `rbf_svm` | RBF SVM | Yes | None |
| `random_forest` | Random forest | No | Feature importance |
| `extra_trees` | Extremely randomized trees | No | Feature importance |
| `xgboost` | Gradient-boosted trees | No | Feature importance |
| `knn` | Distance-weighted KNN | Yes | None |
| `mlp_neural_net` | Two-layer MLP | Yes | First-layer absolute weights |

XGBoost is optional and imported only when requested. Install it with
`pip install -e ".[xgboost]"`.
