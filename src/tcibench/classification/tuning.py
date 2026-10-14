"""Hyperparameter tuning for classification models."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold


CLASSIFICATION_GRIDS = {
    "xgboost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.10],
    },
    "random_forest": {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
    },
    "logistic_regression": {
        "C": [0.005, 0.05, 0.5, 5, 50],
    },
}


def tune_classifier(
    model_name: str,
    estimator: Any,
    features: pd.DataFrame,
    target: Any,
    *,
    seed: int,
) -> tuple[dict[str, Any], float]:
    """Tune a classifier using only the original training data."""

    if model_name not in CLASSIFICATION_GRIDS:
        raise ValueError(
            f"Classifier '{model_name}' does not have a tuning grid."
        )

    grid = CLASSIFICATION_GRIDS[model_name]

    search = GridSearchCV(
        estimator=estimator,
        param_grid={
            f"classifier__{name}": values
            for name, values in grid.items()
        },
        scoring="f1_macro",
        cv=StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        ),
        n_jobs=1,
        refit=False,
    )

    search.fit(features, target)

    best_params = {
        name.removeprefix("classifier__"): value
        for name, value in search.best_params_.items()
    }

    return best_params, float(search.best_score_)