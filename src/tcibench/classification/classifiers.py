"""Classification models used in the predictive evaluation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_classifier(
    model_name: str,
    features: pd.DataFrame,
    *,
    seed: int,
    model_params: dict[str, Any] | None = None,
):
    """Build a classification model with its required preprocessing."""

    params = model_params or {}

    if model_name == "xgboost":
        return _build_xgboost(features, seed, params)

    if model_name == "random_forest":
        return _build_random_forest(features, seed, params)

    if model_name == "logistic_regression":
        return _build_logistic_regression(features, seed, params)

    if model_name == "tabpfn":
        return _build_tabpfn(features, seed)

    raise ValueError(f"Unknown classifier: {model_name}")


def _build_xgboost(
    features: pd.DataFrame,
    seed: int,
    params: dict[str, Any],
) -> Pipeline:
    """Build an XGBoost classifier with native categorical features."""

    from xgboost import XGBClassifier

    categorical = features.select_dtypes(
        exclude="number"
    ).columns.tolist()

    transformer = _NativeCategoryTransformer(categorical)

    model = XGBClassifier(
        random_state=seed,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="logloss",
        **params,
    )

    return Pipeline([
        ("preprocessor", transformer),
        ("classifier", model),
    ])


def _build_random_forest(
    features: pd.DataFrame,
    seed: int,
    params: dict[str, Any],
) -> Pipeline:
    """Build a random forest classifier."""

    preprocessor = _one_hot_preprocessor(
        features,
        scale_numeric=False,
    )

    model = RandomForestClassifier(
        random_state=seed,
        n_jobs=-1,
        **params,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def _build_logistic_regression(
    features: pd.DataFrame,
    seed: int,
    params: dict[str, Any],
) -> Pipeline:
    """Build a logistic regression classifier."""

    preprocessor = _one_hot_preprocessor(
        features,
        scale_numeric=True,
    )

    model = LogisticRegression(
        random_state=seed,
        solver="lbfgs",
        max_iter=1000,
        **params,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def _build_tabpfn(
    features: pd.DataFrame,
    seed: int,
):
    """Build a TabPFN classifier using its default configuration."""

    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    categorical_indices = [
        index
        for index, column in enumerate(features.columns)
        if not pd.api.types.is_numeric_dtype(features[column])
    ]

    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V3,
        categorical_features_indices=categorical_indices,
        random_state=seed,
    )


def _one_hot_preprocessor(
    features: pd.DataFrame,
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build preprocessing for models using one-hot encoding."""

    numeric = features.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical = features.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numeric_transformer = (
        StandardScaler()
        if scale_numeric
        else "passthrough"
    )

    return ColumnTransformer([
        ("numeric", numeric_transformer, numeric),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical,
        ),
    ])


class _NativeCategoryTransformer(
    TransformerMixin,
    BaseEstimator,
):
    """Convert categorical columns to pandas category dtype."""

    def __init__(
        self,
        categorical_columns: list[str],
    ):
        self.categorical_columns = categorical_columns
        self.categories_: dict[str, list[object]] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        """Learn categorical levels from the training data."""

        self.categories_ = {
            column: X[column].dropna().unique().tolist()
            for column in self.categorical_columns
        }

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert categorical columns using learned levels."""

        X = X.copy()

        for column, categories in self.categories_.items():
            X[column] = pd.Categorical(
                X[column],
                categories=categories,
            )

        return X