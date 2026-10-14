"""Classical methods for balancing imbalanced training data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from ...preparation.metadata import DatasetMetadata


DEFAULT_OUTPUT_ROOT = Path("artifacts/generation")


def balance_data(
    train: pd.DataFrame,
    *,
    test: pd.DataFrame,
    metadata: DatasetMetadata,
    method: str,
    fold_id: int,
    random_seed: int,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate the rows to be added (ROS, SMOTENC) or retained (RUS) for one fold.

    ROS / SMOTENC: the new minority rows that bring the fold to 1:1.
    RUS:           the majority rows kept, as many as the minority rows.

    `test` is used only to count exact copies after resampling (SMOTENC).
    """

    target_column = metadata.target_column
    X = train.drop(columns=[target_column])
    y = train[target_column]

    counts = y.value_counts()
    majority_class, minority_class = counts.idxmax(), counts.idxmin()

    if method == "ros":
        sampler = RandomOverSampler(random_state=random_seed)

    elif method == "smotenc":
        categorical = [c for c in metadata.categorical_columns if c in X.columns]

        # SMOTENC requires at least one categorical feature;
        # for all-numeric datasets it is equivalent to SMOTE.
        sampler = (
            SMOTENC(categorical_features=categorical, random_state=random_seed)
            if categorical
            else SMOTE(random_state=random_seed)
        )
        output_method = "smotenc" if categorical else "smote"

    elif method == "rus":
        sampler = RandomUnderSampler(random_state=random_seed)
        output_method = "rus"

    else:
        raise ValueError(f"Unknown balancing method: {method}")

    if method == "ros":
        output_method = "ros"

    output_path = (
        Path(output_root)
        / metadata.dataset_id
        / "baselines"
        / output_method
        / f"fold_{fold_id}.csv"
    )

    # Checked before resampling so no work is wasted.
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    resampled = X_resampled.copy()
    resampled[target_column] = y_resampled
    resampled = resampled[train.columns]  # same column order as train

    # Note: imblearn casts resampled columns back to the original dtypes, so
    # SMOTE/SMOTENC values in integer columns are truncated (60.9 -> 60).

    if method == "rus":
        output = resampled[resampled[target_column] == majority_class]
        expected_rows = counts[minority_class]
        expected_label = majority_class
    else:
        # The resampler returns the original rows first, then the new ones.
        output = resampled.iloc[len(train):]
        expected_rows = counts[majority_class] - counts[minority_class]
        expected_label = minority_class

    output = output.reset_index(drop=True)

    check_artifact(
        output,
        train,
        target_column=target_column,
        expected_rows=expected_rows,
        expected_label=expected_label,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    # Copies only make sense for SMOTENC: ROS rows are copies by definition,
    # and RUS keeps real rows instead of creating new ones.
    creates_rows = method == "smotenc"

    stats = pd.DataFrame(
        [
            {
                "method": method,
                "random_seed": random_seed,
                "rows": len(output),
                "copies_train": count_exact_copies(output, train) if creates_rows else None,
                "copies_test": count_exact_copies(output, test) if creates_rows else None,
            }
        ]
    )

    return output, stats


def check_artifact(
    artifact: pd.DataFrame,
    train: pd.DataFrame,
    *,
    target_column: str,
    expected_rows: int,
    expected_label,
) -> None:
    """
    Light structural checks for a generated artifact.

    Does not judge plausibility or quality, only whether the file can be
    safely concatenated to the training fold.
    """

    numeric = artifact.select_dtypes("number").to_numpy(dtype=float, na_value=np.nan)

    # Numeric in one and not in the other (e.g. a category read back from CSV
    # as numbers). Exact dtypes are not compared: int64 vs float64 is fine.
    kind_mismatch = [
        column
        for column in train.columns.intersection(artifact.columns)
        if is_numeric_dtype(train[column]) != is_numeric_dtype(artifact[column])
    ]

    checks = {
        "columns differ from the training fold": list(artifact.columns) != list(train.columns),
        f"numeric/non-numeric mismatch in {kind_mismatch}": bool(kind_mismatch),
        f"{len(artifact)} rows, expected {expected_rows}": len(artifact) != expected_rows,
        f"labels other than {expected_label!r}": not artifact[target_column].eq(expected_label).all(),
        "missing values": artifact.isna().any().any(),
        "infinite values": np.isinf(numeric).any(),
    }

    failed = [message for message, is_bad in checks.items() if is_bad]
    if failed:
        raise ValueError("Invalid artifact: " + "; ".join(failed))


def count_exact_copies(synthetic: pd.DataFrame, reference: pd.DataFrame) -> int:
    """
    Count synthetic rows identical to at least one row of `reference`.

    All columns of `reference` are compared, including the target. Numbers are
    compared as floats rounded to 6 decimals, so 45 and 45.0 count as equal.
    """

    def normalized_rows(frame: pd.DataFrame):
        normalized = frame[reference.columns].copy()
        for column in normalized.columns:
            if is_numeric_dtype(normalized[column]):
                normalized[column] = normalized[column].astype(float).round(6)
            else:
                normalized[column] = normalized[column].astype(str)
        return normalized.itertuples(index=False, name=None)

    reference_rows = set(normalized_rows(reference))
    return sum(row in reference_rows for row in normalized_rows(synthetic))
