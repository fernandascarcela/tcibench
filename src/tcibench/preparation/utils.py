"""Shared preparation utilities."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def target_counts(
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, int]:
    """Count observations of each target class."""

    counts = df[target_column].value_counts(
        dropna=False
    )

    return {
        str(value): int(count)
        for value, count in counts.items()
    }


def to_python_scalar(
    value: Any,
) -> Any:
    """Convert NumPy/Pandas scalar values to native Python values."""

    return (
        value.item()
        if hasattr(value, "item")
        else value
    )


def dataframe_fingerprint(
    df: pd.DataFrame,
) -> str:
    """Create a stable fingerprint for a DataFrame."""

    hasher = hashlib.sha256()

    hasher.update(
        "|".join(
            str(column)
            for column in df.columns
        ).encode("utf-8")
    )

    hasher.update(
        "|".join(
            str(dtype)
            for dtype in df.dtypes
        ).encode("utf-8")
    )

    hashed_rows = pd.util.hash_pandas_object(
        df,
        index=False,
    ).values

    hasher.update(
        hashed_rows.tobytes()
    )

    return hasher.hexdigest()