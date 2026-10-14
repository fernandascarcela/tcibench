"""JSON Schema utilities for structured synthetic data generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ...preparation.metadata import DatasetMetadata
from ...preparation.utils import dataframe_fingerprint, to_python_scalar


DEFAULT_RECORDS_PER_BATCH = 10
DEFAULT_SCHEMAS_ROOT = Path("artifacts/generation/schemas")


def build_schema(
    train: pd.DataFrame,
    metadata: DatasetMetadata,
    *,
    minority_class: Any,
    records_per_batch: int = DEFAULT_RECORDS_PER_BATCH,
) -> dict[str, Any]:
    """Build the structured-output schema from the training data."""

    properties = {
        column: _build_column_schema(
            train[column],
            metadata.columns[column]["type"],
        )
        for column in train.columns
    }

    target_column = metadata.target_column

    properties[target_column] = {
        "type": properties[target_column]["type"],
        "const": to_python_scalar(minority_class),
    }

    record_schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "items": record_schema,
                "minItems": records_per_batch,
                "maxItems": records_per_batch,
            },
        },
        "required": ["records"],
        "additionalProperties": False,
    }


def save_schema(
    schema: dict[str, Any],
    train: pd.DataFrame,
    metadata: DatasetMetadata,
    *,
    minority_class: Any,
    fold_id: int,
    records_per_batch: int = DEFAULT_RECORDS_PER_BATCH,
    output_root: str | Path = DEFAULT_SCHEMAS_ROOT,
) -> Path:
    """Save a fold-specific schema and the information used to build it."""

    path = _schema_path(
        metadata,
        fold_id,
        output_root,
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset_id": metadata.dataset_id,
        "fold_id": fold_id,
        "train_fingerprint": dataframe_fingerprint(train),
        "minority_class": to_python_scalar(minority_class),
        "records_per_batch": records_per_batch,
        "schema": schema,
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def load_schema(
    train: pd.DataFrame,
    metadata: DatasetMetadata,
    *,
    minority_class: Any,
    fold_id: int,
    records_per_batch: int = DEFAULT_RECORDS_PER_BATCH,
    output_root: str | Path = DEFAULT_SCHEMAS_ROOT,
) -> dict[str, Any]:
    """Load the schema for a fold or build it if it does not exist."""

    path = _schema_path(
        metadata,
        fold_id,
        output_root,
    )

    minority_class = to_python_scalar(minority_class)

    if path.is_file():
        payload = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )

        expected = {
            "dataset_id": metadata.dataset_id,
            "fold_id": fold_id,
            "train_fingerprint": dataframe_fingerprint(train),
            "minority_class": minority_class,
            "records_per_batch": records_per_batch,
        }

        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"Existing schema does not match the current '{key}': {path}"
                )

        return payload["schema"]

    schema = build_schema(
        train,
        metadata,
        minority_class=minority_class,
        records_per_batch=records_per_batch,
    )

    save_schema(
        schema,
        train,
        metadata,
        minority_class=minority_class,
        fold_id=fold_id,
        records_per_batch=records_per_batch,
        output_root=output_root,
    )

    return schema


def _build_column_schema(
    series: pd.Series,
    column_type: str,
) -> dict[str, Any]:
    """Build the schema for one dataset column."""

    schema: dict[str, Any] = {
        "type": _json_type(series.dtype),
    }

    if column_type == "categorical":
        schema["enum"] = [
            to_python_scalar(value)
            for value in sorted(
                series.dropna().unique(),
                key=str,
            )
        ]

    return schema


def _json_type(dtype: Any) -> str:
    """Convert a pandas dtype to a JSON Schema type."""

    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"

    if pd.api.types.is_integer_dtype(dtype):
        return "integer"

    if pd.api.types.is_numeric_dtype(dtype):
        return "number"

    return "string"


def _schema_path(
    metadata: DatasetMetadata,
    fold_id: int,
    output_root: str | Path,
) -> Path:
    """Return the schema artifact path for one dataset fold."""

    return (
        Path(output_root)
        / metadata.dataset_id
        / f"fold_{fold_id}.json"
    )