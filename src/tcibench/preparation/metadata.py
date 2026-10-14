"""Dataset metadata definitions used throughout TCI-Bench."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass(frozen=True)
class DatasetMetadata:
    """Store dataset-specific metadata used throughout the benchmark."""

    dataset_id: str
    display_name: str
    target_column: str
    description: str
    columns: dict[str, dict[str, str]]

    source_url: str | None = None
    columns_to_drop: list[str] = field(default_factory=list)
    train_limit: int | None = None

    @property
    def feature_columns(self) -> list[str]:
        """Return predictor columns."""

        return [
            column
            for column in self.columns
            if column != self.target_column
        ]

    @property
    def numeric_columns(self) -> list[str]:
        """Return numeric predictor columns."""

        return [
            column
            for column in self.feature_columns
            if self.columns[column]["type"] == "numeric"
        ]

    @property
    def categorical_columns(self) -> list[str]:
        """Return categorical predictor columns."""

        return [
            column
            for column in self.feature_columns
            if self.columns[column]["type"] == "categorical"
        ]

    @property
    def column_descriptions(self) -> dict[str, str]:
        """Return column descriptions."""

        return {
            column: info["description"]
            for column, info in self.columns.items()
        }


def load_metadata(dataset_id: str) -> DatasetMetadata:
    """Load one dataset definition from the bundled metadata file."""

    definitions = json.loads(
        Path(__file__).with_suffix(".json").read_text(encoding="utf-8")
    )
    if dataset_id not in definitions:
        raise ValueError(f"Unknown dataset: {dataset_id}")
    return DatasetMetadata(dataset_id=dataset_id, **definitions[dataset_id])
