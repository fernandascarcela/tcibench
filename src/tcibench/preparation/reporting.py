"""Dataset reporting utilities used throughout TCI-Bench."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .metadata import DatasetMetadata
from .utils import target_counts, to_python_scalar


class DatasetReporter:
    """Build and save dataset preparation reports."""

    def __init__(
        self,
        metadata: DatasetMetadata,
        *,
        output_root: str | Path = "artifacts/dataset_reports",
    ) -> None:
        self.metadata = metadata
        self.output_root = Path(output_root)

    def profile(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Create a structural summary of a dataset."""

        missing_by_column = df.isna().sum()

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "duplicate_rows": int(df.duplicated().sum()),
            "total_missing": int(missing_by_column.sum()),
            "missing_by_column": {
                column: int(count)
                for column, count in missing_by_column.items()
                if count
            },
            "target_counts": target_counts(
                df,
                self.metadata.target_column,
            ),
        }

    def save(
        self,
        df: pd.DataFrame,
        *,
        cleaning_stats: dict[str, Any] | None = None,
        split_audit: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save metadata and preparation information as JSON.

        `df` must be the cleaned dataset returned by
        `DatasetPreprocessor.clean`.
        """

        report_dir = (
            self.output_root
            / self.metadata.dataset_id
        )
        report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "metadata": asdict(self.metadata),
            "dataset_summary": self.profile(df),
            "column_summary": self._column_summary(df),
            "cleaning": cleaning_stats,
            "split": split_audit,
        }

        report_path = report_dir / "report.json"

        report_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return report_path

    def _column_summary(
        self,
        df: pd.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        """Summarize observed values for each dataset column."""

        summary: dict[str, dict[str, Any]] = {}

        for column in df.columns:
            details: dict[str, Any] = {
                "dtype": str(df[column].dtype),
                "type": self.metadata.columns[column]["type"],
                "missing": int(df[column].isna().sum()),
                "unique": int(df[column].nunique(dropna=True)),
            }

            if self.metadata.columns[column]["type"] == "categorical":
                details["allowed_values"] = [
                    to_python_scalar(value)
                    for value in sorted(
                        df[column].dropna().unique(),
                        key=str,
                    )
                ]

            summary[column] = details

        return summary