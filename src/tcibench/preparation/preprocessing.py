"""Dataset preprocessing utilities used throughout TCI-Bench."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..config import DEFAULT_EXPERIMENT_CONFIG
from .metadata import DatasetMetadata
from .utils import target_counts, to_python_scalar


DEFAULT_NUMERIC_DECIMALS = 4
DEFAULT_N_SPLITS = 5


class DatasetPreprocessor:
    """Clean datasets and create stratified cross-validation folds."""

    def __init__(
        self,
        metadata: DatasetMetadata,
        *,
        split_seed: int = DEFAULT_EXPERIMENT_CONFIG.split_seed,
        n_splits: int = DEFAULT_N_SPLITS,
        numeric_decimals: int = DEFAULT_NUMERIC_DECIMALS,
    ) -> None:
        self.metadata = metadata
        self.split_seed = split_seed
        self.n_splits = n_splits
        self.numeric_decimals = numeric_decimals

    def clean(
    self,
    df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Clean the dataset before cross-validation."""

        target = self.metadata.target_column

        if df[target].isna().all():
            raise ValueError(f"Target column '{target}' contains only missing values.")

        all_missing_columns = [
            column
            for column in df.columns
            if column != target and df[column].isna().all()
        ]

        columns_to_drop = list(
            dict.fromkeys([
                *self.metadata.columns_to_drop,
                *all_missing_columns,
            ])
        )

        missing_by_column = df.isna().sum()
        working = df.drop(columns=columns_to_drop).copy()

        missing_mask = working.isna().any(axis=1)
        discarded_missing = working.loc[missing_mask]
        cleaned = working.loc[~missing_mask].copy()

        numeric_columns = [
            column
            for column in self.metadata.numeric_columns
            if column in cleaned.columns
        ]

        if numeric_columns:
            cleaned[numeric_columns] = cleaned[numeric_columns].round(
                self.numeric_decimals
            )

        duplicate_mask = cleaned.duplicated(keep="first")
        discarded_duplicates = cleaned.loc[duplicate_mask]
        cleaned = cleaned.loc[~duplicate_mask].reset_index(drop=True)

        missing_counts = target_counts(discarded_missing, target)
        duplicate_counts = target_counts(discarded_duplicates, target)

        stats = {
            "original_rows": len(df),
            "cleaned_rows": len(cleaned),
            "dropped_columns": columns_to_drop,
            "all_missing_columns": all_missing_columns,
            "missing_values_before_cleaning": int(missing_by_column.sum()),
            "missing_values_by_column_before_cleaning": {
                column: int(count)
                for column, count in missing_by_column.items()
                if count
            },
            "removed_missing_rows": len(discarded_missing),
            "removed_duplicate_rows": len(discarded_duplicates),
            "numeric_decimals": self.numeric_decimals,
            "target_counts_before_cleaning": target_counts(working, target),
            "target_counts_after_cleaning": target_counts(cleaned, target),
            "discarded_missing_by_target": missing_counts,
            "discarded_duplicate_by_target": duplicate_counts,
            "discarded_total_by_target": self._sum_counts(
                missing_counts,
                duplicate_counts,
            ),
        }

        return cleaned, stats

    def split(
        self,
        df: pd.DataFrame,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
        """Create stratified train/test pairs for each cross-validation fold."""

        target = self.metadata.target_column

        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.split_seed,
        )

        folds = []

        for fold_id, (train_idx, test_idx) in enumerate(
            splitter.split(df, df[target]),
            start=1,
        ):
            train = df.iloc[train_idx].copy().reset_index(drop=True)
            test = df.iloc[test_idx].copy().reset_index(drop=True)

            audit = {
                "fold": fold_id,
                "n_splits": self.n_splits,
                "split_seed": self.split_seed,
                "input_rows": len(df),
                "train_rows": len(train),
                "test_rows": len(test),
                "train_target_counts": target_counts(
                    train,
                    target,
                ),
                "test_target_counts": target_counts(
                    test,
                    target,
                ),
                **self._imbalance_summary(train),
            }

            folds.append((train, test, audit))

        return folds

    def _imbalance_summary(
        self,
        train: pd.DataFrame,
    ) -> dict[str, Any]:
        """Describe class imbalance in the training fold."""

        counts = train[
            self.metadata.target_column
        ].value_counts()

        majority_class = counts.idxmax()
        minority_class = counts.idxmin()

        majority_count = int(counts.loc[majority_class])
        minority_count = int(counts.loc[minority_class])

        return {
            "majority_class": to_python_scalar(majority_class),
            "minority_class": to_python_scalar(minority_class),
            "majority_count": majority_count,
            "minority_count": minority_count,
            "imbalance_ratio": majority_count / minority_count,
            "samples_to_balance": majority_count - minority_count,
        }

    @staticmethod
    def _sum_counts(
        *count_maps: dict[str, int],
    ) -> dict[str, int]:
        """Combine target-count dictionaries."""

        result: dict[str, int] = {}

        for count_map in count_maps:
            for value, count in count_map.items():
                result[value] = result.get(value, 0) + count

        return result