"""Dataset representation used throughout TCI-Bench."""

from __future__ import annotations

import pandas as pd

from .metadata import DatasetMetadata
from .preprocessing import DatasetPreprocessor


class Dataset:
    """Represent a tabular dataset and its associated operations."""

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: DatasetMetadata,
    ) -> None:
        self.data = data
        self.metadata = metadata

        self.preprocessing = DatasetPreprocessor(metadata)
