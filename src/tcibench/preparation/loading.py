"""Utilities for loading original tabular datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.io import arff


def load_dataset(
    path: str | Path,
    *,
    csv_encoding: str | None = None,
    column_names: list[str] | None = None,
    na_values: str | list[str] | None = None,
    skipinitialspace: bool = False,
) -> pd.DataFrame:
    """Load a tabular dataset without applying preprocessing."""

    file_path = _require_file(path)
    suffix = file_path.suffix.lower()

    if suffix == ".arff":
        return _load_arff(file_path, encoding=csv_encoding)

    if suffix == ".csv":
        return pd.read_csv(
            file_path,
            encoding=csv_encoding,
            na_values=na_values,
            skipinitialspace=skipinitialspace,
        )

    if suffix == ".data":
        return pd.read_csv(
            file_path,
            header=None,
            names=column_names,
            encoding=csv_encoding,
            na_values=na_values,
            skipinitialspace=skipinitialspace,
        )

    if suffix == ".txt":
        return pd.read_csv(
            file_path,
            sep=None,
            engine="python",
            encoding=csv_encoding,
            na_values=na_values,
            skipinitialspace=skipinitialspace,
        )

    if suffix == ".tsv":
        return pd.read_csv(
            file_path,
            sep="\t",
            encoding=csv_encoding,
            na_values=na_values,
            skipinitialspace=skipinitialspace,
        )

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    if suffix == ".json":
        return pd.read_json(file_path)

    if suffix == ".parquet":
        return pd.read_parquet(file_path)

    raise ValueError(
        "Unsupported file format. "
        "Use ARFF, CSV, DATA, TXT, TSV, Excel, JSON, or Parquet."
    )


def _load_arff(
    file_path: Path,
    *,
    encoding: str | None = None,
) -> pd.DataFrame:
    """Load an ARFF file and convert its values to a pandas DataFrame."""

    data, _ = arff.loadarff(file_path)

    df = pd.DataFrame(data)

    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].map(
            lambda value: (
                value.decode(encoding or "utf-8")
                if isinstance(value, bytes)
                else value
            )
        )

    return df.replace("?", pd.NA)


def _require_file(
    path: str | Path,
) -> Path:
    """Return the dataset path if it exists."""

    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    return file_path