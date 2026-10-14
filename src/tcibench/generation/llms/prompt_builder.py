"""Build prompts for LLM-based tabular synthetic data generation."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ...preparation.metadata import DatasetMetadata
from ...preparation.utils import to_python_scalar
from .prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def build_generation_messages(
    context: pd.DataFrame,
    metadata: DatasetMetadata,
    *,
    minority_class: Any,
    records_per_batch: int,
    examples_per_class: int,
) -> list[dict[str, str]]:
    """Build messages for one synthetic data generation call."""

    user_prompt = USER_PROMPT_TEMPLATE.format(
        dataset_description=metadata.description,
        target_column=metadata.target_column,
        column_descriptions=_format_column_descriptions(
            metadata,
        ),
        examples=_format_grouped_examples(
            context,
            target_column=metadata.target_column,
        ),
        examples_per_class=examples_per_class,
        records_per_batch=records_per_batch,
        minority_class=to_python_scalar(
            minority_class,
        ),
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def _format_column_descriptions(
    metadata: DatasetMetadata,
) -> str:
    """Format column descriptions line by line."""

    return "\n".join(
        f"{column}: {description}"
        for column, description
        in metadata.column_descriptions.items()
    )


def _format_grouped_examples(
    context: pd.DataFrame,
    *,
    target_column: str,
) -> str:
    """Format reference examples grouped by target class."""

    groups = []

    for target_value, group in sorted(
        context.groupby(
            target_column,
            sort=False,
            dropna=False,
        ),
        key=lambda item: str(item[0]),
    ):
        columns = [
            target_column,
            *[
                column
                for column in group.columns
                if column != target_column
            ],
        ]

        records = _format_records(
            group[columns]
        )

        groups.append(
            f"CLASS {target_value}\n"
            f"{records}"
        )

    return "\n\n".join(groups)


def _format_records(
    frame: pd.DataFrame,
) -> str:
    """Format records as a JSON array with one record per line."""

    records = json.loads(
        frame.to_json(
            orient="records",
            force_ascii=False,
        )
    )

    formatted_records = [
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for record in records
    ]

    return (
        "["
        + "\n "
        + ",\n ".join(formatted_records)
        + "\n]"
    )