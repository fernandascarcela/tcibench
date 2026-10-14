"""Sampling utilities for LLM reference contexts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...preparation.metadata import DatasetMetadata
from ...preparation.utils import dataframe_fingerprint, to_python_scalar


DEFAULT_EXAMPLES_PER_CLASS = 10
DEFAULT_SAMPLING_ROOT = Path("artifacts/generation/sampling")


def build_sampling_plan(
    train: pd.DataFrame,
    metadata: DatasetMetadata,
    *,
    records_to_generate: int,
    records_per_batch: int,
    examples_per_class: int = DEFAULT_EXAMPLES_PER_CLASS,
    sampling_seed: int = 42,
) -> dict[str, Any]:
    """Build a reproducible sampling plan from the training data."""

    if records_to_generate < 1:
        raise ValueError("records_to_generate must be at least 1.")

    if records_per_batch < 1:
        raise ValueError("records_per_batch must be at least 1.")

    if examples_per_class < 1:
        raise ValueError("examples_per_class must be at least 1.")

    target_column = metadata.target_column

    groups = {
        to_python_scalar(target_value): [int(position) for position in positions]
        for target_value, positions in train.groupby(
            target_column, sort=True, dropna=False
        ).indices.items()
    }

    if len(groups) < 2:
        raise ValueError("At least two target classes are required.")

    for target_value, positions in groups.items():
        if len(positions) < examples_per_class:
            raise ValueError(
                f"Class {target_value!r} has only {len(positions)} training rows, "
                f"but {examples_per_class} examples per class were requested."
            )

    calls_required = math.ceil(records_to_generate / records_per_batch)

    class_blocks = {
        target_value: _sample_blocks(
            positions,
            blocks=calls_required,
            block_size=examples_per_class,
            seed=sampling_seed,
            group_position=group_position,
        )
        for group_position, (target_value, positions) in enumerate(groups.items())
    }

    contexts = [
        {
            "call": call_index + 1,
            "positions_by_class": {
                str(target_value): blocks[call_index]
                for target_value, blocks in class_blocks.items()
            },
        }
        for call_index in range(calls_required)
    ]

    return {
        "dataset_id": metadata.dataset_id,
        "train_fingerprint": dataframe_fingerprint(train),
        "sampling_seed": sampling_seed,
        "target_column": target_column,
        "examples_per_class": examples_per_class,
        "context_size": examples_per_class * len(groups),
        "records_to_generate": records_to_generate,
        "records_per_batch": records_per_batch,
        "calls_required": calls_required,
        "contexts": contexts,
    }


def save_sampling_plan(
    sampling_plan: dict[str, Any],
    *,
    fold_id: int,
    output_root: str | Path = DEFAULT_SAMPLING_ROOT,
) -> Path:
    """Save a fold-specific sampling plan without overwriting a different one."""

    path = _sampling_plan_path(
        sampling_plan["dataset_id"],
        fold_id,
        sampling_plan["sampling_seed"],
        output_root,
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        existing_plan = json.loads(path.read_text(encoding="utf-8"))

        if existing_plan != sampling_plan:
            raise ValueError(f"A different sampling plan already exists at: {path}")

        return path

    path.write_text(
        json.dumps(sampling_plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return path


def load_sampling_plan(
    train: pd.DataFrame,
    metadata: DatasetMetadata,
    *,
    records_to_generate: int,
    records_per_batch: int,
    fold_id: int,
    examples_per_class: int = DEFAULT_EXAMPLES_PER_CLASS,
    sampling_seed: int = 42,
    output_root: str | Path = DEFAULT_SAMPLING_ROOT,
) -> dict[str, Any]:
    """Load the sampling plan for a fold or build and save it if it does not exist."""

    path = _sampling_plan_path(
        metadata.dataset_id,
        fold_id,
        sampling_seed,
        output_root,
    )

    if path.is_file():
        sampling_plan = json.loads(path.read_text(encoding="utf-8"))

        _validate_sampling_plan(
            sampling_plan,
            train,
            metadata,
            records_to_generate=records_to_generate,
            records_per_batch=records_per_batch,
            examples_per_class=examples_per_class,
            sampling_seed=sampling_seed,
        )

        return sampling_plan

    sampling_plan = build_sampling_plan(
        train,
        metadata,
        records_to_generate=records_to_generate,
        records_per_batch=records_per_batch,
        examples_per_class=examples_per_class,
        sampling_seed=sampling_seed,
    )

    save_sampling_plan(
        sampling_plan,
        fold_id=fold_id,
        output_root=output_root,
    )

    return sampling_plan


def get_context(
    train: pd.DataFrame,
    sampling_plan: dict[str, Any],
    *,
    dataset_id: str,
    call_index: int,
) -> pd.DataFrame:
    """Return the reference context assigned to one planned call."""

    if sampling_plan["dataset_id"] != dataset_id:
        raise ValueError("Sampling plan does not match the dataset.")

    if sampling_plan["train_fingerprint"] != dataframe_fingerprint(train):
        raise ValueError("Sampling plan does not match the provided training data.")

    contexts = sampling_plan["contexts"]

    if not 0 <= call_index < len(contexts):
        raise IndexError(f"call_index must be between 0 and {len(contexts) - 1}.")

    positions_by_class = contexts[call_index]["positions_by_class"]

    positions = [
        position
        for class_positions in positions_by_class.values()
        for position in class_positions
    ]

    return train.iloc[positions].copy().reset_index(drop=True)


def _validate_sampling_plan(
    sampling_plan: dict[str, Any],
    train: pd.DataFrame,
    metadata: DatasetMetadata,
    *,
    records_to_generate: int,
    records_per_batch: int,
    examples_per_class: int,
    sampling_seed: int,
) -> None:
    """Validate a saved sampling plan against the current experiment."""

    target_column = metadata.target_column
    class_values = [
        to_python_scalar(value)
        for value in train[target_column].drop_duplicates()
    ]

    expected_calls = math.ceil(records_to_generate / records_per_batch)

    expected = {
        "dataset_id": metadata.dataset_id,
        "train_fingerprint": dataframe_fingerprint(train),
        "sampling_seed": sampling_seed,
        "target_column": target_column,
        "examples_per_class": examples_per_class,
        "context_size": examples_per_class * len(class_values),
        "records_to_generate": records_to_generate,
        "records_per_batch": records_per_batch,
        "calls_required": expected_calls,
    }

    for key, value in expected.items():
        if sampling_plan.get(key) != value:
            raise ValueError(
                f"Existing sampling plan does not match the current '{key}'."
            )

    contexts = sampling_plan.get("contexts", [])

    if len(contexts) != expected_calls:
        raise ValueError("Sampling plan has an invalid number of contexts.")

    expected_classes = {str(value) for value in class_values}

    for context in contexts:
        positions_by_class = context.get("positions_by_class", {})

        if set(positions_by_class) != expected_classes:
            raise ValueError("Sampling plan classes do not match the training data.")

        for positions in positions_by_class.values():
            if len(positions) != examples_per_class:
                raise ValueError(
                    "Sampling plan context has an invalid number of examples."
                )

            if len(set(positions)) != len(positions):
                raise ValueError(
                    "A context contains repeated rows within the same class."
                )

            if any(position < 0 or position >= len(train) for position in positions):
                raise ValueError(
                    "Sampling plan contains an invalid training-row position."
                )


def _sample_blocks(
    positions: list[int],
    *,
    blocks: int,
    block_size: int,
    seed: int,
    group_position: int,
) -> list[list[int]]:
    """
    Sample without replacement until an epoch is exhausted.

    Repetition is allowed only after all rows of the class have been used,
    and the same row never appears twice within one context.
    """

    sampled_blocks = []
    queue: list[int] = []
    epoch = 0

    for _ in range(blocks):
        selected: list[int] = []

        while len(selected) < block_size:
            if not queue:
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, group_position, epoch])
                )
                queue = [int(position) for position in rng.permutation(positions)]
                epoch += 1

            needed = block_size - len(selected)
            selected_set = set(selected)
            chosen = []
            remaining_queue = []

            for position in queue:
                if len(chosen) < needed and position not in selected_set:
                    chosen.append(position)
                    selected_set.add(position)
                else:
                    remaining_queue.append(position)

            selected.extend(chosen)
            queue = remaining_queue

        sampled_blocks.append(selected)

    return sampled_blocks


def _sampling_plan_path(
    dataset_id: str,
    fold_id: int,
    sampling_seed: int,
    output_root: str | Path,
) -> Path:
    """Return the artifact path for one fold-specific sampling plan."""

    return (
        Path(output_root)
        / dataset_id
        / f"sampling_fold_{fold_id}.json"
    )
