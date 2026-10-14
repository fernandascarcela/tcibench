"""Orchestration for LLM-based synthetic tabular data generation."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any

import pandas as pd

from ...preparation.metadata import DatasetMetadata
from ...preparation.utils import dataframe_fingerprint, to_python_scalar
from .client import LLMClient
from .contracts import GenerationRunResult, GenerationSummary
from .prompt_builder import build_generation_messages
from .sampling import get_context


DEFAULT_MAX_ATTEMPTS_PER_CONTEXT = 10

# artifacts/synthetic/<dataset>/<model>/
#     fold_0.csv ... fold_4.csv
#     report.json   (one report with every fold of this model)
DEFAULT_OUTPUT_ROOT = Path("artifacts/synthetic")
REPORT_FILENAME = "report.json"


def run_generation(
    train: pd.DataFrame,
    metadata: DatasetMetadata,
    schema: dict[str, Any],
    sampling_plan: dict[str, Any],
    client: LLMClient,
    *,
    test: pd.DataFrame,
    fold_id: int,
    generation_seed: int,
    max_attempts_per_context: int = DEFAULT_MAX_ATTEMPTS_PER_CONTEXT,
    max_concurrent_calls: int = 1,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    overwrite: bool = False,
) -> GenerationRunResult:
    """
    Generate minority-class records until the training set of one fold is balanced.

    For this generator, the method is the LLM itself: the model name defines
    the output folder.

    `max_concurrent_calls` sends the first attempt of up to that many contexts
    at the same time. Responses are still processed in context order, and
    retries happen at the same point as in a sequential run, so the accepted
    records do not depend on the concurrency level.
    """

    if max_attempts_per_context < 1:
        raise ValueError("max_attempts_per_context must be at least 1.")

    if max_concurrent_calls < 1:
        raise ValueError("max_concurrent_calls must be at least 1.")

    if fold_id < 0:
        raise ValueError("fold_id must be non-negative.")

    model = client.config.model

    # Resolve the output paths first, so an existing fold is detected
    # before any LLM call is made.
    model_directory = _run_directory(metadata, model=model, output_root=output_root)
    synthetic_data_path = model_directory / f"fold_{fold_id}.csv"
    report_path = model_directory / REPORT_FILENAME

    existing_report = _load_report(report_path)
    fold_key = str(fold_id)

    if (
        synthetic_data_path.exists() or fold_key in existing_report["folds"]
    ) and not overwrite:
        raise FileExistsError(
            f"Fold {fold_id} already exists in {model_directory}. "
            "Use overwrite=True to replace it, or another output_root "
            "for generation replicates."
        )

    minority_class, records_to_generate = _generation_target(
        train, metadata.target_column
    )

    _validate_sampling_plan(
        sampling_plan,
        train,
        metadata,
        records_to_generate,
    )

    records_per_batch = sampling_plan["records_per_batch"]
    examples_per_class = sampling_plan["examples_per_class"]
    planned_calls = sampling_plan["calls_required"]

    _validate_schema(
        schema,
        metadata.target_column,
        minority_class,
        records_per_batch,
    )

    print(
        f"\nDataset: {metadata.dataset_id} | "
        f"Model: {model} | "
        f"Fold: {fold_id} | "
        f"Minority class: {minority_class} | "
        f"Target records: {records_to_generate} | "
        f"Planned contexts: {planned_calls}\n"
    )

    columns = list(train.columns)
    train_keys = _frame_keys(train, columns)
    test_keys = _frame_keys(test, columns)

    accepted_records: list[dict[str, Any]] = []
    accepted_keys: set[tuple[Any, ...]] = set()

    returned_records = 0
    valid_records = 0
    invalid_records = 0
    duplicate_records = 0
    copied_prompt_records = 0
    copied_train_test_records = 0

    actual_calls = 0
    retry_calls = 0
    successful_calls = 0

    input_tokens = 0
    output_tokens = 0

    errors: list[dict[str, Any]] = []

    started_at = perf_counter()

    def prepare(call_index: int) -> tuple[set[tuple[Any, ...]], list[dict[str, Any]]]:
        context = get_context(
            train,
            sampling_plan,
            dataset_id=metadata.dataset_id,
            call_index=call_index,
        )

        messages = build_generation_messages(
            context,
            metadata,
            minority_class=minority_class,
            records_per_batch=records_per_batch,
            examples_per_class=examples_per_class,
        )

        return _frame_keys(context, columns), messages

    prepared = [prepare(call_index) for call_index in range(planned_calls)]

    with ThreadPoolExecutor(max_workers=max_concurrent_calls) as pool:
        # First attempts run ahead in a window of `max_concurrent_calls` contexts.
        first_attempts: dict[int, Future] = {}

        def submit_first_attempts(start_index: int) -> None:
            for index in range(start_index, min(start_index + max_concurrent_calls, planned_calls)):
                if index not in first_attempts:
                    first_attempts[index] = pool.submit(
                        _call_llm, client, prepared[index][1], schema, generation_seed,
                    )

        for call_index in range(planned_calls):
            if len(accepted_records) >= records_to_generate:
                break

            submit_first_attempts(call_index)

            context_keys, messages = prepared[call_index]

            records_needed = min(
                records_per_batch,
                records_to_generate - len(accepted_records),
            )

            accepted_before_context = len(accepted_records)

            for attempt in range(max_attempts_per_context):
                accepted_in_context = len(accepted_records) - accepted_before_context

                if accepted_in_context >= records_needed:
                    break

                if attempt > 0:
                    retry_calls += 1

                actual_calls += 1

                invalid_before = invalid_records
                duplicate_before = duplicate_records
                copied_prompt_before = copied_prompt_records
                copied_train_test_before = copied_train_test_records
                accepted_before_attempt = len(accepted_records)

                # Retries are decided here, in context order, exactly as in a
                # sequential run, and sent immediately.
                if attempt == 0:
                    response, exc = first_attempts.pop(call_index).result()
                else:
                    response, exc = _call_llm(client, messages, schema, generation_seed)

                if exc is not None:
                    errors.append(
                        {
                            "call": call_index + 1,
                            "attempt": attempt + 1,
                            "error": str(exc),
                        }
                    )

                    print(
                        f"Context {call_index + 1}/{planned_calls} | "
                        f"Attempt {attempt + 1} | Error: {exc}\n"
                    )

                    continue

                successful_calls += 1

                input_tokens += response.input_tokens or 0
                output_tokens += response.output_tokens or 0
                returned_records += len(response.records)

                returned_valid_groups: dict[Any, int] = {}

                for record in response.records:
                    if not _is_valid_record(record, schema):
                        invalid_records += 1
                        continue

                    valid_records += 1

                    target_value = to_python_scalar(record[metadata.target_column])
                    returned_valid_groups[target_value] = (
                        returned_valid_groups.get(target_value, 0) + 1
                    )

                    accepted_in_context = len(accepted_records) - accepted_before_context

                    if accepted_in_context >= records_needed:
                        continue

                    normalized_record = _normalize_record(record, columns)
                    record_key = _record_key(normalized_record, columns)

                    if record_key in context_keys:
                        copied_prompt_records += 1
                        continue

                    if record_key in train_keys or record_key in test_keys:
                        copied_train_test_records += 1
                        continue

                    if record_key in accepted_keys:
                        duplicate_records += 1
                        continue

                    accepted_records.append(normalized_record)
                    accepted_keys.add(record_key)

                returned_now = len(response.records)
                valid_now = returned_now - (invalid_records - invalid_before)
                accepted_now = len(accepted_records) - accepted_before_attempt

                invalid_now = invalid_records - invalid_before
                duplicate_now = duplicate_records - duplicate_before
                copied_prompt_now = copied_prompt_records - copied_prompt_before
                copied_train_test_now = (
                    copied_train_test_records - copied_train_test_before
                )

                accepted_in_context = len(accepted_records) - accepted_before_context

                print(
                    f"Context {call_index + 1}/{planned_calls} | "
                    f"Attempt {attempt + 1} | "
                    f"Returned: {returned_now} | "
                    f"Valid: {valid_now} | "
                    f"Accepted: {accepted_now} | "
                    f"Block: {accepted_in_context}/{records_needed} | "
                    f"Total: {len(accepted_records)}/{records_to_generate}"
                )

                print(
                    f"Requested class: "
                    f"{metadata.target_column}={minority_class}"
                )

                print(f"Returned valid groups: {returned_valid_groups}")

                print(
                    f"Discarded -> "
                    f"Format/schema: {invalid_now} | "
                    f"Duplicate: {duplicate_now} | "
                    f"Copied prompt: {copied_prompt_now} | "
                    f"Copied train/test: {copied_train_test_now}"
                )

                print(
                    f"Input tokens: {response.input_tokens or 0} | "
                    f"Output tokens: {response.output_tokens or 0}\n"
                )

        for future in first_attempts.values():
            future.cancel()

    elapsed_seconds = perf_counter() - started_at

    synthetic = pd.DataFrame(accepted_records, columns=columns)

    if not synthetic.empty:
        synthetic = synthetic.astype(
            {column: train[column].dtype for column in columns}
        )

    if len(accepted_records) == records_to_generate:
        status = "completed"
    elif successful_calls == 0:
        status = "failed"
    else:
        status = "incomplete"

    # For this generator, method = LLM model.
    summary = GenerationSummary(
        method=model,
        model=model,
        generation_seed=generation_seed,
        minority_class=minority_class,
        requested_records=records_to_generate,
        accepted_records=len(accepted_records),
        returned_records=returned_records,
        valid_records=valid_records,
        planned_calls=planned_calls,
        actual_calls=actual_calls,
        retry_calls=retry_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        invalid_records=invalid_records,
        duplicate_records=duplicate_records,
        copied_prompt_records=copied_prompt_records,
        copied_train_test_records=copied_train_test_records,
        elapsed_seconds=elapsed_seconds,
        status=status,
    )

    # Created even for incomplete runs, so their report can still be written.
    model_directory.mkdir(parents=True, exist_ok=True)

    if status == "completed":
        synthetic.to_csv(synthetic_data_path, index=False)

    fold_report: dict[str, Any] = {
        "synthetic_data_file": (
            synthetic_data_path.name
            if status == "completed"
            else None
        ),
        "train_fingerprint": dataframe_fingerprint(train),
        "sampling_seed": sampling_plan["sampling_seed"],
        "examples_per_class": examples_per_class,
        "records_per_batch": records_per_batch,
        "max_attempts_per_context": max_attempts_per_context,
        "max_concurrent_calls": max_concurrent_calls,
        "summary": asdict(summary),
    }

    if errors:
        fold_report["errors"] = errors

    # Re-read just before writing, so the report reflects the file on disk.
    report = _load_report(report_path)
    report["dataset_id"] = metadata.dataset_id
    report["model"] = model
    report["folds"][fold_key] = fold_report
    report["folds"] = dict(sorted(report["folds"].items(), key=lambda item: int(item[0])))
    report["totals"] = _report_totals(report["folds"])

    _write_json_atomic(report_path, report)

    return GenerationRunResult(
        summary=summary,
        synthetic_data_path=synthetic_data_path,
        report_path=report_path,
    )


def _call_llm(
    client: LLMClient,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    seed: int,
) -> tuple[Any, Exception | None]:
    """One LLM call. Errors are returned instead of raised, so they are handled in order."""

    try:
        return client.generate(messages, schema, seed=seed), None
    except Exception as exc:  # noqa: BLE001 - every API error counts as a failed attempt
        return None, exc


def _generation_target(
    train: pd.DataFrame,
    target_column: str,
) -> tuple[Any, int]:
    """Identify the minority class and the required number of records."""

    counts = train[target_column].value_counts(dropna=False)

    if len(counts) != 2:
        raise ValueError("Generation requires a binary target.")

    majority_class = counts.idxmax()
    minority_class = counts.idxmin()

    majority_count = int(counts[majority_class])
    minority_count = int(counts[minority_class])
    records_to_generate = majority_count - minority_count

    if records_to_generate <= 0:
        raise ValueError("The training set is already balanced.")

    return to_python_scalar(minority_class), records_to_generate


def _validate_sampling_plan(
    sampling_plan: dict[str, Any],
    train: pd.DataFrame,
    metadata: DatasetMetadata,
    records_to_generate: int,
) -> None:
    """Check that the sampling plan belongs to the current training data."""

    expected = {
        "dataset_id": metadata.dataset_id,
        "train_fingerprint": dataframe_fingerprint(train),
        "target_column": metadata.target_column,
        "records_to_generate": records_to_generate,
    }

    for key, value in expected.items():
        if sampling_plan.get(key) != value:
            raise ValueError(
                f"Sampling plan does not match the current '{key}'."
            )


def _validate_schema(
    schema: dict[str, Any],
    target_column: str,
    minority_class: Any,
    records_per_batch: int,
) -> None:
    """Check that the schema matches the generation target."""

    try:
        records_schema = schema["properties"]["records"]
        target_schema = records_schema["items"]["properties"][target_column]

    except KeyError as exc:
        raise ValueError("Invalid generation schema.") from exc

    if target_schema.get("const") != minority_class:
        raise ValueError("Schema target does not match the minority class.")

    if (
        records_schema.get("minItems") != records_per_batch
        or records_schema.get("maxItems") != records_per_batch
    ):
        raise ValueError(
            "Schema batch size does not match the sampling plan."
        )


def _is_valid_record(
    record: Any,
    schema: dict[str, Any],
) -> bool:
    """Validate one generated record against the flat record schema."""

    if not isinstance(record, dict):
        return False

    record_schema = schema["properties"]["records"]["items"]
    properties = record_schema["properties"]

    if set(record) != set(properties):
        return False

    for column, rules in properties.items():
        value = record[column]

        if not _matches_json_type(value, rules["type"]):
            return False

        if "const" in rules and value != rules["const"]:
            return False

        if "enum" in rules and value not in rules["enum"]:
            return False

    return True


def _matches_json_type(
    value: Any,
    json_type: str,
) -> bool:
    """Check whether a Python value matches a JSON primitive type."""

    if json_type == "boolean":
        return isinstance(value, bool)

    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)

    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if json_type == "string":
        return isinstance(value, str)

    return False


def _normalize_record(
    record: dict[str, Any],
    columns: list[str],
) -> dict[str, Any]:
    """Return a record in the training-column order."""

    return {
        column: to_python_scalar(record[column])
        for column in columns
    }


def _record_key(
    record: dict[str, Any],
    columns: list[str],
) -> tuple[Any, ...]:
    """Create a hashable representation of one record."""

    return tuple(
        to_python_scalar(record[column])
        for column in columns
    )


def _frame_keys(
    frame: pd.DataFrame,
    columns: list[str],
) -> set[tuple[Any, ...]]:
    """Create record keys for all rows in a DataFrame."""

    return {
        tuple(to_python_scalar(value) for value in row)
        for row in frame[columns].itertuples(index=False, name=None)
    }


def _load_report(report_path: Path) -> dict[str, Any]:
    """Load the model report, or return an empty one."""

    if not report_path.is_file():
        return {"folds": {}}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.setdefault("folds", {})
    return report


def _report_totals(folds: dict[str, Any]) -> dict[str, Any]:
    """Aggregate cost and outcome indicators over all folds of a model."""

    summaries = [fold["summary"] for fold in folds.values()]

    summed_fields = (
        "requested_records",
        "accepted_records",
        "returned_records",
        "valid_records",
        "invalid_records",
        "duplicate_records",
        "copied_prompt_records",
        "copied_train_test_records",
        "planned_calls",
        "actual_calls",
        "retry_calls",
        "input_tokens",
        "output_tokens",
        "elapsed_seconds",
    )

    totals: dict[str, Any] = {
        field: sum(summary[field] for summary in summaries)
        for field in summed_fields
    }

    totals["folds_completed"] = sorted(
        int(fold_id)
        for fold_id, fold in folds.items()
        if fold["summary"]["status"] == "completed"
    )
    totals["folds_not_completed"] = sorted(
        int(fold_id)
        for fold_id, fold in folds.items()
        if fold["summary"]["status"] != "completed"
    )

    return totals


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temporary file, so a crash never truncates the report."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_directory(
    metadata: DatasetMetadata,
    *,
    model: str,
    output_root: str | Path,
) -> Path:
    """Return <output_root>/<dataset>/<model>; the fold is the file name."""

    return (
        Path(output_root)
        / metadata.dataset_id
        / _safe_name(model)
    )


def _safe_name(value: str) -> str:
    """Convert an identifier into a filesystem-safe name."""

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")