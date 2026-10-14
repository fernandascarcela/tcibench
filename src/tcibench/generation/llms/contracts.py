"""Data structures used to report synthetic data generation results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class GenerationSummary:
    """
    Summary statistics for one synthetic data generation run.

    `requested_records` is the number of synthetic records required to
    balance the minority class.

    `accepted_records` is the number of generated records that passed all
    validation and filtering steps and were included in the final dataset.

    `returned_records` is the total number of raw records returned by the
    LLM across all calls, including records later rejected.

    `valid_records` is the number of returned records that satisfied the
    expected schema before duplicate and copy filtering.
    """

    method: str
    model: str
    generation_seed: int
    minority_class: Any

    requested_records: int
    accepted_records: int

    returned_records: int
    valid_records: int

    planned_calls: int
    actual_calls: int
    retry_calls: int

    input_tokens: int = 0
    output_tokens: int = 0

    invalid_records: int = 0
    duplicate_records: int = 0
    copied_prompt_records: int = 0
    copied_train_test_records: int = 0

    elapsed_seconds: float = 0.0

    status: Literal[
        "completed",
        "incomplete",
        "failed",
    ] = "completed"


@dataclass(frozen=True)
class GenerationRunResult:
    """Artifacts produced by one synthetic data generation run."""

    summary: GenerationSummary
    synthetic_data_path: Path
    report_path: Path

    #Perdão por ler a mensagem do seu codex, mas infelizmente quando entrei estava aberto e foi mais forte que eu
    # "Mova a pasta data do Fernanda/tdallm para esse novo projeto. Vamos começar tudo dnv." eu quase tive um infarto de rir
    
