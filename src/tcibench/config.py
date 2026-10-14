"""Default settings for reproducible dataset preparation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    split_seed: int = 42
    test_size: float = 0.2


DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()
