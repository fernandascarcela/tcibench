"""Synthetic data generation with generative tabular baselines."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

from ...preparation.metadata import DatasetMetadata
from .classical_baselines import check_artifact, count_exact_copies


DEFAULT_OVERSAMPLE_RATIO = 0.10
DEFAULT_MAX_GENERATION_ATTEMPTS = 10
DEFAULT_OUTPUT_ROOT = Path("artifacts/generation")


def generate_with_synthcity(
    train: pd.DataFrame,
    *,
    test: pd.DataFrame,
    metadata: DatasetMetadata,
    model_name: str,
    n_samples: int,
    random_seed: int,
    target_column: str,
    minority_class: Any,
    fold_id: int,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    overwrite: bool = False,
    model_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate minority-class samples and save them for one fold.

    `test` is used only to count exact copies after generation; it never
    reaches the model.
    """

    if minority_class not in train[target_column].unique():
        raise ValueError(
            f"Minority class {minority_class!r} is not present in the training data."
        )

    output_path = (
        Path(output_root)
        / metadata.dataset_id
        / "baselines"
        / model_name
        / f"fold_{fold_id}.csv"
    )

    # Checked before fitting: training CTGAN/TVAE/TabDDPM is the expensive part.
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    model_kwargs = dict(model_kwargs or {})

    target_values = sorted(
        train[target_column].unique(),
        key=str,
    )

    condition_map = {
        value: i
        for i, value in enumerate(target_values)
    }
    reverse_map = {code: value for value, code in condition_map.items()}

    train_cond = train[target_column].map(condition_map).to_numpy()
    minority_condition = condition_map[minority_class]

    if model_name == "ddpm":
        model_kwargs["is_classification"] = True

    model = Plugins().get(
        model_name,
        random_state=random_seed,
        **model_kwargs,
    )

    if model_name == "ddpm":
        fit_data = train.copy()
        fit_data[target_column] = fit_data[target_column].map(condition_map)

        loader = GenericDataLoader(
            fit_data,
            target_column=target_column,
            random_state=random_seed,
        )

        model.fit(loader)

    else:
        model.fit(
            train,
            cond=train_cond,
        )

    requested = ceil(
        n_samples * (1 + DEFAULT_OVERSAMPLE_RATIO)
    )

    attempts = 0
    total_generated = 0
    total_wrong_class = 0
    total_duplicates = 0

    pool = None
    

    for attempt in range(DEFAULT_MAX_GENERATION_ATTEMPTS):
        attempts = attempt + 1

        batch = model.generate(
            count=requested,
            cond=np.repeat(minority_condition, requested),
            random_state=random_seed + attempt,
        ).dataframe()

        total_generated += len(batch)

        if model_name == "ddpm":
            # Null, non-numeric or out-of-range codes become NaN and are then
            # discarded by the minority-class filter (counted as wrong_class).
            batch[target_column] = (
                pd.to_numeric(batch[target_column], errors="coerce")
                .round()
                .map(reverse_map)
            )

        valid = batch[batch[target_column] == minority_class]
        wrong_now = len(batch) - len(valid)
        total_wrong_class += wrong_now

        before = len(valid) + (0 if pool is None else len(pool))
        pool = valid if pool is None else pd.concat([pool, valid], ignore_index=True)
        pool = pool.drop_duplicates(ignore_index=True)
        duplicates_now = before - len(pool)
        total_duplicates += duplicates_now


        print(
            f"{model_name} | fold {fold_id} | attempt {attempts}/{DEFAULT_MAX_GENERATION_ATTEMPTS} | "
            f"generated {len(batch)} | wrong class {len(batch) - before} | duplicates {duplicates_now} | "
            f"accumulated {len(pool)/n_samples:.2f}  "
            f"classes {batch[target_column].value_counts(dropna=False).to_dict()}"
            f" | copies of train {count_exact_copies(valid, train)}"
        )

        if len(valid) >= n_samples:
            valid =pool  # use the pool of valid samples, not just the last batch
            
            if set(valid.columns) != set(train.columns):
                raise ValueError(
                    f"Generated columns differ from train: "
                    f"{sorted(set(valid.columns) ^ set(train.columns))}"
                )

            synthetic = (
                valid[train.columns]  # same column order as train
                .iloc[:n_samples]
                .sample(frac=1, random_state=random_seed)
                .reset_index(drop=True)
            )

            check_artifact(
                synthetic,
                train,
                target_column=target_column,
                expected_rows=n_samples,
                expected_label=minority_class,
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            synthetic.to_csv(
                output_path,
                index=False,
            )

            stats = pd.DataFrame(
                [
                    {
                        "model": model_name,
                        "minority_class": minority_class,
                        "random_seed": random_seed,
                        "generation_seed": random_seed + attempt,
                        "needed": n_samples,
                        "requested": requested,
                        "generated": total_generated,
                        "valid": len(valid),
                        "wrong_class": total_wrong_class,
                        "duplicates": total_duplicates,
                        "excess": len(valid) - n_samples,
                        "attempts": attempts,
                        "copies_train": count_exact_copies(synthetic, train),
                        "copies_test": count_exact_copies(synthetic, test),
                    }
                ]
            )

            return synthetic, stats

    raise RuntimeError(
        f"Not enough valid samples for minority class "
        f"{minority_class!r}: needed {n_samples}, "
        f"requested {requested} per attempt, and "
        f"no attempt produced enough valid samples after "
        f"{DEFAULT_MAX_GENERATION_ATTEMPTS} attempts."
    )
