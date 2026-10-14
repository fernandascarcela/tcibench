"""Preparation visualizations used throughout TCI-Bench."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metadata import DatasetMetadata


# Maximum number of categories displayed per categorical column.
DEFAULT_MAX_CATEGORIES = 15


def plot_feature_distributions(
    cleaned: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    metadata: DatasetMetadata,
    *,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
):
    """
    Compare feature distributions across cleaned, train, and test datasets.

    Numeric features are shown as density histograms.
    Categorical features are shown as normalized frequency bar charts,
    limited to the `max_categories` most frequent values in the cleaned data.

    The caller is responsible for saving and closing the returned figure
    (`plt.close(fig)`), especially when plotting several datasets in a loop.
    """

    columns = metadata.feature_columns + [metadata.target_column]

    ncols = min(3, len(columns))
    nrows = int(np.ceil(len(columns) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )

    datasets = (
        ("Cleaned", cleaned),
        ("Train", train),
        ("Test", test),
    )

    for ax, column in zip(
        axes.flat,
        columns,
    ):
        column_type = metadata.columns[column]["type"]

        if column_type == "numeric":
            bins = np.histogram_bin_edges(
                cleaned[column],
                bins=30,
            )

            for label, frame in datasets:
                ax.hist(
                    frame[column],
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.8,
                    label=label,
                )

            ax.set_ylabel("Density")
            title = column

        else:
            # Keep only the most frequent categories so wide columns
            # (e.g. native-country in Adult) remain readable.
            category_counts = cleaned[column].value_counts()
            shown = category_counts.index[:max_categories]

            proportions = pd.DataFrame(
                {
                    label: frame[column]
                    .value_counts(normalize=True)
                    .reindex(shown)
                    for label, frame in datasets
                }
            ).fillna(0)

            proportions.plot.bar(
                ax=ax,
                width=0.8,
            )

            ax.set_ylabel("Proportion")
            ax.tick_params(
                axis="x",
                labelrotation=70,
            )

            # Signal in the title when some categories were omitted.
            hidden = len(category_counts) - len(shown)
            title = (
                f"{column} (top {len(shown)} of {len(category_counts)})"
                if hidden > 0
                else column
            )

        ax.set_title(title)
        ax.legend(fontsize="small")

    for ax in axes.flat[len(columns):]:
        ax.set_visible(False)

    fig.suptitle(
        "Feature distributions: cleaned, train, and test",
        fontsize=14,
    )

    fig.tight_layout()

    return fig