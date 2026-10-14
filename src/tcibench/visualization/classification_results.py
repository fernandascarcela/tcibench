"""
Visual EDA of TCI-Bench aggregated results (one dataset per CSV).

Usage
-----
    python visualize_results.py aggregated_results.csv --dataset Pima --out figures/

Expected columns: classifier, method, <metric>_mean, <metric>_std for
macro_f1, auprc, balanced_accuracy, minority_precision, minority_recall, minority_f1.

Figures
-------
01_delta_vs_real.png    Heatmap: how much each method gains/loses vs Real (no balancing)
02_delta_vs_ros.png     Heatmap: same, vs ROS (the simplest oversampling baseline)
03_macro_f1_dots.png    Macro-F1 mean ± sd per classifier, methods sorted
04_precision_recall.png Minority precision × recall, with iso-F1 curves
05_f1_vs_auprc.png      ΔMacro-F1 × ΔAUPRC vs Real: threshold shift or better ranking?
06_mean_rank.png        Mean Macro-F1 rank of each method across classifiers
07_bootstrap_vs_real.png Paired bootstrap Δ and 95% CI vs Real

Reading notes
-------------
* Δ values are shown in percentage points (pp): +1.5 means +0.015 in the metric.
* The "sd" columns are the variability of each method across the 5 folds.
  They are NOT the uncertainty of a difference between two methods: that
  needs the per-fold paired results and the bootstrap CIs.
* Bootstrap cells show the observed paired difference and its 95% CI. Bold values have CIs that do not cross zero.
* Differences of ~1 pp with sd of ~3-4 pp are inside the fold noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Labels, order and palette
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    "real": "Real",
    "ros": "ROS",
    "rus": "RUS",
    "class_weight": "Pesos por classe",
    "smote": "SMOTE",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "ddpm": "TabDDPM",
    "llm": "LLM",
}
METHOD_ORDER = list(METHOD_LABELS)

CLASSIFIER_LABELS = {
    "logistic_regression": "Regressão logística",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "tabpfn": "TabPFN",
}
CLASSIFIER_ORDER = list(CLASSIFIER_LABELS)

METRIC_LABELS = {
    "macro_f1": "Macro-F1",
    "auprc": "AUPRC",
    "balanced_accuracy": "Acurácia balanceada",
    "minority_precision": "Precisão (minoritária)",
    "minority_recall": "Recall (minoritária)",
    "minority_f1": "F1 (minoritária)",
}
MAIN_METRICS = ["macro_f1", "auprc", "balanced_accuracy"]

# Combinations where the method does not apply (shown as n/a).
NOT_APPLICABLE = {("tabpfn", "class_weight")}

# The method to highlight; everything else stays gray.
FOCUS_METHOD = "llm"

BLUE = "#2a78d6"
RED = "#e34948"
MIDPOINT = "#f0efec"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
OTHER = "#b5b3ab"  # gray for non-highlighted methods

DIVERGING = LinearSegmentedColormap.from_list("red_gray_blue", [RED, MIDPOINT, BLUE])
CLASSIFIER_MARKERS = {
    "logistic_regression": "o",
    "random_forest": "s",
    "xgboost": "D",
    "tabpfn": "^",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
})


# ---------------------------------------------------------------------------
# Data loading and checks
# ---------------------------------------------------------------------------

def load_results(path: str | Path) -> pd.DataFrame:
    """Read the CSV, mark n/a combinations and keep a fixed order."""

    df = pd.read_csv(path)

    unknown = set(df["method"]) - set(METHOD_ORDER)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}. Add them to METHOD_LABELS.")

    value_columns = [c for c in df.columns if c.endswith(("_mean", "_std"))]
    for classifier, method in NOT_APPLICABLE:
        mask = (df["classifier"] == classifier) & (df["method"] == method)
        df.loc[mask, value_columns] = np.nan

    df["method"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
    df["classifier"] = pd.Categorical(
        df["classifier"],
        [c for c in CLASSIFIER_ORDER if c in set(df["classifier"])],
        ordered=True,
    )
    return df.sort_values(["classifier", "method"]).reset_index(drop=True)


def find_identical_to_real(df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Flag (classifier, method) rows whose metrics are identical to Real.

    For a resampling or generative method this almost never happens by
    chance; it usually means the method was not actually applied.
    """

    mean_columns = [f"{m}_mean" for m in METRIC_LABELS]
    flagged = []
    for classifier, group in df.groupby("classifier", observed=True):
        real = group.loc[group["method"] == "real", mean_columns]
        if real.empty:
            continue
        real_values = real.to_numpy()[0]
        for _, row in group[group["method"] != "real"].iterrows():
            values = row[mean_columns].to_numpy(dtype=float)
            if not np.isnan(values).all() and np.allclose(values, real_values, atol=1e-9):
                flagged.append((str(classifier), str(row["method"])))
    return flagged


def delta_table(df: pd.DataFrame, metric: str, reference: str) -> pd.DataFrame:
    """Δ = method − reference, in percentage points, methods × classifiers."""

    means = df.pivot_table(
        index="method", columns="classifier", values=f"{metric}_mean",
        observed=False, dropna=False,
    )
    delta = (means - means.loc[reference]) * 100
    return delta.drop(index=reference)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_delta_heatmaps(
    df: pd.DataFrame,
    reference: str,
    flagged: set[tuple[str, str]],
    dataset: str,
):
    """One heatmap per main metric: blue = better than reference, red = worse."""

    tables = {m: delta_table(df, m, reference) for m in MAIN_METRICS}
    limit = max(np.nanmax(np.abs(t.to_numpy())) for t in tables.values())
    limit = np.ceil(limit)

    fig, axes = plt.subplots(1, len(MAIN_METRICS), figsize=(13, 4.6), sharey=True)

    for ax, (metric, table) in zip(axes, tables.items()):
        values = table.to_numpy(dtype=float)
        image = ax.imshow(values, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")

        for i, method in enumerate(table.index):
            for j, classifier in enumerate(table.columns):
                value = values[i, j]
                if (classifier, method) in NOT_APPLICABLE:
                    text, color, style = "n/a", MUTED, "italic"
                elif (classifier, method) in flagged:
                    text, color, style = "= Real ⚠", INK_2, "italic"
                else:
                    text = f"{value:+.1f}"
                    strong = abs(value) > 0.6 * limit
                    color, style = ("white" if strong else INK), "normal"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=8.5, color=color, style=style)

        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels([CLASSIFIER_LABELS[c] for c in table.columns],
                           rotation=25, ha="right")
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in table.index])
        ax.set_title(METRIC_LABELS[metric])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Thin surface-colored gaps between cells.
        ax.set_xticks(np.arange(-0.5, len(table.columns)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(table.index)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)

        # Bold label for the highlighted method.
        for label in ax.get_yticklabels():
            if label.get_text() == METHOD_LABELS[FOCUS_METHOD]:
                label.set_fontweight("bold")
                label.set_color(INK)

    colorbar_ax = fig.add_axes([0.885, 0.25, 0.012, 0.55])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(f"Δ vs {METHOD_LABELS[reference]} (pontos percentuais)")
    colorbar.outline.set_visible(False)

    fig.suptitle(
        f"{dataset}: ganho ou perda de cada método em relação a {METHOD_LABELS[reference]}",
        x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.text(
        0.01, 0.905,
        "Azul = melhor que a referência · vermelho = pior · média dos 5 folds. "
        "Δ de ~1 pp está dentro da variação entre folds (dp ≈ 3–4 pp).",
        ha="left", fontsize=9, color=INK_2,
    )
    if flagged:
        fig.text(
            0.01, 0.01,
            "⚠ Resultado idêntico ao Real em todas as métricas: o método provavelmente "
            "não foi aplicado (verifique a implementação).",
            ha="left", fontsize=8.5, color=INK_2,
        )
    fig.subplots_adjust(left=0.1, right=0.86, top=0.82, bottom=0.2, wspace=0.06)
    return fig


def plot_metric_dots(df: pd.DataFrame, metric: str, flagged: set[tuple[str, str]], dataset: str):
    """Mean ± sd per method, one panel per classifier, sorted best to worst."""

    classifiers = list(df["classifier"].cat.categories)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)

    low = (df[f"{metric}_mean"] - df[f"{metric}_std"]).min()
    high = (df[f"{metric}_mean"] + df[f"{metric}_std"]).max()

    for ax, classifier in zip(axes.flat, classifiers):
        group = df[df["classifier"] == classifier].dropna(subset=[f"{metric}_mean"])
        group = group[[(classifier, m) not in flagged for m in group["method"]]]
        group = group.sort_values(f"{metric}_mean")
        y = np.arange(len(group))

        real_value = group.loc[group["method"] == "real", f"{metric}_mean"]
        if not real_value.empty:
            ax.axvline(real_value.iloc[0], color=BASELINE, linewidth=1, linestyle="--", zorder=0)

        for yi, (_, row) in zip(y, group.iterrows()):
            method = row["method"]
            if method == FOCUS_METHOD:
                color, size = BLUE, 9
            elif method == "real":
                color, size = INK, 8
            else:
                color, size = OTHER, 8
            ax.errorbar(
                row[f"{metric}_mean"], yi, xerr=row[f"{metric}_std"],
                fmt="o", color=color, markersize=size, linewidth=1.6,
                capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.5,
            )
            if method in (FOCUS_METHOD, "real"):
                ax.annotate(
                    f"{row[f'{metric}_mean']:.3f}",
                    (row[f"{metric}_mean"] + row[f"{metric}_std"], yi),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2,
                )

        ax.set_yticks(y)
        ax.set_yticklabels([METHOD_LABELS[m] for m in group["method"]])
        for label in ax.get_yticklabels():
            if label.get_text() == METHOD_LABELS[FOCUS_METHOD]:
                label.set_fontweight("bold")
                label.set_color(BLUE)
        ax.set_title(CLASSIFIER_LABELS[classifier], loc="left")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(low - 0.01, high + 0.03)

    for ax in axes[-1]:
        ax.set_xlabel(f"{METRIC_LABELS[metric]} (média ± dp entre folds)")

    fig.suptitle(
        f"{dataset}: {METRIC_LABELS[metric]} por classificador",
        x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.text(
        0.01, 0.925,
        "Ordenado do pior (baixo) ao melhor (topo). Linha tracejada = Real. "
        "Barras muito sobrepostas = diferença não distinguível só por estes números.",
        ha="left", fontsize=9, color=INK_2,
    )
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.08, hspace=0.3, wspace=0.35)
    return fig


def plot_precision_recall(df: pd.DataFrame, flagged: set[tuple[str, str]], dataset: str):
    """Minority precision × recall. Iso-F1 curves show where F1 is equal."""

    classifiers = list(df["classifier"].cat.categories)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)

    data = df.dropna(subset=["minority_precision_mean"])
    x_min, x_max = data["minority_recall_mean"].min() - 0.03, data["minority_recall_mean"].max() + 0.03
    y_min, y_max = data["minority_precision_mean"].min() - 0.03, data["minority_precision_mean"].max() + 0.03

    recall_grid = np.linspace(0.01, 1, 400)
    iso_levels = np.arange(0.55, 0.80, 0.05)

    for ax, classifier in zip(axes.flat, classifiers):
        # Iso-F1 curves: P = F1·R / (2R − F1)
        for f1 in iso_levels:
            with np.errstate(divide="ignore", invalid="ignore"):
                precision = f1 * recall_grid / (2 * recall_grid - f1)
            valid = (precision > 0) & (precision <= 1)
            ax.plot(recall_grid[valid], precision[valid], color=GRID, linewidth=0.9, zorder=0)
            # Label each curve at the right edge of the visible area.
            label_r = x_max - 0.005
            label_p = f1 * label_r / (2 * label_r - f1)
            if y_min < label_p < y_max:
                ax.text(label_r, label_p, f"F1={f1:.2f}", fontsize=7, color=MUTED,
                        ha="right", va="bottom")

        group = data[data["classifier"] == classifier]
        group = group[[(classifier, m) not in flagged for m in group["method"]]]
        labels = []
        for _, row in group.iterrows():
            method = row["method"]
            if method == FOCUS_METHOD:
                color, size, text_color, weight = BLUE, 90, BLUE, "bold"
            elif method == "real":
                color, size, text_color, weight = INK, 70, INK, "bold"
            else:
                color, size, text_color, weight = OTHER, 55, INK_2, "normal"
            ax.scatter(row["minority_recall_mean"], row["minority_precision_mean"],
                       s=size, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            labels.append(ax.annotate(
                METHOD_LABELS[method],
                (row["minority_recall_mean"], row["minority_precision_mean"]),
                xytext=(6, 4), textcoords="offset points",
                fontsize=7.5, color=text_color, fontweight=weight,
            ))
        ax._tci_labels = labels

        ax.set_title(CLASSIFIER_LABELS[classifier], loc="left")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    for ax in axes[-1]:
        ax.set_xlabel("Recall da classe minoritária  →  encontra mais positivos")
    for ax in axes[:, 0]:
        ax.set_ylabel("Precisão da classe minoritária  →  menos alarmes falsos")

    fig.suptitle(
        f"{dataset}: troca entre precisão e recall na classe minoritária",
        x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.text(
        0.01, 0.935,
        "Real costuma ficar no alto à esquerda (preciso, mas perde positivos); balancear empurra "
        "para a direita. Pontos na mesma curva cinza têm o mesmo F1.",
        ha="left", fontsize=9, color=INK_2,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.07, hspace=0.18, wspace=0.08)
    for ax in axes.flat:
        _place_labels(fig, ax, getattr(ax, "_tci_labels", []))
    return fig


def _place_labels(fig, ax, labels) -> None:
    """Move each point label to the first nearby offset that does not collide."""

    candidates = [(6, 4), (6, -11), (-6, 4), (-6, -11), (0, 9), (0, -15),
                  (10, 12), (10, -19), (-10, 12), (-10, -19), (14, -4), (-14, -4)]
    renderer = fig.canvas.get_renderer()
    taken = []
    # Point markers also count as occupied space.
    for label in labels:
        x, y = ax.transData.transform(label.xy)
        taken.append((x - 5, y - 5, x + 5, y + 5))

    def overlaps(box):
        return any(not (box.x1 < a or box.x0 > c or box.y1 < b or box.y0 > d)
                   for a, b, c, d in taken)

    for label in labels:
        chosen = None
        for dx, dy in candidates:
            label.set_position((dx, dy))
            label.set_ha("left" if dx >= 0 else "right")
            box = label.get_window_extent(renderer).expanded(1.05, 1.15)
            if not overlaps(box):
                chosen = box
                break
        if chosen is None:  # no free spot: keep the default
            label.set_position(candidates[0])
            label.set_ha("left")
            chosen = label.get_window_extent(renderer)
        taken.append((chosen.x0, chosen.y0, chosen.x1, chosen.y1))


def plot_f1_vs_auprc(df: pd.DataFrame, flagged: set[tuple[str, str]], dataset: str):
    """
    ΔMacro-F1 × ΔAUPRC vs Real for every (classifier, method).

    F1 up with AUPRC flat is compatible with a threshold shift, not evidence
    of a better ranking. Both up is the stronger result.
    """

    f1 = delta_table(df, "macro_f1", "real")
    auprc = delta_table(df, "auprc", "real")

    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    ax.axvline(0, color=BASELINE, linewidth=1, zorder=1)

    for method in f1.index:
        for classifier in f1.columns:
            if (classifier, method) in NOT_APPLICABLE or (classifier, method) in flagged:
                continue
            x, y = auprc.loc[method, classifier], f1.loc[method, classifier]
            if np.isnan(x) or np.isnan(y):
                continue
            focus = method == FOCUS_METHOD
            ax.scatter(
                x, y, marker=CLASSIFIER_MARKERS[classifier],
                s=85 if focus else 50, color=BLUE if focus else OTHER,
                edgecolor=SURFACE, linewidth=1.5, zorder=4 if focus else 3,
            )
            if focus:
                ax.annotate(f"LLM · {CLASSIFIER_LABELS[classifier]}", (x, y),
                            xytext=(7, 4), textcoords="offset points",
                            fontsize=8, color=BLUE, fontweight="bold")

    x_lim = max(abs(np.nanmin(auprc.to_numpy())), abs(np.nanmax(auprc.to_numpy()))) + 0.8
    y_lim = max(abs(np.nanmin(f1.to_numpy())), abs(np.nanmax(f1.to_numpy()))) + 0.8
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)

    quadrant_style = dict(fontsize=8.5, color=MUTED, style="italic")
    ax.text(x_lim * 0.97, y_lim * 0.95, "F1 e AUPRC sobem\n(melhor de fato)",
            ha="right", va="top", **quadrant_style)
    ax.text(-x_lim * 0.97, y_lim * 0.95, "F1 sobe, AUPRC não\n(compatível com mudança de limiar)",
            ha="left", va="top", **quadrant_style)
    ax.text(-x_lim * 0.97, -y_lim * 0.95, "ambos pioram",
            ha="left", va="bottom", **quadrant_style)
    ax.text(x_lim * 0.97, -y_lim * 0.95, "ranking melhora,\nlimiar não ajuda",
            ha="right", va="bottom", **quadrant_style)

    ax.set_xlabel("Δ AUPRC vs Real (pp)  →  ordena melhor os positivos")
    ax.set_ylabel("Δ Macro-F1 vs Real (pp)  →  acerta mais a classe prevista")
    ax.grid(True)
    ax.set_axisbelow(True)

    handles = [
        Line2D([], [], marker=CLASSIFIER_MARKERS[c], linestyle="", color=MUTED,
               markersize=7, label=CLASSIFIER_LABELS[c])
        for c in f1.columns
    ]
    handles += [
        Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=8, label="LLM"),
        Line2D([], [], marker="o", linestyle="", color=OTHER, markersize=7, label="Outros métodos"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1))

    fig.suptitle(
        f"{dataset}: o ganho em F1 vem acompanhado de ganho em AUPRC?",
        x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.text(0.01, 0.915, "Cada ponto é um par (método, classificador), comparado ao Real.",
             ha="left", fontsize=9, color=INK_2)
    fig.subplots_adjust(left=0.1, right=0.78, top=0.87, bottom=0.1)
    return fig


def mean_rank_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rank methods inside each classifier (1 = best), then average across classifiers."""

    means = df.pivot_table(index="method", columns="classifier",
                           values=f"{metric}_mean", observed=True)
    ranks = means.rank(ascending=False, method="average")
    summary = pd.DataFrame({
        "mean_rank": ranks.mean(axis=1),
        "n_classifiers": ranks.notna().sum(axis=1),
        "best_rank": ranks.min(axis=1),
        "worst_rank": ranks.max(axis=1),
    })
    return summary.sort_values("mean_rank")


def plot_mean_rank(df: pd.DataFrame, metric: str, flagged: set[tuple[str, str]], dataset: str):
    """Horizontal bars of mean rank (shorter = better), with best–worst range."""

    data = df.copy()
    for classifier, method in flagged:
        mask = (data["classifier"] == classifier) & (data["method"] == method)
        data.loc[mask, f"{metric}_mean"] = np.nan
    table = mean_rank_table(data, metric).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(table))
    colors = [BLUE if m == FOCUS_METHOD else (INK if m == "real" else OTHER) for m in table.index]
    ax.barh(y, table["mean_rank"], color=colors, height=0.6)
    ax.hlines(y, table["best_rank"], table["worst_rank"], color=MUTED, linewidth=1.2, zorder=3)

    for yi, (method, row) in zip(y, table.iterrows()):
        note = "" if row["n_classifiers"] == len(df["classifier"].cat.categories) else \
            f"  ({int(row['n_classifiers'])} classif.)"
        ax.text(row["worst_rank"] + 0.15, yi, f"{row['mean_rank']:.1f}{note}",
                va="center", fontsize=8.5, color=INK_2)

    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[m] for m in table.index])
    for label in ax.get_yticklabels():
        if label.get_text() == METHOD_LABELS[FOCUS_METHOD]:
            label.set_fontweight("bold")
            label.set_color(BLUE)
    ax.set_xlim(0, len(METHOD_ORDER) + 1)
    ax.set_xlabel(f"Posição média em {METRIC_LABELS[metric]} (1 = melhor)  ·  traço = melhor–pior posição")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    fig.suptitle(f"{dataset}: ranking médio dos métodos entre os classificadores",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.9, "Barra curta = método costuma ficar entre os melhores. "
             "Ranking ignora o tamanho da diferença.", ha="left", fontsize=9, color=INK_2)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.83, bottom=0.13)
    return fig


# ---------------------------------------------------------------------------
# Bootstrap figures and tables
# ---------------------------------------------------------------------------

BOOTSTRAP_METRICS = [
    "macro_f1",
    "auprc",
    "balanced_accuracy",
    "minority_precision",
    "minority_recall",
    "minority_f1",
]


def bootstrap_table(
    bootstrap_df: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Format paired bootstrap results as a compact presentation table."""

    metrics = metrics or BOOTSTRAP_METRICS
    rows = []

    for _, row in bootstrap_df.iterrows():
        output = {
            "classifier": row["classifier"],
            "method": row["method"] if "method" in row else row["comparison"],
        }

        for metric in metrics:
            delta = row[f"{metric}_delta"]
            lower = row[f"{metric}_ci_lower"]
            upper = row[f"{metric}_ci_upper"]

            output[metric] = (
                f"{delta:+.3f} "
                f"[{lower:+.3f}; {upper:+.3f}]"
            )

        rows.append(output)

    table = pd.DataFrame(rows)
    table["classifier"] = table["classifier"].map(
        lambda value: CLASSIFIER_LABELS.get(str(value), str(value))
    )

    if "method" in table.columns:
        table["method"] = table["method"].map(
            lambda value: METHOD_LABELS.get(str(value), str(value))
        )

    return table.rename(columns={
        "classifier": "Classificador",
        "method": "Método − Real",
        "macro_f1": "Macro-F1 Δ [IC95%]",
        "auprc": "AUPRC Δ [IC95%]",
        "balanced_accuracy": "Acurácia balanceada Δ [IC95%]",
        "minority_precision": "Precisão minoritária Δ [IC95%]",
        "minority_recall": "Recall minoritário Δ [IC95%]",
        "minority_f1": "F1 minoritário Δ [IC95%]",
    })


def plot_bootstrap_heatmaps(
    bootstrap_df: pd.DataFrame,
    dataset: str,
):
    """Show paired bootstrap differences with 95% CIs in compact heatmaps."""

    fig, axes = plt.subplots(
        1,
        len(MAIN_METRICS),
        figsize=(14, 6.2),
        sharey=True,
    )

    if len(MAIN_METRICS) == 1:
        axes = [axes]

    for ax, metric in zip(axes, MAIN_METRICS):
        value_col = f"{metric}_delta"
        lower_col = f"{metric}_ci_lower"
        upper_col = f"{metric}_ci_upper"

        table = bootstrap_df.pivot(
            index="method",
            columns="classifier",
            values=value_col,
        )

        lower = bootstrap_df.pivot(
            index="method",
            columns="classifier",
            values=lower_col,
        )

        upper = bootstrap_df.pivot(
            index="method",
            columns="classifier",
            values=upper_col,
        )

        table = table.reindex(
            index=[m for m in METHOD_ORDER if m != "real" and m in table.index],
            columns=[c for c in CLASSIFIER_ORDER if c in table.columns],
        )
        lower = lower.reindex(index=table.index, columns=table.columns)
        upper = upper.reindex(index=table.index, columns=table.columns)

        values = table.to_numpy(dtype=float)
        finite = np.isfinite(values)

        if finite.any():
            limit = max(
                0.01,
                np.nanmax(np.abs(values[finite])),
                np.nanmax(np.abs(lower.to_numpy(dtype=float)[np.isfinite(lower)])),
                np.nanmax(np.abs(upper.to_numpy(dtype=float)[np.isfinite(upper)])),
            )
        else:
            limit = 0.01

        limit = np.ceil(limit * 100) / 100

        image = ax.imshow(
            values,
            cmap=DIVERGING,
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )

        for i, method in enumerate(table.index):
            for j, classifier in enumerate(table.columns):
                delta = table.iloc[i, j]
                ci_low = lower.iloc[i, j]
                ci_high = upper.iloc[i, j]

                if not np.isfinite(delta):
                    continue

                excludes_zero = ci_low > 0 or ci_high < 0
                text_color = "white" if abs(delta) > 0.65 * limit else INK

                ax.text(
                    j,
                    i,
                    f"{delta:+.3f}\n[{ci_low:+.3f}, {ci_high:+.3f}]",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=text_color,
                    fontweight="bold" if excludes_zero else "normal",
                )

        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(
            [CLASSIFIER_LABELS[c] for c in table.columns],
            rotation=25,
            ha="right",
        )
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in table.index])

        ax.set_title(METRIC_LABELS[metric])
        ax.tick_params(length=0)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks(
            np.arange(-0.5, len(table.columns)),
            minor=True,
        )
        ax.set_yticks(
            np.arange(-0.5, len(table.index)),
            minor=True,
        )
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)

    colorbar_ax = fig.add_axes([0.90, 0.25, 0.012, 0.52])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Δ vs Real")
    colorbar.outline.set_visible(False)

    fig.suptitle(
        f"{dataset}: efeito dos métodos em relação ao Real",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )

    fig.text(
        0.01,
        0.905,
        "Valor central = diferença observada. "
        "Colchetes = IC95% do bootstrap pareado. "
        "Negrito = IC95% não cruza zero.",
        ha="left",
        fontsize=9,
        color=INK_2,
    )

    fig.subplots_adjust(
        left=0.1,
        right=0.87,
        top=0.82,
        bottom=0.2,
        wspace=0.08,
    )

    return fig


def show_results(
    aggregated_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame | None = None,
    *,
    dataset: str = "Dataset",
    figures: tuple[str, ...] = (
        "delta",
        "macro_f1",
        "precision_recall",
        "f1_auprc",
        "rank",
        "bootstrap",
    ),
) -> dict[str, object]:
    """Display the main TCI-Bench results directly in a notebook."""

    aggregated_df = aggregated_df.copy()
    aggregated_df["method"] = pd.Categorical(
        aggregated_df["method"], METHOD_ORDER, ordered=True
    )
    aggregated_df["classifier"] = pd.Categorical(
        aggregated_df["classifier"],
        [c for c in CLASSIFIER_ORDER if c in set(aggregated_df["classifier"])],
        ordered=True,
    )

    flagged_list = find_identical_to_real(aggregated_df)
    flagged = set(flagged_list)

    available = {
        "delta": plot_delta_heatmaps(
            aggregated_df, "real", flagged, dataset
        ),
        "macro_f1": plot_metric_dots(
            aggregated_df, "macro_f1", flagged, dataset
        ),
        "precision_recall": plot_precision_recall(
            aggregated_df, flagged, dataset
        ),
        "f1_auprc": plot_f1_vs_auprc(
            aggregated_df, flagged, dataset
        ),
        "rank": plot_mean_rank(
            aggregated_df, "macro_f1", flagged, dataset
        ),
    }

    if bootstrap_df is not None:
        bootstrap_df = bootstrap_df.copy()
        if "method" not in bootstrap_df.columns:
            bootstrap_df["method"] = bootstrap_df["comparison"].str.replace(
                " - real", "", regex=False
            )
        available["bootstrap"] = plot_bootstrap_heatmaps(
            bootstrap_df,
            dataset,
        )

    for name in figures:
        if name not in available:
            continue

        fig = available[name]

        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            pass

    if bootstrap_df is not None:
        table = bootstrap_table(bootstrap_df)
        try:
            from IPython.display import display
            display(table)
        except ImportError:
            print(table.to_string(index=False))

    return available


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, flagged: list[tuple[str, str]]) -> None:
    print("\n=== Verificações ===")
    for classifier, method in sorted(NOT_APPLICABLE):
        print(f"  n/a: {METHOD_LABELS[method]} em {CLASSIFIER_LABELS[classifier]} (não se aplica)")
    for classifier, method in flagged:
        print(f"  ⚠ {METHOD_LABELS[method]} em {CLASSIFIER_LABELS[classifier]} é idêntico ao Real "
              f"em todas as métricas: provavelmente não foi aplicado.")

    print(f"\n=== {METHOD_LABELS[FOCUS_METHOD]}: Δ em pontos percentuais ===")
    for reference in ("real", "ros"):
        rows = []
        for metric in MAIN_METRICS:
            delta = delta_table(df, metric, reference).loc[FOCUS_METHOD]
            rows.append(delta.rename(METRIC_LABELS[metric]))
        table = pd.DataFrame(rows).rename(columns=CLASSIFIER_LABELS)
        print(f"\nvs {METHOD_LABELS[reference]}:")
        print(table.round(1).to_string())

    print("\n=== Ranking médio em Macro-F1 (1 = melhor) ===")
    data = df.copy()
    for classifier, method in flagged:
        mask = (data["classifier"] == classifier) & (data["method"] == method)
        data.loc[mask, "macro_f1_mean"] = np.nan
    ranks = mean_rank_table(data, "macro_f1")
    ranks.index = [METHOD_LABELS[m] for m in ranks.index]
    print(ranks.round(2).to_string())
    print("\nLembrete: dp = variação de cada método entre folds, não incerteza da diferença.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="aggregated_results.csv")
    parser.add_argument("--dataset", default="Dataset", help="Name shown in the titles")
    parser.add_argument("--out", default="figures", help="Output folder")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--bootstrap",
        default=None,
        help="Optional bootstrap_summary.csv",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_results(args.csv)
    flagged_list = find_identical_to_real(df)
    flagged = set(flagged_list)

    figures = {
        "01_delta_vs_real.png": plot_delta_heatmaps(df, "real", flagged, args.dataset),
        "02_delta_vs_ros.png": plot_delta_heatmaps(df, "ros", flagged, args.dataset),
        "03_macro_f1_dots.png": plot_metric_dots(df, "macro_f1", flagged, args.dataset),
        "04_precision_recall.png": plot_precision_recall(df, flagged, args.dataset),
        "05_f1_vs_auprc.png": plot_f1_vs_auprc(df, flagged, args.dataset),
        "06_mean_rank.png": plot_mean_rank(df, "macro_f1", flagged, args.dataset),
    }

    bootstrap_df = None
    if args.bootstrap:
        bootstrap_df = pd.read_csv(args.bootstrap)
        figures["07_bootstrap_vs_real.png"] = plot_bootstrap_heatmaps(
            bootstrap_df,
            args.dataset,
        )
    for name, fig in figures.items():
        fig.savefig(out / name, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    if bootstrap_df is not None:
        bootstrap_table(bootstrap_df).to_csv(
            out / "bootstrap_summary_formatted.csv",
            index=False,
        )

    # Table view of the heatmaps, for the text and appendix.
    for reference in ("real", "ros"):
        tables = []
        for metric in MAIN_METRICS:
            table = delta_table(df, metric, reference).round(2)
            table.insert(0, "metric", metric)
            tables.append(table.reset_index())
        pd.concat(tables).to_csv(out / f"delta_vs_{reference}_pp.csv", index=False)

    print_summary(df, flagged_list)
    print(f"\nFiguras e tabelas salvas em: {out.resolve()}")


if __name__ == "__main__":
    main()