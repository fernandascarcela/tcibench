"""
Visual EDA of TCI-Bench aggregated results (one dataset at a time).

Usage (notebook)
----------------
    from tcibench.visualization import results_eda
    paths = results_eda.make_figures(aggregated_results, figures_dir, dataset="Pima")

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

Reading notes
-------------
* Δ values are shown in percentage points (pp): +1.5 means +0.015 in the metric.
* The "sd" columns are the variability of each method across the 5 folds.
  They are NOT the uncertainty of a difference between two methods: that
  needs the per-fold paired results and the bootstrap CIs.
* Differences of ~1 pp with sd of ~3-4 pp are inside the fold noise.
"""

from __future__ import annotations

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
    "smotenc": "SMOTE-NC",
    "ctgan": "CTGAN",
    "tvae": "TVAE",
    "ddpm": "TabDDPM",
    "llm": "LLM",
}
METHOD_ORDER = list(METHOD_LABELS)


def method_label(method) -> str:
    """Readable name; LLM runs are stored as 'llm_<model>'."""
    method = str(method)
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    if method.startswith("llm_"):
        return f"LLM ({method[4:]})"
    return method


def is_focus(method) -> bool:
    """LLM methods are highlighted; everything else stays gray."""
    return str(method).startswith("llm")

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

def load_results(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Read aggregated results, mark n/a combinations and keep a fixed order."""

    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)

    # Known methods first, in the fixed order; anything else (e.g. LLM runs) after.
    extra = sorted(set(df["method"]) - set(METHOD_ORDER))
    order = [m for m in METHOD_ORDER if m in set(df["method"])] + extra

    value_columns = [c for c in df.columns if c.endswith(("_mean", "_std"))]
    for classifier, method in NOT_APPLICABLE:
        mask = (df["classifier"] == classifier) & (df["method"] == method)
        df.loc[mask, value_columns] = np.nan

    df["method"] = pd.Categorical(df["method"], order, ordered=True)
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
        ax.set_yticklabels([method_label(m) for m in table.index])
        ax.set_title(METRIC_LABELS[metric])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Thin surface-colored gaps between cells.
        ax.set_xticks(np.arange(-0.5, len(table.columns)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(table.index)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)

        # Bold label for the highlighted methods.
        for label, method in zip(ax.get_yticklabels(), table.index):
            if is_focus(method):
                label.set_fontweight("bold")
                label.set_color(INK)

    colorbar_ax = fig.add_axes([0.885, 0.25, 0.012, 0.55])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(f"Δ vs {method_label(reference)} (pontos percentuais)")
    colorbar.outline.set_visible(False)

    fig.suptitle(
        f"{dataset}: ganho ou perda de cada método em relação a {method_label(reference)}",
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
            if is_focus(method):
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
            if is_focus(method) or method == "real":
                ax.annotate(
                    f"{row[f'{metric}_mean']:.3f}",
                    (row[f"{metric}_mean"] + row[f"{metric}_std"], yi),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2,
                )

        ax.set_yticks(y)
        ax.set_yticklabels([method_label(m) for m in group["method"]])
        for label, method in zip(ax.get_yticklabels(), group["method"]):
            if is_focus(method):
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
            if is_focus(method):
                color, size, text_color, weight = BLUE, 90, BLUE, "bold"
            elif method == "real":
                color, size, text_color, weight = INK, 70, INK, "bold"
            else:
                color, size, text_color, weight = OTHER, 55, INK_2, "normal"
            ax.scatter(row["minority_recall_mean"], row["minority_precision_mean"],
                       s=size, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            labels.append(ax.annotate(
                method_label(method),
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
            focus = is_focus(method)
            ax.scatter(
                x, y, marker=CLASSIFIER_MARKERS[classifier],
                s=85 if focus else 50, color=BLUE if focus else OTHER,
                edgecolor=SURFACE, linewidth=1.5, zorder=4 if focus else 3,
            )
            if focus:
                ax.annotate(f"{method_label(method)} · {CLASSIFIER_LABELS[classifier]}", (x, y),
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
    colors = [BLUE if is_focus(m) else (INK if m == "real" else OTHER) for m in table.index]
    ax.barh(y, table["mean_rank"], color=colors, height=0.6)
    ax.hlines(y, table["best_rank"], table["worst_rank"], color=MUTED, linewidth=1.2, zorder=3)

    for yi, (method, row) in zip(y, table.iterrows()):
        note = "" if row["n_classifiers"] == len(df["classifier"].cat.categories) else \
            f"  ({int(row['n_classifiers'])} classif.)"
        ax.text(row["worst_rank"] + 0.15, yi, f"{row['mean_rank']:.1f}{note}",
                va="center", fontsize=8.5, color=INK_2)

    ax.set_yticks(y)
    ax.set_yticklabels([method_label(m) for m in table.index])
    for label, method in zip(ax.get_yticklabels(), table.index):
        if is_focus(method):
            label.set_fontweight("bold")
            label.set_color(BLUE)
    ax.set_xlim(0, len(table) + 1)
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
# Entry points
# ---------------------------------------------------------------------------

def checks(df: pd.DataFrame) -> list[str]:
    """Human-readable warnings about the results table."""

    df = load_results(df)
    messages = [
        f"n/a: {method_label(m)} em {CLASSIFIER_LABELS.get(c, c)} (não se aplica)"
        for c, m in sorted(NOT_APPLICABLE)
    ]
    messages += [
        f"⚠ {method_label(m)} em {CLASSIFIER_LABELS.get(c, c)} é idêntico ao Real em "
        "todas as métricas: provavelmente o método não foi aplicado."
        for c, m in find_identical_to_real(df)
    ]
    return messages


def make_figures(
    aggregated: pd.DataFrame | str | Path,
    out_dir: str | Path,
    *,
    dataset: str,
    dpi: int = 200,
) -> dict[str, Path]:
    """Build every figure, save it as PNG in `out_dir`, and return the paths."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(aggregated)
    flagged = set(find_identical_to_real(df))
    methods = set(df["method"].astype(str))

    builders = {
        "01_delta_vs_real.png": lambda: plot_delta_heatmaps(df, "real", flagged, dataset),
        "02_delta_vs_ros.png": lambda: plot_delta_heatmaps(df, "ros", flagged, dataset),
        "03_macro_f1_dots.png": lambda: plot_metric_dots(df, "macro_f1", flagged, dataset),
        "04_precision_recall.png": lambda: plot_precision_recall(df, flagged, dataset),
        "05_f1_vs_auprc.png": lambda: plot_f1_vs_auprc(df, flagged, dataset),
        "06_mean_rank.png": lambda: plot_mean_rank(df, "macro_f1", flagged, dataset),
    }
    if "ros" not in methods:
        builders.pop("02_delta_vs_ros.png")

    paths = {}
    for name, build in builders.items():
        fig = build()
        fig.savefig(out_dir / name, dpi=dpi)
        plt.close(fig)
        paths[name] = out_dir / name

    return paths