"""Plotting helpers for EDA — grid layouts for univariate/bivariate analysis
so the notebook calls one function per section instead of hand-writing
dozens of near-identical matplotlib cells.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_categorical_grid(df: pd.DataFrame, columns: list[str], ncols: int = 4, save_path: str | None = None):
    """Univariate: bar chart of value counts for each categorical/binary column."""
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.countplot(x=df[col], ax=axes[i], color="steelblue")
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_numeric_distributions(df: pd.DataFrame, columns: list[str], save_path: str | None = None):
    """Univariate: histogram + KDE for continuous/near-continuous columns."""
    ncols = min(len(columns), 3)
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten() if len(columns) > 1 else [axes]

    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i], color="darkorange")
        axes[i].set_title(f"Distribution — {col}", fontsize=11)

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_bivariate_categorical_grid(df: pd.DataFrame, columns: list[str], target: str, ncols: int = 3, save_path: str | None = None):
    """Bivariate: each categorical/binary feature's distribution split by target class,
    shown as normalized (percentage) bars so class imbalance doesn't distort the comparison.
    """
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ct = pd.crosstab(df[col], df[target], normalize="index") * 100
        ct.plot(kind="bar", stacked=False, ax=axes[i], color=["#4c72b0", "#dd8452"])
        axes[i].set_title(f"{col} vs {target}", fontsize=10)
        axes[i].set_ylabel("% within group")
        axes[i].legend(title=target, fontsize=7)

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_bivariate_numeric(df: pd.DataFrame, columns: list[str], target: str, ncols: int = 3, save_path: str | None = None):
    """Bivariate: boxplots of each continuous/ordinal feature split by target class."""
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten() if len(columns) > 1 else [axes]

    for i, col in enumerate(columns):
        sns.boxplot(x=target, y=col, data=df, hue=target, palette=["#4c72b0", "#dd8452"], legend=False, ax=axes[i])
        axes[i].set_title(f"{col} by {target}", fontsize=10)

    for j in range(len(columns), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_association_ranking(assoc_df: pd.DataFrame, save_path: str | None = None):
    """Horizontal bar chart ranking features by association strength with the target."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(assoc_df) * 0.35)))
    colors = assoc_df["value"].apply(lambda v: "#c44e52" if v < 0 else "#4c72b0")
    ax.barh(assoc_df["feature"], assoc_df["abs_value"], color=colors)
    ax.set_xlabel("Association strength with target (absolute value)")
    ax.set_title("Feature Associations with Diabetes Status (ranked)")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
