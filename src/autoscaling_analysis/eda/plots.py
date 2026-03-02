# src/autoscaling_analysis/eda/plots.py

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_timeseries(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str = "time", ylabel: str = ""):
    fig = plt.figure(figsize=(12, 3))
    plt.plot(df[x], df[y], linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel or y)
    plt.tight_layout()
    return fig


def plot_hist(series: pd.Series, title: str, xlabel: str, bins: int = 80):
    fig = plt.figure(figsize=(12, 3))
    plt.hist(pd.to_numeric(series, errors="coerce").dropna().values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    return fig


def plot_heatmap_dow_hour(heat: pd.DataFrame, title: str):
    """
    heat: DataFrame index=dow(0-6), columns=hour(0-23)
    """
    fig = plt.figure(figsize=(12, 3))
    plt.imshow(heat.values, aspect="auto", interpolation="nearest")
    plt.title(title)
    plt.xlabel("hour")
    plt.ylabel("day_of_week")
    plt.colorbar(label="mean hits")
    plt.tight_layout()
    return fig


def plot_mean_by_hour(df: pd.DataFrame, hour_col: str, y: str, title: str):
    fig = plt.figure(figsize=(12, 3))
    df.groupby(hour_col)[y].mean().plot()
    plt.title(title)
    plt.xlabel("hour")
    plt.ylabel(f"mean {y}")
    plt.tight_layout()
    return fig


def plot_mean_by_dow(df: pd.DataFrame, dow_col: str, y: str, title: str):
    fig = plt.figure(figsize=(12, 3))
    df.groupby(dow_col)[y].mean().plot()
    plt.title(title)
    plt.xlabel("dow (0=Mon)")
    plt.ylabel(f"mean {y}")
    plt.tight_layout()
    return fig


def plot_corr_lower_triangle(df: pd.DataFrame, cols: List[str], title: str):
    """
    Lower-triangle correlation heatmap with numeric text values, like notebook CELL 04.
    """
    C = df[cols].corr(numeric_only=True).reindex(index=cols, columns=cols).values
    mask = np.triu(np.ones_like(C, dtype=bool), k=1)
    C_m = np.ma.array(C, mask=mask)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    fig = plt.figure(figsize=(12, 3))
    plt.imshow(C_m, aspect="auto", interpolation="nearest", vmin=-1, vmax=1, cmap=cmap)
    plt.title(title)
    plt.xticks(np.arange(len(cols)), cols, rotation=30, ha="right")
    plt.yticks(np.arange(len(cols)), cols)
    plt.colorbar()

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if not mask[i, j]:
                plt.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_acf_simple(series: pd.Series, max_lag: int, title: str, bucket_min: int):
    """
    Simple ACF by dot product (no statsmodels), like notebook.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float).fillna(0).values
    s = s - s.mean()
    denom = float(np.dot(s, s)) if float(np.dot(s, s)) != 0 else 1.0

    acf = [1.0]
    for k in range(1, max_lag + 1):
        if k >= len(s):
            acf.append(0.0)
        else:
            acf.append(float(np.dot(s[:-k], s[k:]) / denom))

    fig = plt.figure(figsize=(12, 3))
    plt.stem(range(max_lag + 1), acf)
    plt.title(title)
    plt.xlabel(f"lag (bucket = {bucket_min} min)")
    plt.ylabel("ACF")
    plt.tight_layout()
    return fig


def plot_overlay_anomalies(df: pd.DataFrame, time_col: str, value_col: str, flag_col: str, title: str, ylabel: str):
    """
    Overlay anomalies as scatter points.
    """
    fig = plt.figure(figsize=(12, 3))
    plt.plot(df[time_col], df[value_col], alpha=0.5, linewidth=1)

    a = df[df[flag_col].astype(int) == 1]
    if not a.empty:
        plt.scatter(a[time_col], a[value_col], s=40)

    plt.title(title)
    plt.xlabel("time")
    plt.ylabel(ylabel)
    plt.tight_layout()
    return fig