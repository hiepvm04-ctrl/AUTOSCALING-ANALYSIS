# src/autoscaling_analysis/benchmark/build_benchmark.py

from __future__ import annotations

import pandas as pd


def load_metrics_long(metrics_csv: str) -> pd.DataFrame:
    """
    Expected schema:
      model,target,window,split,metric,value
    """
    if not metrics_csv:
        return pd.DataFrame(columns=["model", "target", "window", "split", "metric", "value"])
    try:
        df = pd.read_csv(metrics_csv)
    except FileNotFoundError:
        return pd.DataFrame(columns=["model", "target", "window", "split", "metric", "value"])

    need = ["model", "target", "window", "split", "metric", "value"]
    for c in need:
        if c not in df.columns:
            df[c] = pd.NA
    return df[need].copy()


def build_benchmark_table(metrics_long: pd.DataFrame, *, split: str = "test") -> pd.DataFrame:
    """
    long -> wide table like notebook CELL 8:
      index = (target, window, metric)
      columns = model
    """
    if metrics_long.empty:
        return pd.DataFrame(columns=["target", "window", "metric"])

    mdf = metrics_long.copy()
    mdf["split"] = mdf["split"].astype(str).str.lower()
    mdf["model"] = mdf["model"].astype(str)
    mdf["target"] = mdf["target"].astype(str)
    mdf["window"] = mdf["window"].astype(str)
    mdf["metric"] = mdf["metric"].astype(str)

    test_m = mdf[mdf["split"].eq(str(split).lower())].copy()

    bench = (
        test_m.pivot_table(
            index=["target", "window", "metric"],
            columns=["model"],
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["target", "window", "metric"])
        .reset_index(drop=True)
    )

    return bench