# src/autoscaling_analysis/models/seasonal_naive.py

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from .metrics import compute_metrics, write_metrics_long
from autoscaling_analysis.features.transforms import steps_per_day


@dataclass
class SeasonalNaiveResult:
    """
    Compatibility result object (exported by models/__init__.py).

    Current scripts use the tuple return (pred_df, metrics_dict).
    This dataclass exists so imports from autoscaling_analysis.models work as declared.
    """
    tag: str
    target: str
    season_len: int
    pred_csv_path: str
    pred_parquet_path: str
    test_metrics: Dict[str, float]
    eval_df: pd.DataFrame


def seasonal_naive_forecast(hist: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    hist = np.asarray(hist, dtype=float)
    if len(hist) == 0:
        return np.zeros(horizon, dtype=float)
    if len(hist) < season_len:
        return np.array([hist[-1]] * horizon, dtype=float)
    last_season = hist[-season_len:]
    reps = int(math.ceil(horizon / season_len))
    return np.tile(last_season, reps)[:horizon].astype(float)


def _safe_float(x) -> float:
    v = float(pd.to_numeric(x, errors="coerce"))
    return 0.0 if np.isnan(v) or np.isinf(v) else v


def run_seasonal_naive_one(
    *,
    ts3_root: str,
    tag: str,
    target: str,
    pred_out_dir: str,
    metrics_csv: str,
    model_name: str = "seasonal_naive",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Train/eval seasonal naive baseline and write:
      pred_{target}_{tag}_{model_name}.csv/parquet
    """
    TIME_COL = "bucket_start"

    tr_path = os.path.join(ts3_root, "train", f"ts3_{tag}.parquet")
    te_path = os.path.join(ts3_root, "test", f"ts3_{tag}.parquet")
    tr = pd.read_parquet(tr_path)
    te = pd.read_parquet(te_path)

    tr[TIME_COL] = pd.to_datetime(tr[TIME_COL])
    te[TIME_COL] = pd.to_datetime(te[TIME_COL])

    tr = tr.sort_values(TIME_COL).reset_index(drop=True)
    te = te.sort_values(TIME_COL).reset_index(drop=True)

    # remove gaps in train
    if "is_gap" in tr.columns:
        tr = tr[pd.to_numeric(tr["is_gap"], errors="coerce").fillna(0).astype(int) == 0].copy()
        tr = tr.sort_values(TIME_COL).reset_index(drop=True)

    te2 = te.copy()
    te2["true_next"] = pd.to_numeric(te2[target], errors="coerce").astype(float).shift(-1)
    eval_df = te2[te2["true_next"].notna()].copy().reset_index(drop=True)

    os.makedirs(pred_out_dir, exist_ok=True)

    csv_path = os.path.join(pred_out_dir, f"pred_{target}_{tag}_{model_name}.csv")
    pq_path = os.path.join(pred_out_dir, f"pred_{target}_{tag}_{model_name}.parquet")

    if len(eval_df) == 0:
        out = te2[[TIME_COL, target, "true_next"]].head(0).copy()
        out["pred"] = np.nan
        out.to_csv(csv_path, index=False, encoding="utf-8-sig")
        out.to_parquet(pq_path, index=False)

        tm = {"RMSE": np.nan, "MSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
        rows = [
            {"model": model_name, "target": target, "window": tag, "split": "test", "metric": k,
             "value": float(v) if v is not None else np.nan}
            for k, v in tm.items()
        ]
        write_metrics_long(metrics_csv, rows)
        return out, tm

    # history starts from train series
    hist = pd.to_numeric(tr[target], errors="coerce").astype(float).fillna(0.0).values.tolist()
    season_len = int(steps_per_day(tag))

    preds = []
    # IMPORTANT: Use eval_df length, but update history using te in chronological order
    for i in range(len(eval_df)):
        y_t = _safe_float(te.iloc[i][target])
        hist.append(y_t)
        hist_arr = np.asarray(hist, dtype=float)
        p = seasonal_naive_forecast(hist_arr, horizon=1, season_len=season_len)[0]
        preds.append(max(0.0, float(p)))

    eval_df["pred"] = np.asarray(preds, dtype=float)

    eval_df[[TIME_COL, target, "true_next", "pred"]].to_csv(csv_path, index=False, encoding="utf-8-sig")
    eval_df[[TIME_COL, target, "true_next", "pred"]].to_parquet(pq_path, index=False)

    tm = compute_metrics(eval_df["true_next"].values, eval_df["pred"].values, target=target)
    rows = [
        {"model": model_name, "target": target, "window": tag, "split": "test", "metric": k,
         "value": float(v) if v is not None else np.nan}
        for k, v in tm.items()
    ]
    write_metrics_long(metrics_csv, rows)

    return eval_df[[TIME_COL, target, "true_next", "pred"]], tm


def run_seasonal_naive_all(
    *,
    ts3_root: str,
    tags: List[str],
    targets: List[str],
    pred_out_dir: str,
    metrics_csv: str,
) -> List[Dict[str, Any]]:
    out = []
    for target in targets:
        for tag in tags:
            _, tm = run_seasonal_naive_one(
                ts3_root=ts3_root,
                tag=tag,
                target=target,
                pred_out_dir=pred_out_dir,
                metrics_csv=metrics_csv,
                model_name="seasonal_naive",
            )
            out.append({"model": "seasonal_naive", "target": target, "window": tag, **tm})
            print(f"[seasonal_naive] ✅ {target}/{tag} TEST:", tm)
    return out


# -------------------------------------------------------------------
# Compatibility aliases expected by autoscaling_analysis.models.__init__
# -------------------------------------------------------------------
def train_seasonal_naive_one(
    *,
    ts3_root: str,
    tag: str,
    target: str,
    pred_out_dir: str,
    metrics_csv: str,
    model_name: str = "seasonal_naive",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Alias for backward-compatibility with the public API.
    """
    return run_seasonal_naive_one(
        ts3_root=ts3_root,
        tag=tag,
        target=target,
        pred_out_dir=pred_out_dir,
        metrics_csv=metrics_csv,
        model_name=model_name,
    )