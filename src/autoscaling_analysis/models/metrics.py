# src/autoscaling_analysis/models/metrics.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def mape_threshold(y_true, y_pred, *, min_y: float = 1.0) -> float:
    """
    MAPE but only for points where |y_true| >= min_y (avoid exploding on tiny denominators).
    Returns NaN if no points satisfy the threshold.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) >= float(min_y)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_metrics(y_true, y_pred, *, target: str) -> Dict[str, float]:
    """
    Returns a dict of forecast metrics.

    NOTE:
    - For bytes_sum we often threshold MAPE at 1024 to avoid noise from near-zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = mape_threshold(y_true, y_pred, min_y=1024.0 if target == "bytes_sum" else 1.0)

    return {"RMSE": rmse, "MSE": mse, "MAE": mae, "MAPE": mape}


def write_metrics_long(metrics_csv: str, rows: List[Dict[str, Any]]) -> None:
    """
    Append metrics rows in long format:
      model,target,window,split,metric,value
    """
    if not metrics_csv:
        return

    out_dir = os.path.dirname(metrics_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dfm = pd.DataFrame(rows)

    # if file doesn't exist -> write header
    write_header = not os.path.exists(metrics_csv)
    dfm.to_csv(
        metrics_csv,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )


# -------------------------------------------------------------------
# NEW: Compatibility class expected by models/__init__.py
# -------------------------------------------------------------------
@dataclass
class MetricsWriter:
    """
    Small helper to accumulate/append metrics to a CSV.

    This exists mainly for backward-compatibility because models/__init__.py
    exports MetricsWriter. Current code paths may still call write_metrics_long()
    directly, and that's totally fine.
    """

    metrics_csv: str

    def write(self, rows: List[Dict[str, Any]]) -> None:
        write_metrics_long(self.metrics_csv, rows)

    def write_one(
        self,
        *,
        model: str,
        target: str,
        window: str,
        split: str,
        metrics: Dict[str, float],
    ) -> None:
        out_rows = [
            {
                "model": str(model),
                "target": str(target),
                "window": str(window),
                "split": str(split),
                "metric": str(k),
                "value": float(v) if v is not None else np.nan,
            }
            for k, v in metrics.items()
        ]
        self.write(out_rows)