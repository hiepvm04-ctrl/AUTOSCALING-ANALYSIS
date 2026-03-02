# src/autoscaling_analysis/scaling/anomaly.py

from __future__ import annotations

import numpy as np
import pandas as pd


def mad_anomaly_flags(
    series: pd.Series,
    *,
    window_pts: int,
    k: float,
    min_points: int = 10,
) -> tuple[pd.Series, pd.Series]:
    """
    MAD-based spike score + flags.

    score = |x - rolling_median| / rolling_MAD
    flag if score > k

    min_periods is clamped so it never exceeds window_pts.
    Mirrors the notebook FIX in CELL 10.
    """
    x = pd.to_numeric(series, errors="coerce").astype(float).copy()

    wp = int(max(1, window_pts))
    mp = max(1, min(wp, max(int(min_points), wp // 2)))

    med = x.rolling(wp, min_periods=mp).median()
    mad = (x - med).abs().rolling(wp, min_periods=mp).median()

    score = (x - med).abs() / mad.replace(0, np.nan)
    is_spike = (score > float(k)).fillna(False).astype(int)

    return score.fillna(0.0), is_spike


def ddos_flag(is_spike: pd.Series, consec: int) -> pd.Series:
    """
    DDoS = consecutive spikes (rolling sum >= consec).
    """
    c = int(max(1, consec))
    run = pd.to_numeric(is_spike, errors="coerce").fillna(0).astype(int).rolling(c, min_periods=c).sum()
    return (run >= c).fillna(False).astype(int)