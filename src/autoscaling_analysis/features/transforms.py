# src/autoscaling_analysis/features/transforms.py

from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


def tag_minutes(tag: str) -> int:
    return {"1m": 1, "5m": 5, "15m": 15}[tag]


def steps_per_day(tag: str) -> int:
    return int(24 * 60 / tag_minutes(tag))


def steps_per_hour(tag: str) -> int:
    return int(60 / tag_minutes(tag))


def resolve_roll_windows(tag: str, roll_windows: List[str]) -> Dict[str, int]:
    sph = steps_per_hour(tag)
    spd = steps_per_day(tag)
    out = {}
    for w in roll_windows:
        if w == "1h":
            out[w] = 1 * sph
        elif w == "6h":
            out[w] = 6 * sph
        elif w == "1d":
            out[w] = 1 * spd
        else:
            raise ValueError(f"Unsupported roll window: {w}")
    return out


# -------------------------------------------------------------------
# NEW: helpers expected by autoscaling_analysis.features.__init__
# -------------------------------------------------------------------
def assert_required_cols(df: pd.DataFrame, cols: List[str], name: str = "df") -> None:
    """
    Raise a clear error if df is missing any required columns.
    This is imported by autoscaling_analysis.features.__init__.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing required cols: {missing}")


def to_tznaive_dt64(s: pd.Series) -> pd.Series:
    """
    Convert a datetime-like Series to tz-naive datetime64.
    (Existing function used by feature builder.)
    """
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(None)
    return s


def to_tz_naive(s: pd.Series) -> pd.Series:
    """
    Backward-compatible alias expected by autoscaling_analysis.features.__init__.
    """
    return to_tznaive_dt64(s)


def build_time_key(ts: pd.Series) -> pd.Series:
    """
    Build a stable int64 join key from timestamps (tz-naive recommended).
    Useful for safe merges without tz surprises.
    """
    t = to_tznaive_dt64(ts)
    # view('int64') gives nanoseconds since epoch
    return t.view("int64")


# -------------------------------------------------------------------
# Feature transforms
# -------------------------------------------------------------------
def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    hits = pd.to_numeric(d["hits"], errors="coerce").fillna(0.0).astype(float)
    bsum = pd.to_numeric(d["bytes_sum"], errors="coerce").fillna(0.0).astype(float)
    d["avg_bytes_per_req"] = bsum / np.maximum(hits, 1.0)
    return d


def create_time_features(
    df: pd.DataFrame,
    *,
    tag: str,
    time_col: str,
    ref_time: pd.Timestamp,
    use_cyclic: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy()
    t = pd.to_datetime(d[time_col], errors="coerce")
    ref_time = pd.to_datetime(ref_time)

    # keep timezone consistent
    if getattr(t.dt, "tz", None) is not None and getattr(ref_time, "tzinfo", None) is None:
        ref_time = ref_time.tz_localize(t.dt.tz)
    elif getattr(t.dt, "tz", None) is None and getattr(ref_time, "tzinfo", None) is not None:
        t = t.dt.tz_localize(ref_time.tzinfo)

    d["hour"] = t.dt.hour.astype("int16")
    d["minute"] = t.dt.minute.astype("int16")
    d["dayofweek"] = t.dt.dayofweek.astype("int16")
    d["month"] = t.dt.month.astype("int16")
    d["dayofyear"] = t.dt.dayofyear.astype("int16")
    d["is_weekend"] = (d["dayofweek"] >= 5).astype("int8")

    step_seconds = tag_minutes(tag) * 60
    d["time_idx"] = ((t - ref_time).dt.total_seconds() / step_seconds).astype("int64")

    cols = ["hour", "minute", "dayofweek", "month", "dayofyear", "is_weekend", "time_idx"]

    if use_cyclic:
        hour = d["hour"].astype(float)
        dow = d["dayofweek"].astype(float)
        d["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        d["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        d["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        d["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        cols += ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    return d, cols


def add_lags(
    df: pd.DataFrame,
    *,
    tag: str,
    target: str,
    time_col: str,
    seg_col: str,
    lag_days: List[int],
) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy()
    spd = steps_per_day(tag)
    lag_steps = [int(x * spd) for x in lag_days]
    pref = f"{target}_"

    ok = d[d[seg_col] >= 0].copy()
    gap = d[d[seg_col] < 0].copy()

    def _per_seg(g):
        g = g.sort_values(time_col).copy()
        y = pd.to_numeric(g[target], errors="coerce").astype(float)
        for days, k in zip(lag_days, lag_steps):
            g[f"{pref}lag_{days}d"] = y.shift(k)
        if len(lag_days) >= 2:
            d0, d1 = lag_days[0], lag_days[1]
            g[f"{pref}diff_lag_{d0}d_{d1}d"] = g[f"{pref}lag_{d0}d"] - g[f"{pref}lag_{d1}d"]
        return g

    ok = ok.groupby(seg_col, group_keys=False).apply(_per_seg)
    out = pd.concat([ok, gap], ignore_index=True).sort_values(time_col).reset_index(drop=True)

    feat_cols = [f"{pref}lag_{d}d" for d in lag_days]
    if len(lag_days) >= 2:
        d0, d1 = lag_days[0], lag_days[1]
        feat_cols.append(f"{pref}diff_lag_{d0}d_{d1}d")
    return out, feat_cols


def add_rolling(
    df: pd.DataFrame,
    *,
    tag: str,
    target: str,
    time_col: str,
    seg_col: str,
    roll_windows: List[str],
    use_std: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy()
    roll_map = resolve_roll_windows(tag, roll_windows)
    pref = f"{target}_"

    ok = d[d[seg_col] >= 0].copy()
    gap = d[d[seg_col] < 0].copy()

    def _per_seg(g):
        g = g.sort_values(time_col).copy()
        y = pd.to_numeric(g[target], errors="coerce").astype(float)
        y_shift = y.shift(1)  # prevent leakage
        for wname, win in roll_map.items():
            g[f"{pref}roll_mean_{wname}"] = y_shift.rolling(win, min_periods=win).mean()
            if use_std:
                g[f"{pref}roll_std_{wname}"] = y_shift.rolling(win, min_periods=win).std()
        return g

    ok = ok.groupby(seg_col, group_keys=False).apply(_per_seg)
    out = pd.concat([ok, gap], ignore_index=True).sort_values(time_col).reset_index(drop=True)

    cols = []
    for wname in roll_map.keys():
        cols.append(f"{pref}roll_mean_{wname}")
        if use_std:
            cols.append(f"{pref}roll_std_{wname}")
    return out, cols


def add_labels(
    df: pd.DataFrame,
    *,
    target: str,
    label_col: str,
    horizon_steps: int,
    time_col: str,
    seg_col: str,
) -> pd.DataFrame:
    d = df.copy()
    ok = d[d[seg_col] >= 0].copy()
    gap = d[d[seg_col] < 0].copy()

    def _per_seg(g):
        g = g.sort_values(time_col).copy()
        g[label_col] = pd.to_numeric(g[target], errors="coerce").astype(float).shift(-int(horizon_steps))
        return g

    ok = ok.groupby(seg_col, group_keys=False).apply(_per_seg)
    out = pd.concat([ok, gap], ignore_index=True).sort_values(time_col).reset_index(drop=True)
    return out