from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GapConfig:
    storm_start: pd.Timestamp
    storm_end: pd.Timestamp
    unknown_gap_min_hours: int = 12


def _ensure_ts(x) -> pd.Timestamp:
    return pd.to_datetime(x)


def detect_unknown_gaps(
    is_missing_bucket: pd.Series,
    *,
    freq: str,
    unknown_gap_min_hours: int = 12,
) -> pd.Series:
    """
    Detect long consecutive runs of missing buckets.

    Parameters
    ----------
    is_missing_bucket:
        Boolean/int Series indicating missing buckets (True/1 means missing).
    freq:
        Bucket frequency string, e.g. "1min", "5min", "15min".
    unknown_gap_min_hours:
        Minimum hours of consecutive missing buckets to be considered an unknown gap.

    Returns
    -------
    pd.Series[bool] aligned with input index.
    """
    is_m = pd.Series(is_missing_bucket).astype(bool)

    # identify consecutive runs
    run_id = (is_m != is_m.shift()).cumsum()

    step_minutes = int(pd.Timedelta(freq).total_seconds() / 60.0)
    min_len = int((int(unknown_gap_min_hours) * 60) / max(step_minutes, 1))

    return (is_m & (is_m.groupby(run_id).transform("sum") >= min_len)).astype(bool)


def mark_storm_gap(
    ts3: pd.DataFrame,
    *,
    time_col: str,
    freq: str,
    gap_cfg: GapConfig,
) -> pd.Series:
    """
    Mark storm gap window as a boolean Series.

    Returns
    -------
    pd.Series[bool] aligned with ts3 index.
    """
    t = pd.to_datetime(ts3[time_col], errors="coerce")
    ss = _ensure_ts(gap_cfg.storm_start).floor(freq)
    ee = _ensure_ts(gap_cfg.storm_end).floor(freq)
    return ((t >= ss) & (t < ee)).astype(bool)


def label_gaps_ts3(
    ts3: pd.DataFrame,
    *,
    time_col: str,
    hits_col: str,
    freq: str,
    gap_cfg: GapConfig,
) -> pd.DataFrame:
    """
    Adds:
      - is_missing_bucket
      - is_gap_storm
      - is_gap_unknown
      - is_gap
    """
    out = ts3.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")

    # missing bucket if hits is NaN (created by merge with full range)
    out["is_missing_bucket"] = out[hits_col].isna().astype("int8")

    # storm gap
    out["is_gap_storm"] = mark_storm_gap(out, time_col=time_col, freq=freq, gap_cfg=gap_cfg).astype("int8")

    # unknown gap: long consecutive missing buckets
    unk = detect_unknown_gaps(
        out["is_missing_bucket"],
        freq=freq,
        unknown_gap_min_hours=int(gap_cfg.unknown_gap_min_hours),
    )
    out["is_gap_unknown"] = unk.astype("int8")

    out["is_gap"] = ((out["is_gap_storm"] == 1) | (out["is_gap_unknown"] == 1)).astype("int8")
    return out