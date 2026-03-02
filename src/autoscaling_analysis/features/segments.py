# src/autoscaling_analysis/features/segments.py

from __future__ import annotations

import pandas as pd


def build_segment_id(df: pd.DataFrame, *, gap_col: str = "is_gap", seg_col: str = "segment_id") -> pd.DataFrame:
    """
    Assign segment_id to contiguous non-gap regions.
    Gap rows get segment_id = -1.
    """
    d = df.copy()
    is_gap = pd.to_numeric(d[gap_col], errors="coerce").fillna(0).astype("int8")
    d[gap_col] = is_gap

    is_ok = is_gap == 0
    prev_gap = is_gap.shift(1).fillna(1).astype("int8")
    new_seg = (is_ok & (prev_gap == 1)).astype("int8")
    seg = new_seg.cumsum().astype("int32")

    d[seg_col] = seg.where(is_ok, other=-1).astype("int32")
    return d