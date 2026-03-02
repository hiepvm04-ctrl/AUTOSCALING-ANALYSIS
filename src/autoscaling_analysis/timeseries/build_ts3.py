from __future__ import annotations

import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .gaps import GapConfig, label_gaps_ts3


FREQS = {"1m": "1min", "5m": "5min", "15m": "15min"}
FILL_COLS = ["hits", "bytes_sum", "avg_bytes_per_req", "err_4xx", "err_5xx", "error_rate", "unique_hosts"]


def _agg_raw_to_ts2(df_raw: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Aggregate parsed raw logs to buckets with stats:
      hits, bytes_sum, unique_hosts, err_4xx, err_5xx, avg_bytes_per_req, error_rate
    """
    d = df_raw[["datetime", "host", "status", "bytes"]].copy()
    d["datetime"] = pd.to_datetime(d["datetime"], utc=False, errors="coerce")
    bucket = d["datetime"].dt.floor(freq)

    st = pd.to_numeric(d["status"], errors="coerce")
    d["bytes_num"] = pd.to_numeric(d["bytes"], errors="coerce")

    g = d.assign(bucket_start=bucket).groupby("bucket_start", sort=True)

    idx = g.size().index
    ts2 = pd.DataFrame(
        {
            "bucket_start": idx,
            "hits": g.size().astype("int64").values,
            "bytes_sum": g["bytes_num"].sum(min_count=1).astype("float64").reindex(idx).values,
            "unique_hosts": g["host"].nunique().astype("int64").reindex(idx).values,
            "err_4xx": st.between(400, 499).groupby(bucket).sum().astype("int64").reindex(idx, fill_value=0).values,
            "err_5xx": st.between(500, 599).groupby(bucket).sum().astype("int64").reindex(idx, fill_value=0).values,
        }
    ).sort_values("bucket_start").reset_index(drop=True)

    ts2["avg_bytes_per_req"] = np.where(ts2["hits"] > 0, ts2["bytes_sum"] / ts2["hits"], 0.0)
    ts2["error_rate"] = np.where(ts2["hits"] > 0, (ts2["err_4xx"] + ts2["err_5xx"]) / ts2["hits"], 0.0)
    return ts2


def _to_ts3(ts2: pd.DataFrame, freq: str, gap_cfg: GapConfig) -> pd.DataFrame:
    s = ts2["bucket_start"].min()
    e = ts2["bucket_start"].max()

    # build full range and merge (introduces NaNs for missing buckets)
    ts_full = pd.DataFrame({"bucket_start": pd.date_range(s, e, freq=freq, tz=s.tz)})
    out = ts_full.merge(ts2, on="bucket_start", how="left")

    out = label_gaps_ts3(out, time_col="bucket_start", hits_col="hits", freq=freq, gap_cfg=gap_cfg)

    # Fill: non-gap NaNs -> 0, but gap rows stay NaN for FILL_COLS
    for c in FILL_COLS:
        out.loc[(out["is_gap"] == 0) & (out[c].isna()), c] = 0
        out.loc[out["is_gap"] == 1, c] = np.nan

    return out


def build_ts3_for_split(
    *,
    df_raw: pd.DataFrame,
    split: str,
    tags: List[str],
    out_dir: str,
    gap_cfg: Dict[str, Any],
) -> None:
    """
    Writes:
      {out_dir}/{split}/ts3_{tag}.parquet
    """
    out_split = os.path.join(out_dir, split)
    os.makedirs(out_split, exist_ok=True)

    gcfg = GapConfig(
        storm_start=pd.Timestamp(gap_cfg["storm_start"]),
        storm_end=pd.Timestamp(gap_cfg["storm_end"]),
        unknown_gap_min_hours=int(gap_cfg.get("unknown_gap_min_hours", 12)),
    )

    for tag in tags:
        if tag not in FREQS:
            raise ValueError(f"Unsupported tag: {tag}. Supported: {list(FREQS.keys())}")
        freq = FREQS[tag]

        ts2 = _agg_raw_to_ts2(df_raw, freq)
        ts3 = _to_ts3(ts2, freq, gcfg)

        p = os.path.join(out_split, f"ts3_{tag}.parquet")
        ts3.to_parquet(p, index=False)

        miss = int(ts3["is_missing_bucket"].sum())
        gap = int(ts3["is_gap"].sum())
        storm = int(ts3["is_gap_storm"].sum())
        unk = int(ts3["is_gap_unknown"].sum())

        print(
            f"[build_ts3] {split}/{tag} rows={len(ts3):,} range={ts3.bucket_start.min()} -> {ts3.bucket_start.max()} "
            f"missing={miss:,} gap={gap:,} (storm={storm:,}, unknown={unk:,}) saved={p}"
        )