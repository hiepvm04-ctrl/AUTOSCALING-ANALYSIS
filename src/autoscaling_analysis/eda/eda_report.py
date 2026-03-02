# src/autoscaling_analysis/eda/eda_report.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .plots import (
    plot_timeseries,
    plot_hist,
    plot_heatmap_dow_hour,
    plot_mean_by_hour,
    plot_mean_by_dow,
    plot_corr_lower_triangle,
    plot_acf_simple,
    plot_overlay_anomalies,
)


def tag_minutes(tag: str) -> int:
    return {"1m": 1, "5m": 5, "15m": 15}[tag]


def steps_per_day(tag: str) -> int:
    return int(24 * 60 / tag_minutes(tag))


def pct(s: pd.Series, p: float) -> float:
    return float(np.nanpercentile(pd.to_numeric(s, errors="coerce"), p))


def load_ts3(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["bucket_start"] = pd.to_datetime(df["bucket_start"], utc=False, errors="coerce")
    return df


def _tail_row(ts3: pd.DataFrame, label: str) -> pd.DataFrame:
    x = ts3[pd.to_numeric(ts3["is_gap"], errors="coerce").fillna(0).astype(int) == 0].copy()

    return pd.DataFrame(
        [
            {
                "label": label,
                "rows": len(x),
                "hits_mean": float(np.nanmean(pd.to_numeric(x["hits"], errors="coerce"))),
                "hits_med": float(np.nanmedian(pd.to_numeric(x["hits"], errors="coerce"))),
                "hits_p95": pct(x["hits"], 95),
                "hits_p99": pct(x["hits"], 99),
                "hits_max": float(np.nanmax(pd.to_numeric(x["hits"], errors="coerce"))),
                "bytes_mean": float(np.nanmean(pd.to_numeric(x["bytes_sum"], errors="coerce"))),
                "bytes_p95": pct(x["bytes_sum"], 95),
                "bytes_p99": pct(x["bytes_sum"], 99),
                "bytes_max": float(np.nanmax(pd.to_numeric(x["bytes_sum"], errors="coerce"))),
                "err_mean": float(np.nanmean(pd.to_numeric(x["error_rate"], errors="coerce"))),
                "err_p95": pct(x["error_rate"], 95),
                "err_max": float(np.nanmax(pd.to_numeric(x["error_rate"], errors="coerce"))),
                "hosts_mean": float(np.nanmean(pd.to_numeric(x["unique_hosts"], errors="coerce"))),
                "hosts_p95": pct(x["unique_hosts"], 95),
            }
        ]
    )


def _acf_at_lag(series: pd.Series, lag: int) -> float:
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(float).values
    s = s - s.mean()
    denom = float(np.dot(s, s)) if float(np.dot(s, s)) != 0 else 1.0
    if lag >= len(s):
        return float("nan")
    return float(np.dot(s[:-lag], s[lag:]) / denom)


def _lag_corr(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    s = pd.to_numeric(df[col], errors="coerce").astype(float)
    return pd.DataFrame([{"metric": col, "lag": L, "corr": float(s.corr(s.shift(L)))} for L in lags])


def run_eda_compact(
    *,
    train_paths_by_tag: Dict[str, str],
    test_paths_by_tag: Dict[str, str],
    main_tag: str = "5m",
    out_eda_dir: str = "reports/eda",
    out_fig_dir: str = "reports/figures",
) -> Dict[str, Any]:
    """
    Generate EDA artifacts.
    Returns a small dict of key outputs for logs/CLI.
    """
    os.makedirs(out_eda_dir, exist_ok=True)
    os.makedirs(out_fig_dir, exist_ok=True)

    if main_tag not in train_paths_by_tag or main_tag not in test_paths_by_tag:
        raise ValueError(f"main_tag={main_tag} missing from provided paths")

    # -------------------------
    # 1) OVERVIEW (MAIN)
    # -------------------------
    tr0 = load_ts3(train_paths_by_tag[main_tag])
    te0 = load_ts3(test_paths_by_tag[main_tag])

    tr = tr0[pd.to_numeric(tr0["is_gap"], errors="coerce").fillna(0).astype(int) == 0].copy()
    te = te0[pd.to_numeric(te0["is_gap"], errors="coerce").fillna(0).astype(int) == 0].copy()

    base_cols = ["hits", "bytes_sum", "avg_bytes_per_req", "err_4xx", "err_5xx", "error_rate", "unique_hosts"]
    for df in (tr, te):
        for c in base_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["rate_4xx"] = np.where(df["hits"] > 0, df["err_4xx"] / df["hits"], 0.0)
        df["rate_5xx"] = np.where(df["hits"] > 0, df["err_5xx"] / df["hits"], 0.0)

    bucket_min = tag_minutes(main_tag)

    # -------------------------
    # 2) TAIL-RISK (ALL TAGS)
    # -------------------------
    all_tags = sorted(train_paths_by_tag.keys())
    train_stats = pd.concat([_tail_row(load_ts3(train_paths_by_tag[k]), f"train_{k}") for k in all_tags], ignore_index=True)
    test_stats = pd.concat([_tail_row(load_ts3(test_paths_by_tag[k]), f"test_{k}") for k in all_tags], ignore_index=True)

    train_stats.to_csv(os.path.join(out_eda_dir, "tail_risk_train.csv"), index=False)
    test_stats.to_csv(os.path.join(out_eda_dir, "tail_risk_test.csv"), index=False)

    # -------------------------
    # 3) TREND + DISTRIBUTION (MAIN TRAIN)
    # -------------------------
    figs = []
    figs.append((plot_timeseries(tr, "bucket_start", "hits", f"Train | {main_tag} | Hits per bucket", ylabel="hits"),
                 "trend_hits.png"))
    figs.append((plot_timeseries(tr, "bucket_start", "bytes_sum", f"Train | {main_tag} | Bytes sum per bucket", ylabel="bytes_sum"),
                 "trend_bytes_sum.png"))
    figs.append((plot_hist(tr["hits"], f"Train | {main_tag} | Distribution of hits per bucket", "hits"),
                 "hist_hits.png"))
    figs.append((plot_hist(tr["bytes_sum"], f"Train | {main_tag} | Distribution of bytes_sum per bucket", "bytes_sum"),
                 "hist_bytes_sum.png"))

    # -------------------------
    # 4) SEASONALITY (MAIN TRAIN) + ACF peaks 12h/24h
    # -------------------------
    tr["hour"] = tr["bucket_start"].dt.hour
    tr["dow"] = tr["bucket_start"].dt.dayofweek

    heat = (
        tr.groupby(["dow", "hour"])["hits"].mean().unstack("hour")
        .reindex(index=range(7), columns=range(24)).fillna(0)
    )

    figs.append((plot_heatmap_dow_hour(heat, f"Train | {main_tag} | Mean hits heatmap (DOW x hour)"),
                 "seasonality_heatmap_hits.png"))
    figs.append((plot_mean_by_hour(tr, "hour", "hits", f"Seasonality | Train | {main_tag} | Mean hits by hour-of-day"),
                 "seasonality_mean_hits_by_hour.png"))
    figs.append((plot_mean_by_dow(tr, "dow", "hits", f"Seasonality | Train | {main_tag} | Mean hits by day-of-week"),
                 "seasonality_mean_hits_by_dow.png"))

    lag_12h = int((12 * 60) / bucket_min)
    lag_24h = int((24 * 60) / bucket_min)

    acf_peaks = pd.DataFrame(
        [
            {"metric": "hits", "lag": "12h", "lag_steps": lag_12h, "acf": _acf_at_lag(tr["hits"], lag_12h)},
            {"metric": "hits", "lag": "24h", "lag_steps": lag_24h, "acf": _acf_at_lag(tr["hits"], lag_24h)},
        ]
    )
    acf_peaks.to_csv(os.path.join(out_eda_dir, "acf_peaks_hits.csv"), index=False)

    # -------------------------
    # 5) CORRELATION (LOWER TRIANGLE)
    # -------------------------
    corr_cols = ["hits", "bytes_sum", "avg_bytes_per_req", "unique_hosts", "error_rate", "rate_5xx"]
    figs.append((plot_corr_lower_triangle(tr, corr_cols, f"Train | {main_tag} | Correlation (lower triangle)"),
                 "corr_lower_triangle.png"))

    # -------------------------
    # 6) LAG + ACF
    # -------------------------
    LAGS = [1, 2, 3, 6, 12, 24, 36, 48]
    lag_tbl = pd.concat([_lag_corr(tr, "hits", LAGS), _lag_corr(tr, "bytes_sum", LAGS)], ignore_index=True).round(4)
    lag_tbl.to_csv(os.path.join(out_eda_dir, "lag_corr_table.csv"), index=False)

    figs.append((plot_acf_simple(tr["hits"], 60, f"ACF | Train | {main_tag} | hits", bucket_min=bucket_min),
                 "acf_hits.png"))
    figs.append((plot_acf_simple(tr["bytes_sum"], 60, f"ACF | Train | {main_tag} | bytes_sum", bucket_min=bucket_min),
                 "acf_bytes_sum.png"))

    # -------------------------
    # 7) SPIKE LABELING (COUNTS ONLY)
    # -------------------------
    x = tr.copy()
    p99_hits = np.nanpercentile(x["hits"], 99)
    p99_bytes = np.nanpercentile(x["bytes_sum"], 99)
    p99_err = np.nanpercentile(x["error_rate"], 99)
    p99_abpr = np.nanpercentile(x["avg_bytes_per_req"], 99)

    x["is_spike_hits"] = (x["hits"] > p99_hits).astype(int)
    x["is_spike_bytes"] = (x["bytes_sum"] > p99_bytes).astype(int)
    x["is_spike_err"] = (x["error_rate"] > p99_err).astype(int)

    x["spike_type"] = "normal"
    x.loc[x["is_spike_err"] == 1, "spike_type"] = "error_incident"
    x.loc[(x["is_spike_hits"] == 1) & (x["unique_hosts"] > np.nanpercentile(x["unique_hosts"], 95)) & (x["is_spike_err"] == 0), "spike_type"] = "flash_crowd"
    x.loc[(x["is_spike_hits"] == 1) & (x["unique_hosts"] <= np.nanpercentile(x["unique_hosts"], 50)) & (x["is_spike_err"] == 0), "spike_type"] = "bot_crawler"
    x.loc[(x["is_spike_bytes"] == 1) & (x["hits"] <= np.nanpercentile(x["hits"], 75)) & (x["avg_bytes_per_req"] > p99_abpr), "spike_type"] = "bandwidth_spike"

    spike_counts = x["spike_type"].value_counts().rename_axis("spike_type").reset_index(name="count")
    spike_counts.to_csv(os.path.join(out_eda_dir, "spike_type_counts_train.csv"), index=False)

    # -------------------------
    # 8) ANOMALY DETECTION (IsolationForest)
    # -------------------------
    feat = x.assign(hour=x["bucket_start"].dt.hour, dow=x["bucket_start"].dt.dayofweek)[
        ["hits", "bytes_sum", "avg_bytes_per_req", "unique_hosts", "error_rate", "rate_4xx", "rate_5xx", "hour", "dow"]
    ].fillna(0)

    Xs = StandardScaler().fit_transform(feat)
    iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=42, n_jobs=-1).fit(Xs)
    score = -iso.score_samples(Xs)
    thr = float(np.percentile(score, 99.0))

    tr_if = x.copy()
    tr_if["if_score"] = score
    tr_if["if_is_anomaly"] = (score >= thr).astype("int8")

    anom_out = tr_if.loc[tr_if["if_is_anomaly"] == 1, ["bucket_start", "if_score", "hits", "bytes_sum", "error_rate", "unique_hosts", "rate_4xx", "rate_5xx"]] \
        .sort_values("if_score", ascending=False)
    anom_out.to_csv(os.path.join(out_eda_dir, "anomalies_isoforest_train.csv"), index=False)

    figs.append((plot_overlay_anomalies(tr_if, "bucket_start", "hits", "if_is_anomaly",
                                        f"Train | {main_tag} | IsolationForest overlay: hits", "hits"),
                 "if_overlay_hits.png"))
    figs.append((plot_overlay_anomalies(tr_if, "bucket_start", "bytes_sum", "if_is_anomaly",
                                        f"Train | {main_tag} | IsolationForest overlay: bytes_sum", "bytes_sum"),
                 "if_overlay_bytes_sum.png"))
    figs.append((plot_overlay_anomalies(tr_if, "bucket_start", "error_rate", "if_is_anomaly",
                                        f"Train | {main_tag} | IsolationForest overlay: error_rate", "error_rate"),
                 "if_overlay_error_rate.png"))

    # -------------------------
    # 9) DRIFT (TAIL TRAIN VS TEST)
    # -------------------------
    drift = pd.DataFrame(
        [
            {"metric": "hits", "train_p95": pct(tr["hits"], 95), "train_p99": pct(tr["hits"], 99), "train_max": float(np.nanmax(tr["hits"])),
             "test_p95": pct(te["hits"], 95), "test_p99": pct(te["hits"], 99), "test_max": float(np.nanmax(te["hits"]))},
            {"metric": "bytes_sum", "train_p95": pct(tr["bytes_sum"], 95), "train_p99": pct(tr["bytes_sum"], 99), "train_max": float(np.nanmax(tr["bytes_sum"])),
             "test_p95": pct(te["bytes_sum"], 95), "test_p99": pct(te["bytes_sum"], 99), "test_max": float(np.nanmax(te["bytes_sum"]))},
            {"metric": "error_rate", "train_p95": pct(tr["error_rate"], 95), "train_p99": pct(tr["error_rate"], 99), "train_max": float(np.nanmax(tr["error_rate"])),
             "test_p95": pct(te["error_rate"], 95), "test_p99": pct(te["error_rate"], 99), "test_max": float(np.nanmax(te["error_rate"]))},
            {"metric": "unique_hosts", "train_p95": pct(tr["unique_hosts"], 95), "train_p99": pct(tr["unique_hosts"], 99), "train_max": float(np.nanmax(tr["unique_hosts"])),
             "test_p95": pct(te["unique_hosts"], 95), "test_p99": pct(te["unique_hosts"], 99), "test_max": float(np.nanmax(te["unique_hosts"]))},
            {"metric": "avg_bytes_per_req", "train_p95": pct(tr["avg_bytes_per_req"], 95), "train_p99": pct(tr["avg_bytes_per_req"], 99), "train_max": float(np.nanmax(tr["avg_bytes_per_req"])),
             "test_p95": pct(te["avg_bytes_per_req"], 95), "test_p99": pct(te["avg_bytes_per_req"], 99), "test_max": float(np.nanmax(te["avg_bytes_per_req"]))},
        ]
    ).round(4)
    drift.to_csv(os.path.join(out_eda_dir, "tail_drift_train_vs_test.csv"), index=False)

    # -------------------------
    # Save figures
    # -------------------------
    for fig, name in figs:
        fig.savefig(os.path.join(out_fig_dir, name), dpi=160, bbox_inches="tight")

    return {
        "main_tag": main_tag,
        "bucket_min": bucket_min,
        "train_rows_main": int(len(tr)),
        "test_rows_main": int(len(te)),
        "figures_saved": len(figs),
        "eda_dir": out_eda_dir,
        "fig_dir": out_fig_dir,
    }