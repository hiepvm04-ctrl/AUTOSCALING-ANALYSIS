# src/autoscaling_analysis/features/make_features.py

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .segments import build_segment_id
from .transforms import (
    add_ratio_features,
    create_time_features,
    add_lags,
    add_rolling,
    add_labels,
    to_tznaive_dt64,
)


def _assert_required_cols(df: pd.DataFrame, cols: List[str], name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[{name}] missing required cols: {miss}")


def _load_ts3(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["bucket_start"] = pd.to_datetime(df["bucket_start"], utc=False, errors="coerce")
    return df


def build_features_for_tag(
    *,
    tag: str,
    train_ts3_path: str,
    test_ts3_path: str,
    out_dir: str,
    feature_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Writes into out_dir:
      - xgb_train_{tag}.parquet
      - xgb_test_{tag}.parquet              (truth)
      - xgb_test_features_{tag}.parquet     (features + current obs)
      - meta_{tag}.json
    """
    os.makedirs(out_dir, exist_ok=True)

    TIME_COL = feature_cfg.get("time_col", "bucket_start")
    GAP_COL = feature_cfg.get("gap_col", "is_gap")
    SEG_COL = feature_cfg.get("segment_col", "segment_id")

    require_cols = feature_cfg.get("require_cols", ["bucket_start", "hits", "bytes_sum", "is_gap"])

    lag_days = list(feature_cfg.get("lag_days", [1, 2, 3, 4, 5, 6, 7]))
    roll_windows = list(feature_cfg.get("roll_windows", ["1h", "6h", "1d"]))
    roll_use_std = bool(feature_cfg.get("roll_use_std", True))
    use_cyclic = bool(feature_cfg.get("use_cyclic", True))
    horizon_steps = int(feature_cfg.get("horizon_steps", 1))

    keep_raw_extra = list(feature_cfg.get("keep_raw_extra", []))

    tr = _load_ts3(train_ts3_path)
    te = _load_ts3(test_ts3_path)

    _assert_required_cols(tr, require_cols, f"train_{tag}")
    _assert_required_cols(te, require_cols, f"test_{tag}")

    tr = build_segment_id(tr, gap_col=GAP_COL, seg_col=SEG_COL)
    te = build_segment_id(te, gap_col=GAP_COL, seg_col=SEG_COL)

    # Train excludes gaps
    tr_clean = tr[pd.to_numeric(tr[GAP_COL], errors="coerce").fillna(0).astype(int) == 0].copy()
    tr_clean = add_ratio_features(tr_clean)

    ref_time = pd.to_datetime(pd.concat([tr[[TIME_COL]], te[[TIME_COL]]], ignore_index=True)[TIME_COL].min())
    tr_clean, time_cols = create_time_features(
        tr_clean, tag=tag, time_col=TIME_COL, ref_time=ref_time, use_cyclic=use_cyclic
    )

    tr_clean, hits_lag = add_lags(
        tr_clean, tag=tag, target="hits", time_col=TIME_COL, seg_col=SEG_COL, lag_days=lag_days
    )
    tr_clean, hits_roll = add_rolling(
        tr_clean,
        tag=tag,
        target="hits",
        time_col=TIME_COL,
        seg_col=SEG_COL,
        roll_windows=roll_windows,
        use_std=roll_use_std,
    )

    tr_clean, bytes_lag = add_lags(
        tr_clean, tag=tag, target="bytes_sum", time_col=TIME_COL, seg_col=SEG_COL, lag_days=lag_days
    )
    tr_clean, bytes_roll = add_rolling(
        tr_clean,
        tag=tag,
        target="bytes_sum",
        time_col=TIME_COL,
        seg_col=SEG_COL,
        roll_windows=roll_windows,
        use_std=roll_use_std,
    )

    tr_clean, ratio_lag = add_lags(
        tr_clean, tag=tag, target="avg_bytes_per_req", time_col=TIME_COL, seg_col=SEG_COL, lag_days=lag_days
    )
    tr_clean, ratio_roll = add_rolling(
        tr_clean,
        tag=tag,
        target="avg_bytes_per_req",
        time_col=TIME_COL,
        seg_col=SEG_COL,
        roll_windows=roll_windows,
        use_std=roll_use_std,
    )

    tr_clean = add_labels(
        tr_clean,
        target="hits",
        label_col="y_hits_next",
        horizon_steps=horizon_steps,
        time_col=TIME_COL,
        seg_col=SEG_COL,
    )
    tr_clean = add_labels(
        tr_clean,
        target="bytes_sum",
        label_col="y_bytes_sum_next",
        horizon_steps=horizon_steps,
        time_col=TIME_COL,
        seg_col=SEG_COL,
    )

    keep_extra = [c for c in keep_raw_extra if c in tr_clean.columns]

    hits_feat_cols = time_cols + hits_lag + hits_roll
    bytes_feat_cols = time_cols + bytes_lag + bytes_roll + ratio_roll + hits_roll
    all_feat_cols = sorted(set(hits_feat_cols + bytes_feat_cols))

    keep_cols_train = (
        [TIME_COL, GAP_COL, SEG_COL, "hits", "bytes_sum", "avg_bytes_per_req"]
        + keep_extra
        + ["y_hits_next", "y_bytes_sum_next"]
        + all_feat_cols
    )

    before = len(tr_clean)
    tr_out = tr_clean[keep_cols_train].copy()
    tr_out = tr_out.dropna(subset=["y_hits_next", "y_bytes_sum_next"]).reset_index(drop=True)
    after = len(tr_out)

    train_out_path = os.path.join(out_dir, f"xgb_train_{tag}.parquet")
    tr_out.to_parquet(train_out_path, index=False)

    # --- Test features: concat history + test -> compute features -> slice test back with tz-safe join
    hist_and_test = pd.concat([tr, te], ignore_index=True).sort_values(TIME_COL).reset_index(drop=True)
    hist_and_test = build_segment_id(hist_and_test, gap_col=GAP_COL, seg_col=SEG_COL)
    hist_and_test = add_ratio_features(hist_and_test)
    hist_and_test, _ = create_time_features(
        hist_and_test, tag=tag, time_col=TIME_COL, ref_time=ref_time, use_cyclic=use_cyclic
    )

    hist_and_test, _ = add_lags(
        hist_and_test, tag=tag, target="hits", time_col=TIME_COL, seg_col=SEG_COL, lag_days=lag_days
    )
    hist_and_test, _ = add_rolling(
        hist_and_test,
        tag=tag,
        target="hits",
        time_col=TIME_COL,
        seg_col=SEG_COL,
        roll_windows=roll_windows,
        use_std=roll_use_std,
    )
    hist_and_test, _ = add_lags(
        hist_and_test, tag=tag, target="bytes_sum", time_col=TIME_COL, seg_col=SEG_COL, lag_days=lag_days
    )
    hist_and_test, _ = add_rolling(
        hist_and_test,
        tag=tag,
        target="bytes_sum",
        time_col=TIME_COL,
        seg_col=SEG_COL,
        roll_windows=roll_windows,
        use_std=roll_use_std,
    )
    hist_and_test, _ = add_lags(
        hist_and_test,
        tag=tag,
        target="avg_bytes_per_req",
        time_col=TIME_COL,
        seg_col=SEG_COL,
        lag_days=lag_days,
    )
    hist_and_test, _ = add_rolling(
        hist_and_test,
        tag=tag,
        target="avg_bytes_per_req",
        time_col=TIME_COL,
        seg_col=SEG_COL,
        roll_windows=roll_windows,
        use_std=roll_use_std,
    )

    # tz normalize
    hist_and_test[TIME_COL] = to_tznaive_dt64(hist_and_test[TIME_COL])
    te[TIME_COL] = to_tznaive_dt64(te[TIME_COL])

    hist_and_test["_tkey"] = hist_and_test[TIME_COL].view("int64")
    te_key = te[[TIME_COL]].copy()
    te_key["_tkey"] = te_key[TIME_COL].view("int64")

    if hist_and_test["_tkey"].duplicated().any():
        dup_n = int(hist_and_test["_tkey"].duplicated().sum())
        raise ValueError(f"[make_features] hist_and_test has duplicate timestamps: {dup_n}")

    te_features = (
        te_key.merge(
            hist_and_test.drop(columns=[TIME_COL]),
            on="_tkey",
            how="left",
            validate="one_to_one",
        )
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )

    te_features.drop(columns=["_tkey"], inplace=True)

    # sanity check
    for c in ["hits", "bytes_sum"]:
        na_rate = float(te_features[c].isna().mean())
        if na_rate > 0.01:
            raise ValueError(f"[make_features] join failed: te_features[{c}] NA rate = {na_rate:.4f}")

    keep_cols_test_feat = [TIME_COL, GAP_COL, SEG_COL, "hits", "bytes_sum", "avg_bytes_per_req"] + all_feat_cols
    test_feat_out_path = os.path.join(out_dir, f"xgb_test_features_{tag}.parquet")
    te_features[keep_cols_test_feat].to_parquet(test_feat_out_path, index=False)

    te_truth = te[[TIME_COL, "hits", "bytes_sum", GAP_COL, SEG_COL]].copy()
    te_truth = te_truth.rename(columns={"hits": "hits_true", "bytes_sum": "bytes_sum_true"})
    test_truth_out_path = os.path.join(out_dir, f"xgb_test_{tag}.parquet")
    te_truth.to_parquet(test_truth_out_path, index=False)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "time_col": TIME_COL,
        "gap_col": GAP_COL,
        "segment_col": SEG_COL,
        "horizon_steps": horizon_steps,
        "labels": {"hits": "y_hits_next", "bytes_sum": "y_bytes_sum_next"},
        "time_features": time_cols,
        "hits_feature_cols": hits_feat_cols,
        "bytes_feature_cols": bytes_feat_cols,
        "all_feature_cols": all_feat_cols,
        "paths": {
            "train_in": train_ts3_path,
            "test_in": test_ts3_path,
            "train_out": train_out_path,
            "test_truth_out": test_truth_out_path,
            "test_features_out": test_feat_out_path,
        },
        "spec": {
            "gap_policy": "train excludes is_gap==1; segment-safe; keep NaN in features; drop NaN only in labels",
            "lag_days": lag_days,
            "roll_windows": roll_windows,
        },
    }

    meta_path = os.path.join(out_dir, f"meta_{tag}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "tag": tag,
        "rows_out_train": int(after),
        "dropped_na_train_labels": int(before - after),
        "n_features": int(len(all_feat_cols)),
        "train_out": train_out_path,
        "test_features_out": test_feat_out_path,
        "test_truth_out": test_truth_out_path,
        "meta_out": meta_path,
    }


# -------------------------------------------------------------------
# NEW: Backward-compatible name expected by features/__init__.py
# -------------------------------------------------------------------
def build_feature_frames_for_tag(
    *,
    tag: str,
    train_ts3_path: str,
    test_ts3_path: str,
    out_dir: str,
    feature_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compatibility wrapper.
    Older code / __init__.py exports build_feature_frames_for_tag.
    Internally we use build_features_for_tag.
    """
    return build_features_for_tag(
        tag=tag,
        train_ts3_path=train_ts3_path,
        test_ts3_path=test_ts3_path,
        out_dir=out_dir,
        feature_cfg=feature_cfg,
    )


def build_features_all_tags(
    *,
    train_paths_by_tag: Dict[str, str],
    test_paths_by_tag: Dict[str, str],
    out_dir: str,
    feature_cfg: Dict[str, Any],
    tags: List[str],
) -> pd.DataFrame:
    rows = []
    for tag in tags:
        r = build_features_for_tag(
            tag=tag,
            train_ts3_path=train_paths_by_tag[tag],
            test_ts3_path=test_paths_by_tag[tag],
            out_dir=out_dir,
            feature_cfg=feature_cfg,
        )
        rows.append(r)

    return pd.DataFrame(rows).sort_values("tag").reset_index(drop=True)