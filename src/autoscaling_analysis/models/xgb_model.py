# src/autoscaling_analysis/models/xgb_model.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from .metrics import compute_metrics, write_metrics_long
from autoscaling_analysis.features.transforms import tag_minutes


@dataclass
class XGBTrainResult:
    """
    Compatibility result object (exported by models/__init__.py).

    We keep existing return style from train_xgb_one (df, metrics_dict) so scripts remain unchanged.
    This dataclass is here so external callers can use a richer structured output if they want.
    """
    tag: str
    target: str
    feature_cols: List[str]
    model_path: str
    pred_csv_path: str
    pred_parquet_path: str
    cv_mean: Dict[str, float]
    test_metrics: Dict[str, float]
    eval_df: pd.DataFrame


def _load_meta(feature_dir: str, tag: str) -> Dict[str, Any]:
    p = os.path.join(feature_dir, f"meta_{tag}.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _tz_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(None)
    return s


def train_xgb_one(
    *,
    feature_dir: str,
    tag: str,
    target: str,
    model_out_dir: str,
    pred_out_dir: str,
    metrics_csv: str,
    model_cfg: Dict[str, Any],
    cv_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    os.makedirs(model_out_dir, exist_ok=True)
    os.makedirs(pred_out_dir, exist_ok=True)

    meta = _load_meta(feature_dir, tag)
    TIME_COL = meta["time_col"]

    train = pd.read_parquet(os.path.join(feature_dir, f"xgb_train_{tag}.parquet"))
    testf = pd.read_parquet(os.path.join(feature_dir, f"xgb_test_features_{tag}.parquet"))

    train[TIME_COL] = _tz_naive(train[TIME_COL])
    testf[TIME_COL] = _tz_naive(testf[TIME_COL])

    train = train.sort_values(TIME_COL).reset_index(drop=True)
    testf = testf.sort_values(TIME_COL).reset_index(drop=True)

    if target == "hits":
        FEAT_COLS = list(meta["hits_feature_cols"])
        LABEL = meta["labels"]["hits"]
        TRUE_COL = "hits"
        use_log = False
    else:
        FEAT_COLS = list(meta["bytes_feature_cols"])
        LABEL = meta["labels"]["bytes_sum"]
        TRUE_COL = "bytes_sum"
        use_log = True

    # Safety: don't ever include these as features
    FEAT_COLS = [c for c in FEAT_COLS if c not in (TIME_COL, TRUE_COL, LABEL)]

    # CV sizing like notebook
    freq_min = tag_minutes(tag)
    test_days = int(cv_cfg.get("test_days", 2))
    gap_steps = int(cv_cfg.get("gap_steps", 1))
    n_splits = int(cv_cfg.get("splits", 5))

    test_size = int(test_days * 24 * 60 / freq_min)
    n = len(train)
    max_splits = (n - gap_steps) // max(test_size, 1) - 1
    n_splits_eff = int(min(n_splits, max(0, max_splits)))

    cv_metrics: List[Dict[str, float]] = []
    if n_splits_eff >= 2:
        tss = TimeSeriesSplit(n_splits=n_splits_eff, test_size=test_size, gap=gap_steps)
        for _, (tr_idx, va_idx) in enumerate(tss.split(train)):
            tr = train.iloc[tr_idx]
            va = train.iloc[va_idx]

            X_tr, X_va = tr[FEAT_COLS], va[FEAT_COLS]
            y_tr = tr[LABEL].astype(float).values
            y_va = va[LABEL].astype(float).values

            if use_log:
                y_tr_fit = np.log1p(np.maximum(y_tr, 0.0))
                y_va_fit = np.log1p(np.maximum(y_va, 0.0))
            else:
                y_tr_fit, y_va_fit = y_tr, y_va

            reg = xgb.XGBRegressor(**model_cfg)
            reg.fit(X_tr, y_tr_fit, eval_set=[(X_va, y_va_fit)], verbose=False)

            pred_fit = reg.predict(X_va)
            pred = np.expm1(pred_fit) if use_log else pred_fit
            pred = np.maximum(pred, 0.0)

            cv_metrics.append(compute_metrics(y_va, pred, target=target))

    cv_mean = {
        k: float(np.mean([m[k] for m in cv_metrics])) if cv_metrics else np.nan
        for k in ["RMSE", "MSE", "MAE", "MAPE"]
    }

    # retrain full (disable early stopping for final)
    final_params = dict(model_cfg)
    final_params.pop("early_stopping_rounds", None)

    X_all = train[FEAT_COLS]
    y_all = train[LABEL].astype(float).values
    y_fit = np.log1p(np.maximum(y_all, 0.0)) if use_log else y_all

    model = xgb.XGBRegressor(**final_params)
    model.fit(X_all, y_fit, eval_set=[(X_all, y_fit)], verbose=False)

    # save model + feature list
    model_path = os.path.join(model_out_dir, f"model_xgb_{target}_{tag}.json")
    model.get_booster().save_model(model_path)

    with open(os.path.join(model_out_dir, f"feat_cols_xgb_{target}_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(FEAT_COLS, f, ensure_ascii=False, indent=2)

    # predict test, evaluate y_{t+1}
    df = testf[[TIME_COL, TRUE_COL] + FEAT_COLS].copy().sort_values(TIME_COL).reset_index(drop=True)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if isinstance(df[TRUE_COL], pd.DataFrame):
        df[TRUE_COL] = df[TRUE_COL].iloc[:, 0]

    df["true_next"] = pd.to_numeric(df[TRUE_COL], errors="coerce").astype(float).shift(-1)
    eval_df = df[df["true_next"].notna()].copy()

    csv_path = os.path.join(pred_out_dir, f"pred_{target}_{tag}_xgb.csv")
    pq_path = os.path.join(pred_out_dir, f"pred_{target}_{tag}_xgb.parquet")

    if len(eval_df) == 0:
        out0 = df[[TIME_COL, TRUE_COL, "true_next"]].head(0).copy()
        out0["pred"] = np.nan
        out0.to_csv(csv_path, index=False, encoding="utf-8-sig")
        out0.to_parquet(pq_path, index=False)

        test_m = {"RMSE": np.nan, "MSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    else:
        pred_fit = model.predict(eval_df[FEAT_COLS])
        pred = np.expm1(pred_fit) if use_log else pred_fit
        pred = np.maximum(pred, 0.0)
        eval_df["pred"] = pred

        eval_df[[TIME_COL, TRUE_COL, "true_next", "pred"]].to_csv(csv_path, index=False, encoding="utf-8-sig")
        eval_df[[TIME_COL, TRUE_COL, "true_next", "pred"]].to_parquet(pq_path, index=False)

        test_m = compute_metrics(eval_df["true_next"].values, eval_df["pred"].values, target=target)

    # write metrics long
    rows = []
    for split, metrics in [("cv_mean", cv_mean), ("test", test_m)]:
        for metric_name, v in metrics.items():
            rows.append(
                {
                    "model": "xgb",
                    "target": target,
                    "window": tag,
                    "split": split,
                    "metric": metric_name,
                    "value": float(v) if v is not None else np.nan,
                }
            )
    write_metrics_long(metrics_csv, rows)

    return (eval_df[[TIME_COL, TRUE_COL, "true_next", "pred"]] if len(eval_df) else pd.DataFrame()), test_m


def train_xgb_all(
    *,
    feature_dir: str,
    tags: List[str],
    targets: List[str],
    model_out_dir: str,
    pred_out_dir: str,
    metrics_csv: str,
    model_cfg: Dict[str, Any],
    cv_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    out = []
    for target in targets:
        for tag in tags:
            _, tm = train_xgb_one(
                feature_dir=feature_dir,
                tag=tag,
                target=target,
                model_out_dir=model_out_dir,
                pred_out_dir=pred_out_dir,
                metrics_csv=metrics_csv,
                model_cfg=model_cfg,
                cv_cfg=cv_cfg,
            )
            out.append({"model": "xgb", "target": target, "window": tag, **tm})
            print(f"[xgb] ✅ {target}/{tag} TEST:", tm)
    return out