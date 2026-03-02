#!/usr/bin/env python3
# scripts/train.py

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from autoscaling_analysis.config import load_config
from autoscaling_analysis.models.xgb_model import train_xgb_all
from autoscaling_analysis.models.seasonal_naive import run_seasonal_naive_all


def _ensure_dirs(cfg: Dict[str, Any]) -> None:
    art = Path(cfg["paths"]["artifacts_dir"])
    (art / "models").mkdir(parents=True, exist_ok=True)
    (art / "predictions").mkdir(parents=True, exist_ok=True)
    (art / "metrics").mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Train models + export predictions + metrics_forecast.csv")
    ap.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    _ensure_dirs(cfg)

    feat_dir = Path(cfg["paths"]["data_processed"]) / "features"
    if not feat_dir.exists():
        raise FileNotFoundError(f"Missing features dir: {feat_dir}. Run scripts/features.py first.")

    artifacts = Path(cfg["paths"]["artifacts_dir"])
    models_dir = artifacts / "models"
    pred_dir = artifacts / "predictions"
    metrics_dir = artifacts / "metrics"
    metrics_path = metrics_dir / "metrics_forecast.csv"

    tags = cfg.get("tags", ["1m", "5m", "15m"])
    targets = cfg.get("targets", ["hits", "bytes_sum"])

    # 1) XGB
    print("[train] Running XGB...")
    xgb_metrics = train_xgb_all(
        feature_dir=str(feat_dir),
        tags=tags,
        targets=targets,
        model_out_dir=str(models_dir),
        pred_out_dir=str(pred_dir),
        metrics_csv=str(metrics_path),
        model_cfg=cfg["modeling"]["xgb"],
        cv_cfg=cfg["modeling"]["cv"],
    )
    print(pd.DataFrame(xgb_metrics).sort_values(["target", "window"]).to_string(index=False))

    # 2) Seasonal naive
    print("[train] Running seasonal_naive...")
    sn_metrics = run_seasonal_naive_all(
        ts3_root=str(Path(cfg["paths"]["data_processed"]) / "ts3"),
        tags=tags,
        targets=targets,
        pred_out_dir=str(pred_dir),
        metrics_csv=str(metrics_path),
    )
    print(pd.DataFrame(sn_metrics).sort_values(["target", "window"]).to_string(index=False))

    print(f"[train] ✅ metrics: {metrics_path}")
    print("[train] ✅ done")


if __name__ == "__main__":
    main()