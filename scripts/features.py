#!/usr/bin/env python3
# scripts/features.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from autoscaling_analysis.config import load_config
from autoscaling_analysis.features.make_features import build_features_all_tags


def _ensure_dirs(cfg: Dict[str, Any]) -> None:
    Path(cfg["paths"]["data_processed"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["artifacts_dir"]).mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Build model features (segment-safe) from TS3")
    ap.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    _ensure_dirs(cfg)

    ts3_root = Path(cfg["paths"]["data_processed"]) / "ts3"
    feat_out = Path(cfg["paths"]["data_processed"]) / "features"
    feat_out.mkdir(parents=True, exist_ok=True)

    tags = cfg.get("tags", ["1m", "5m", "15m"])

    # input paths
    train_paths = {t: ts3_root / "train" / f"ts3_{t}.parquet" for t in tags}
    test_paths = {t: ts3_root / "test" / f"ts3_{t}.parquet" for t in tags}

    for t, p in train_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing TS3 train for {t}: {p}")
    for t, p in test_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing TS3 test for {t}: {p}")

    report_df = build_features_all_tags(
        train_paths_by_tag={k: str(v) for k, v in train_paths.items()},
        test_paths_by_tag={k: str(v) for k, v in test_paths.items()},
        out_dir=str(feat_out),
        feature_cfg=cfg["features"],
        tags=tags,
    )

    rep_path = feat_out / "feature_build_report.csv"
    report_df.to_csv(rep_path, index=False)
    print(report_df.to_string(index=False))
    print(f"[features] ✅ saved: {rep_path}")


if __name__ == "__main__":
    main()