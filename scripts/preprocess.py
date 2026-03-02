#!/usr/bin/env python3
# scripts/preprocess.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from autoscaling_analysis.config import load_config
from autoscaling_analysis.ingest.parse_logs import parse_file_streaming
from autoscaling_analysis.timeseries.build_ts3 import build_ts3_for_split


def _ensure_dirs(cfg: Dict[str, Any]) -> None:
    Path(cfg["paths"]["data_interim"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["data_processed"]).mkdir(parents=True, exist_ok=True)


def _resolve_path(project_root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (project_root / pp)


def _find_repo_root(cfg_path: Path) -> Path:
    """
    Find repo root by walking upward from the config file location.
    Repo root is assumed to contain both 'src' and 'configs' directories.
    """
    start = cfg_path.parent
    for candidate in [start] + list(start.parents):
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate.resolve()
    # fallback: parent of configs/ if structure is standard
    return cfg_path.parent.parent.resolve()


def main():
    ap = argparse.ArgumentParser(description="Preprocess raw NASA logs -> TS3 parquet (train/test x 1m/5m/15m)")
    ap.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    ap.add_argument("--chunk-lines", type=int, default=None, help="Override streaming chunk size (lines)")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))

    project_root = _find_repo_root(cfg_path)

    # Resolve paths relative to repo root (NOT CWD, NOT configs/)
    raw_train = _resolve_path(project_root, cfg["paths"]["raw_train"])
    raw_test = _resolve_path(project_root, cfg["paths"]["raw_test"])

    cfg["paths"]["data_interim"] = str(_resolve_path(project_root, cfg["paths"]["data_interim"]))
    cfg["paths"]["data_processed"] = str(_resolve_path(project_root, cfg["paths"]["data_processed"]))

    _ensure_dirs(cfg)

    if not raw_train.exists():
        raise FileNotFoundError(f"Missing raw train file: {raw_train}")
    if not raw_test.exists():
        raise FileNotFoundError(f"Missing raw test file: {raw_test}")

    chunk_lines = int(args.chunk_lines or cfg["env"].get("chunk_size_lines", 300000))

    print(f"[preprocess] repo_root = {project_root}")
    print(f"[preprocess] parsing train: {raw_train} (chunk={chunk_lines})")
    df_train = parse_file_streaming(str(raw_train), chunk_lines=chunk_lines)
    print(f"[preprocess] train parsed: {df_train.shape}")

    print(f"[preprocess] parsing test : {raw_test} (chunk={chunk_lines})")
    df_test = parse_file_streaming(str(raw_test), chunk_lines=chunk_lines)
    print(f"[preprocess] test parsed : {df_test.shape}")

    interim_dir = Path(cfg["paths"]["data_interim"])
    train_parsed_pq = interim_dir / "raw_parsed_train.parquet"
    test_parsed_pq = interim_dir / "raw_parsed_test.parquet"
    df_train.to_parquet(train_parsed_pq, index=False)
    df_test.to_parquet(test_parsed_pq, index=False)
    print(f"[preprocess] saved interim: {train_parsed_pq}")
    print(f"[preprocess] saved interim: {test_parsed_pq}")

    processed_dir = Path(cfg["paths"]["data_processed"])
    out_root = processed_dir / "ts3"
    out_root.mkdir(parents=True, exist_ok=True)

    tags = cfg.get("tags", ["1m", "5m", "15m"])

    gap_cfg = {
        "storm_start": cfg["storm"]["start"],
        "storm_end": cfg["storm"]["end"],
        "unknown_gap_min_hours": int(cfg.get("env", {}).get("unknown_gap_min_hours", 12)),
    }

    build_ts3_for_split(df_raw=df_train, split="train", tags=tags, out_dir=str(out_root), gap_cfg=gap_cfg)
    build_ts3_for_split(df_raw=df_test, split="test", tags=tags, out_dir=str(out_root), gap_cfg=gap_cfg)

    print("[preprocess] ✅ done")


if __name__ == "__main__":
    main()