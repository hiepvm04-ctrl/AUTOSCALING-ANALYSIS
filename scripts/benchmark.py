#!/usr/bin/env python3
# scripts/benchmark.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from autoscaling_analysis.config import load_config
from autoscaling_analysis.benchmark import load_metrics_long, build_benchmark_table


def main():
    ap = argparse.ArgumentParser(description="Build benchmark table from metrics_forecast.csv (long -> wide)")
    ap.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    ap.add_argument("--split", default="test", help="split to benchmark (default=test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    artifacts = Path(cfg["paths"]["artifacts_dir"])
    metrics_path = artifacts / "metrics" / "metrics_forecast.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}. Run scripts/train.py first.")

    mdf = load_metrics_long(str(metrics_path))
    bench = build_benchmark_table(mdf, split=args.split)

    out_path = artifacts / "metrics" / f"benchmark_{args.split}.csv"
    bench.to_csv(out_path, index=False)

    print("[benchmark] ✅ Benchmark table")
    print(bench.to_string(index=False))
    print(f"[benchmark] saved: {out_path}")


if __name__ == "__main__":
    main()