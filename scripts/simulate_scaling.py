#!/usr/bin/env python3
# scripts/simulate_scaling.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from autoscaling_analysis.config import load_config
from autoscaling_analysis.scaling import (
    normalize_capacity_keys,
    required_instances,
    simulate_static,
    simulate_predictive,
    summarize_simulation,
)


def _load_pred_case(pred_csv: Path, metric: str) -> pd.DataFrame:
    dfp = pd.read_csv(pred_csv)
    ts = pd.to_datetime(dfp["bucket_start"], utc=True, errors="coerce").dt.tz_convert(None)
    dfp = dfp.assign(timestamp=ts).dropna(subset=["timestamp"]).copy()

    df_case = pd.DataFrame(
        {
            "timestamp": dfp["timestamp"],
            "y_true": pd.to_numeric(dfp[metric], errors="coerce").fillna(0.0),
            "y_pred": pd.to_numeric(dfp["pred"], errors="coerce").fillna(0.0),
        }
    ).sort_values("timestamp").reset_index(drop=True)
    return df_case


def main():
    ap = argparse.ArgumentParser(description="Run autoscaling simulation from prediction artifacts")
    ap.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    ap.add_argument("--metric", default="hits", choices=["hits", "bytes_sum"])
    ap.add_argument("--window", default="5m", choices=["1m", "5m", "15m"])
    ap.add_argument("--model", default="xgb", choices=["xgb", "seasonal_naive"])
    ap.add_argument("--test-start", default="1995-08-23 00:00:00")
    ap.add_argument("--test-end", default="1995-09-01 00:00:00")  # exclusive
    args = ap.parse_args()

    cfg = load_config(args.config)

    artifacts = Path(cfg["paths"]["artifacts_dir"])
    pred_dir = artifacts / "predictions"
    scaling_dir = artifacts / "scaling"
    scaling_dir.mkdir(parents=True, exist_ok=True)

    pred_csv = pred_dir / f"pred_{args.metric}_{args.window}_{args.model}.csv"
    if not pred_csv.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_csv}. Run scripts/train.py first.")

    sc: Dict[str, Any] = cfg["scaling"]
    # normalize capacity key formats (yaml-friendly)
    sc["capacity_per_instance"] = normalize_capacity_keys(sc.get("capacity_per_instance", {}))

    # latency cfg
    slo = sc.get("slo", {})
    latency_cfg = {
        "base_ms": float(slo.get("base_latency_ms", 80.0)),
        "alpha_ms_per_queue_unit": float(slo.get("alpha_latency_per_unit_queue", 0.15)),
        "p95_target_ms": float(slo.get("p95_latency_target_ms", 300.0)),
        "queue_decay": float(sc.get("latency", {}).get("queue_decay", 0.02)),
    }

    anomaly_cfg = sc.get("anomaly", {})
    ddos_cfg = sc.get("ddos_mode", {})

    df_case = _load_pred_case(pred_csv, args.metric)

    TEST_START = pd.Timestamp(args.test_start)
    TEST_END = pd.Timestamp(args.test_end)
    df_case = df_case[(df_case["timestamp"] >= TEST_START) & (df_case["timestamp"] < TEST_END)].copy()
    df_case = df_case.sort_values("timestamp").reset_index(drop=True)
    if df_case.empty:
        raise ValueError("Selected test range is empty after filtering.")

    # static baseline tuned on train-only (no leakage): Jul + 1–22 Aug
    train_end = pd.Timestamp("1995-08-23 00:00:00")
    df_train = df_case[df_case["timestamp"] < train_end].copy()
    if df_train.empty:
        cut = int(len(df_case) * 0.7)
        df_train = df_case.iloc[:cut].copy()

    static_req = df_train["y_true"].apply(lambda x: required_instances(sc, x, args.metric, args.window))
    static_n = int(np.nanpercentile(static_req.values, 95))

    sim_static, ev_static = simulate_static(
        df_case,
        sc=sc,
        metric=args.metric,
        window=args.window,
        static_n=static_n,
    )
    # latency overlay for static
    from autoscaling_analysis.scaling import simulate_queue_latency
    sim_static = simulate_queue_latency(sim_static, latency_cfg)

    sim_pred, ev_pred = simulate_predictive(
        df_case,
        sc=sc,
        metric=args.metric,
        window=args.window,
        latency_cfg=latency_cfg,
        anomaly_cfg=anomaly_cfg,
        ddos_cfg=ddos_cfg,
    )

    sum_static = summarize_simulation(sim_static, ev_static, sc=sc)
    sum_pred = summarize_simulation(sim_pred, ev_pred, sc=sc)
    summary_df = pd.DataFrame([sum_static, sum_pred])

    sim_all = pd.concat([sim_static, sim_pred], ignore_index=True)
    ev_all = pd.concat([ev_static, ev_pred], ignore_index=True)

    sim_path = scaling_dir / "sim_timeseries_all.csv"
    ev_path = scaling_dir / "scaling_events_all.csv"
    sum_path = scaling_dir / "summary_cost_perf.csv"

    sim_all.to_csv(sim_path, index=False)
    ev_all.to_csv(ev_path, index=False)
    summary_df.to_csv(sum_path, index=False)

    saving = float(sum_static["estimated_total_cost"] - sum_pred["estimated_total_cost"])
    saving_pct = saving / max(float(sum_static["estimated_total_cost"]), 1e-9)

    print("=" * 110)
    print(f"[simulate_scaling] window={args.window}, metric={args.metric}, model={args.model}")
    print(f"- static baseline instances (p95 train-style): {static_n}")
    print(f"- saving ≈ {saving:.4f} ({saving_pct*100:.2f}%)")
    print("=" * 110)
    print(summary_df.sort_values("policy_mode").to_string(index=False))
    print("[simulate_scaling] saved:")
    print("-", sim_path)
    print("-", ev_path)
    print("-", sum_path)


if __name__ == "__main__":
    main()