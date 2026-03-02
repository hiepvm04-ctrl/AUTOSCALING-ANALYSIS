# src/autoscaling_analysis/scaling/latency.py

from __future__ import annotations

from typing import Dict, Any

import pandas as pd


def simulate_queue_latency(sim_df: pd.DataFrame, lat_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Simple queue + p95 latency model (same spirit as notebook):
      served = min(load + q, cap_total)
      q <- max(0, load+q-cap_total)
      q <- q * (1 - queue_decay)
      p95 = base_ms + alpha * q
      slo_violation = p95 > target
    """
    df = sim_df.sort_values("timestamp").reset_index(drop=True).copy()

    base_ms = float(lat_cfg.get("base_ms", 80.0))
    alpha = float(lat_cfg.get("alpha_ms_per_queue_unit", 0.15))
    target = float(lat_cfg.get("p95_target_ms", 300.0))
    decay = float(lat_cfg.get("queue_decay", 0.02))

    q = 0.0
    ql, util, p95, slo = [], [], [], []

    for _, r in df.iterrows():
        load = float(r["y_true"])
        cap_total = float(r["capacity_total"])

        _served = min(load + q, cap_total)
        q = max(0.0, (load + q) - cap_total)
        q = max(0.0, q * (1.0 - decay))

        u = 0.0 if cap_total <= 1e-9 else min(2.0, load / cap_total)
        p = base_ms + alpha * q
        v = bool(p > target)

        ql.append(float(q))
        util.append(float(u))
        p95.append(float(p))
        slo.append(v)

    df["queue_len"] = ql
    df["utilization"] = util
    df["p95_latency_ms"] = p95
    df["slo_violation"] = pd.Series(slo).astype(bool)

    return df