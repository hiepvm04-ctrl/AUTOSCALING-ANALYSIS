# src/autoscaling_analysis/scaling/simulate.py

from __future__ import annotations

import math
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from .policy import (
    win_minutes,
    win_hours,
    clamp_instances,
    cap,
    required_instances,
    step_limit,
    apply_step_towards,
)
from .anomaly import mad_anomaly_flags, ddos_flag
from .latency import simulate_queue_latency


def daily_event_counts(ev_df: pd.DataFrame) -> pd.DataFrame:
    if ev_df is None or ev_df.empty:
        return pd.DataFrame(columns=["date", "scale_out", "scale_in", "total"])

    tmp = ev_df.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
    tmp["date"] = tmp["timestamp"].dt.date.astype(str)
    tmp["is_out"] = tmp["action"].astype(str).str.contains("scale_out", case=False, na=False).astype(int)
    tmp["is_in"] = tmp["action"].astype(str).str.contains("scale_in", case=False, na=False).astype(int)

    g = tmp.groupby("date")[["is_out", "is_in"]].sum().reset_index()
    g = g.rename(columns={"is_out": "scale_out", "is_in": "scale_in"})
    g["total"] = g["scale_out"] + g["scale_in"]
    return g.sort_values("date").reset_index(drop=True)


def instance_distribution(sim_df: pd.DataFrame) -> pd.DataFrame:
    g = sim_df["instances"].astype(int).value_counts().sort_index()
    out = pd.DataFrame({"instances": g.index, "count": g.values})
    out["pct_time"] = out["count"] / out["count"].sum()
    return out


def simulate_static(
    df_case: pd.DataFrame,
    *,
    sc: Dict[str, Any],
    metric: str,
    window: str,
    static_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Static policy: fixed instances = static_n for all steps.
    df_case must have: timestamp, y_true, y_pred (y_pred optional, used for required_instances column)
    """
    df = df_case.sort_values("timestamp").reset_index(drop=True).copy()

    wh = win_hours(sc, window)
    unit_cost = float(sc["cost_per_instance_per_hour"])
    inst = clamp_instances(sc, static_n)

    rows = []
    for _, r in df.iterrows():
        ts = r["timestamp"]
        y_true = float(r["y_true"])
        y_pred = float(r.get("y_pred", y_true))

        capacity_total = inst * cap(sc, metric, window)
        headroom = capacity_total - y_true
        under = max(0.0, -headroom)
        over = max(0.0, headroom)
        cost_step = inst * wh * unit_cost

        rows.append(
            {
                "timestamp": ts,
                "metric": metric,
                "window": window,
                "policy_mode": "static",
                "y_true": y_true,
                "y_pred": y_pred,
                "required_instances": int(required_instances(sc, y_pred, metric, window)),
                "instances": int(inst),
                "effective_instances": int(inst),
                "capacity_total": float(capacity_total),
                "headroom": float(headroom),
                "under_provision": float(under),
                "over_provision": float(over),
                "sla_violation": bool(under > 0.0),
                "cost_step": float(cost_step),
                "server_hours_step": float(inst * wh),
                "cost_rate_per_hour": float(inst * unit_cost),
            }
        )

    sim_df = pd.DataFrame(rows)
    ev_df = pd.DataFrame(
        columns=["timestamp", "metric", "window", "policy_mode", "action", "from_instances", "to_instances", "delta", "reason"]
    )
    return sim_df, ev_df


def simulate_predictive(
    df_case: pd.DataFrame,
    *,
    sc: Dict[str, Any],
    metric: str,
    window: str,
    latency_cfg: Dict[str, Any] | None = None,
    anomaly_cfg: Dict[str, Any] | None = None,
    ddos_cfg: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predictive policy: uses y_pred to compute required_instances, then applies hysteresis, cooldown, warmup.
    Mirrors notebook CELL 10 behavior.
    """
    df = df_case.sort_values("timestamp").reset_index(drop=True).copy()

    wh = win_hours(sc, window)
    unit_cost = float(sc["cost_per_instance_per_hour"])

    # hysteresis knobs per window
    h = sc["hysteresis_by_window"][window]
    k_high = int(h["high"])
    k_low = int(h["low"])
    in_margin = float(h["in_margin"])

    cooldown_w = int(math.ceil(float(sc["cooldown_minutes"]["base"]) / win_minutes(sc, window)))
    prov = sc["provisioning_by_window"][window]
    warmup_w = int(prov["warmup_windows"])
    min_uptime_w = int(prov["min_uptime_windows"])

    inst = int(sc["min_instances"])
    above_count = 0
    below_count = 0
    cooldown_left = 0
    warmup_left = 0
    uptime_guard = 0

    # anomaly + ddos precompute from y_true (demo)
    anomaly_cfg = anomaly_cfg or sc.get("anomaly", {})
    ddos_cfg = ddos_cfg or sc.get("ddos_mode", {})

    if bool(anomaly_cfg.get("enabled", True)):
        lookback_hours = float(anomaly_cfg.get("lookback_hours", 2))
        mad_k = float(anomaly_cfg.get("mad_k", 6.0))
        min_points = int(anomaly_cfg.get("min_points", 10))
        lookback_pts = max(5, int((lookback_hours * 60) / win_minutes(sc, window)))
        an_score, is_spike = mad_anomaly_flags(df["y_true"], window_pts=lookback_pts, k=mad_k, min_points=min_points)
    else:
        an_score = pd.Series(np.zeros(len(df)))
        is_spike = pd.Series(np.zeros(len(df), dtype=int))

    ddos_enabled = bool(ddos_cfg.get("enabled", True))
    consec = int(ddos_cfg.get("consecutive_windows", 3))
    is_ddos = ddos_flag(is_spike, consec) if ddos_enabled else pd.Series(np.zeros(len(df), dtype=int))

    force_step_by_window = ddos_cfg.get("force_scale_out_step_by_window", {})
    ddos_max = int(ddos_cfg.get("max_instances_during_ddos", sc["max_instances"]))

    rows = []
    events = []

    for i, r in df.iterrows():
        ts = r["timestamp"]
        y_true = float(r["y_true"])
        y_pred = float(r.get("y_pred", y_true))

        req = required_instances(sc, y_pred, metric, window)

        # update counters
        above_count = (above_count + 1) if (req > inst) else 0
        below_count = (below_count + 1) if (inst > req * (1.0 + in_margin)) else 0

        # tick timers
        cooldown_left = max(0, cooldown_left - 1)
        warmup_left = max(0, warmup_left - 1)
        uptime_guard = max(0, uptime_guard - 1)

        action, reason = "hold", "hold"
        new_inst = inst

        ddos_on = bool(ddos_enabled and int(is_ddos.iloc[i]) == 1)

        # DDoS mode: force scale-out
        if ddos_on:
            force_step = int(force_step_by_window.get(window, step_limit(sc, window)))
            new_inst = min(ddos_max, inst + max(1, force_step))
            action, reason = "scale_out", "ddos_mode(force_step)"

        # normal predictive scaling
        elif cooldown_left == 0:
            if above_count >= k_high:
                new_inst = apply_step_towards(sc, inst, req, step_limit(sc, window))
                action, reason = "scale_out", f"req>inst for {k_high} window(s)"
            elif below_count >= k_low and uptime_guard == 0:
                new_inst = apply_step_towards(sc, inst, req, step_limit(sc, window))
                action, reason = "scale_in", f"inst>req*(1+margin) for {k_low} window(s)"

        # apply change
        if new_inst != inst:
            events.append(
                {
                    "timestamp": ts,
                    "metric": metric,
                    "window": window,
                    "policy_mode": "predictive",
                    "action": action,
                    "from_instances": int(inst),
                    "to_instances": int(new_inst),
                    "delta": int(new_inst - inst),
                    "reason": reason,
                }
            )
            inst = int(new_inst)
            cooldown_left = cooldown_w
            warmup_left = max(warmup_left, warmup_w)
            uptime_guard = max(uptime_guard, min_uptime_w)

        effective_inst = max(0, inst - warmup_left)
        capacity_total = effective_inst * cap(sc, metric, window)
        headroom = capacity_total - y_true
        under = max(0.0, -headroom)
        over = max(0.0, headroom)
        cost_step = inst * wh * unit_cost

        rows.append(
            {
                "timestamp": ts,
                "metric": metric,
                "window": window,
                "policy_mode": "predictive",
                "y_true": y_true,
                "y_pred": y_pred,
                "required_instances": int(req),
                "instances": int(inst),
                "effective_instances": int(effective_inst),
                "warmup_left_windows": int(warmup_left),
                "blocked_by_cooldown": bool(cooldown_left > 0),
                "capacity_total": float(capacity_total),
                "headroom": float(headroom),
                "under_provision": float(under),
                "over_provision": float(over),
                "sla_violation": bool(under > 0.0),
                "cost_step": float(cost_step),
                "server_hours_step": float(inst * wh),
                "cost_rate_per_hour": float(inst * unit_cost),
                "anomaly_score": float(an_score.iloc[i]),
                "is_spike": int(is_spike.iloc[i]),
                "is_ddos": int(is_ddos.iloc[i]),
            }
        )

    sim_pred = pd.DataFrame(rows)
    ev_pred = pd.DataFrame(events)

    # latency overlay
    if latency_cfg is not None:
        sim_pred = simulate_queue_latency(sim_pred, latency_cfg)

    return sim_pred, ev_pred


def summarize_simulation(sim_df: pd.DataFrame, ev_df: pd.DataFrame, *, sc: Dict[str, Any]) -> Dict[str, Any]:
    metric = str(sim_df["metric"].iloc[0])
    window = str(sim_df["window"].iloc[0])
    policy = str(sim_df["policy_mode"].iloc[0])

    n = int(len(sim_df))
    total_cost = float(sim_df["cost_step"].sum())
    total_server_hours = float(sim_df["server_hours_step"].sum())
    avg_instances = float(sim_df["instances"].mean())
    peak_instances = int(sim_df["instances"].max())
    sla_violation_rate = float(sim_df["sla_violation"].mean())
    slo_violation_rate = float(sim_df["slo_violation"].mean()) if "slo_violation" in sim_df.columns else float("nan")
    total_under = float(sim_df["under_provision"].sum())
    max_under = float(sim_df["under_provision"].max())
    num_events = int(len(ev_df)) if ev_df is not None else 0

    sim_hours = (n * win_minutes(sc, window)) / 60.0
    events_per_hour = float(num_events / max(sim_hours, 1e-9))

    return {
        "metric": metric,
        "window": window,
        "policy_mode": policy,
        "estimated_total_cost": total_cost,
        "total_server_hours": total_server_hours,
        "avg_instances": avg_instances,
        "peak_instances": peak_instances,
        "sla_violation_rate": sla_violation_rate,
        "slo_violation_rate": float(slo_violation_rate),
        "total_under_provision": total_under,
        "max_under_provision": max_under,
        "num_scale_events": num_events,
        "events_per_hour": events_per_hour,
        "num_points": n,
    }