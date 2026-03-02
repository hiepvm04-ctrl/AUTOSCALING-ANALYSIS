# src/autoscaling_analysis/scaling/__init__.py

"""
Autoscaling simulation layer:
- policy primitives (required_instances, step limits, hysteresis)
- anomaly detection (MAD) + ddos flag
- queue/latency model
- full simulation runner (static vs predictive)
"""

from .policy import (
    normalize_capacity_keys,
    clamp_instances,
    cap,
    buffer,
    win_minutes,
    win_hours,
    step_limit,
    required_instances,
    apply_step_towards,
)
from .anomaly import (
    mad_anomaly_flags,
    ddos_flag,
)
from .latency import (
    simulate_queue_latency,
)
from .simulate import (
    simulate_static,
    simulate_predictive,
    summarize_simulation,
    daily_event_counts,
    instance_distribution,
)

__all__ = [
    "normalize_capacity_keys",
    "clamp_instances",
    "cap",
    "buffer",
    "win_minutes",
    "win_hours",
    "step_limit",
    "required_instances",
    "apply_step_towards",
    "mad_anomaly_flags",
    "ddos_flag",
    "simulate_queue_latency",
    "simulate_static",
    "simulate_predictive",
    "summarize_simulation",
    "daily_event_counts",
    "instance_distribution",
]