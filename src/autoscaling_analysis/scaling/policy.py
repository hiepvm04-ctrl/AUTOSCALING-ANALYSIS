# src/autoscaling_analysis/scaling/policy.py

from __future__ import annotations

import math
from typing import Any, Dict, Tuple


def normalize_capacity_keys(cap_dict: Any) -> Dict[Tuple[str, str], float]:
    """
    Ensure a dict supports (metric, window) lookup.

    Accepts any of these shapes:
      A) {("hits","5m"): 100, ...}
      B) {"('hits','5m')": 100, ...}
      C) {"hits,5m": 100, ...}
      D) {"hits|5m": 100, ...}
      E) {"hits__5m": 100, ...}
      F) {"hits": {"5m": 100}}   (yaml-friendly)

    Returns:
      {("hits","5m"): 100.0, ...}
    """
    out: Dict[Tuple[str, str], float] = {}
    if not isinstance(cap_dict, dict):
        return out

    # shape F: metric -> window -> value
    is_nested = any(isinstance(v, dict) for v in cap_dict.values())
    if is_nested:
        for metric, wmap in cap_dict.items():
            if not isinstance(wmap, dict):
                continue
            for window, v in wmap.items():
                out[(str(metric), str(window))] = float(v)
        return out

    # shapes A/B/C/D/E
    for k, v in cap_dict.items():
        if isinstance(k, tuple) and len(k) == 2:
            out[(str(k[0]), str(k[1]))] = float(v)
            continue

        if isinstance(k, str):
            ks = k.strip()

            # "hits__5m"
            if "__" in ks:
                a, b = ks.split("__", 1)
                out[(a.strip(), b.strip())] = float(v)
                continue

            # "hits|5m"
            if "|" in ks:
                a, b = ks.split("|", 1)
                out[(a.strip(), b.strip())] = float(v)
                continue

            # "hits,5m"
            if "," in ks and not ks.startswith("("):
                a, b = ks.split(",", 1)
                out[(a.strip(), b.strip())] = float(v)
                continue

            # "('hits','5m')" (no eval)
            if ks.startswith("(") and ks.endswith(")") and "," in ks:
                try:
                    ks2 = ks.strip("()")
                    a, b = ks2.split(",", 1)
                    a = a.strip().strip("'").strip('"')
                    b = b.strip().strip("'").strip('"')
                    out[(a, b)] = float(v)
                    continue
                except Exception:
                    pass

    return out


def win_minutes(sc: Dict[str, Any], window: str) -> int:
    return int(sc["window_minutes"][window])


def win_hours(sc: Dict[str, Any], window: str) -> float:
    return float(win_minutes(sc, window) / 60.0)


def clamp_instances(sc: Dict[str, Any], x: int) -> int:
    return max(int(sc["min_instances"]), min(int(sc["max_instances"]), int(x)))


def cap(sc: Dict[str, Any], metric: str, window: str) -> float:
    cap_map = normalize_capacity_keys(sc.get("capacity_per_instance", {}))
    key = (str(metric), str(window))
    if key not in cap_map:
        raise KeyError(f"capacity_per_instance missing key={key}")
    return float(cap_map[key])


def buffer(sc: Dict[str, Any], metric: str) -> float:
    return float(sc.get("safety_buffer_by_metric", {}).get(metric, 0.2))


def step_limit(sc: Dict[str, Any], window: str) -> int:
    return int(sc.get("max_step_change_by_window", {}).get(window, 10))


def required_instances(sc: Dict[str, Any], demand: float, metric: str, window: str) -> int:
    """
    required = ceil((demand/capacity) * (1+buffer))
    then clamp to [min_instances, max_instances]
    """
    d = max(0.0, float(demand))
    c = max(cap(sc, metric, window), 1e-9)
    need = (d / c) * (1.0 + buffer(sc, metric))
    return clamp_instances(sc, int(math.ceil(need)))


def apply_step_towards(sc: Dict[str, Any], inst: int, target: int, max_step: int) -> int:
    """
    Move inst toward target by at most max_step, then clamp.
    """
    inst = int(inst)
    target = int(target)
    if target == inst:
        return clamp_instances(sc, inst)
    delta = target - inst
    step = int(math.copysign(1, delta)) * min(abs(delta), int(max_step))
    return clamp_instances(sc, inst + step)