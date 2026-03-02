# src/autoscaling_analysis/config.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Mapping, Tuple

import pandas as pd
import yaml


# -----------------------------
# Defaults (repo-friendly)
# -----------------------------
DEFAULT_CFG: Dict[str, Any] = {
    "project_root": ".",  # repo root
    "tags": ["1m", "5m", "15m"],
    "targets": ["hits", "bytes_sum"],
    "paths": {
        "raw_train": "data/raw/train.txt",
        "raw_test": "data/raw/test.txt",
        "data_interim": "data/interim",
        "data_processed": "data/processed",
        "artifacts_dir": "artifacts",
        "reports_dir": "reports",
        "reports_eda": "reports/eda",
        "reports_figures": "reports/figures",
    },
    "ingest": {
        "chunk_size_lines": 300000,
        "dt_format": "%d/%b/%Y:%H:%M:%S %z",
        "log_regex": r'^(?P<host>\S+)\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+"(?P<request>[^"]*)"\s+(?P<status>\d{3})\s+(?P<bytes>\S+)\s*$',
        "req_regex": r'^(?P<method>[A-Z]+)\s+(?P<url>\S+)\s+(?P<version>HTTP/\d\.\d)$',
    },
    "gaps": {
        "storm_start": "1995-08-01 14:52:01-0400",
        "storm_end": "1995-08-03 04:36:13-0400",
        "unknown_gap_min_hours": 12,
    },
    "features": {
        "time_col": "bucket_start",
        "gap_col": "is_gap",
        "segment_col": "segment_id",
        "require_cols": ["bucket_start", "hits", "bytes_sum", "is_gap"],
        "lag_days": [1, 2, 3, 4, 5, 6, 7],
        "roll_windows": ["1h", "6h", "1d"],
        "roll_use_std": True,
        "use_cyclic": True,
        "horizon_steps": 1,
        "keep_raw_extra": [
            "unique_hosts",
            "err_4xx",
            "err_5xx",
            "error_rate",
            "is_missing_bucket",
            "is_gap_storm",
            "is_gap_unknown",
        ],
    },
    "modeling": {
        "cv": {"splits": 5, "test_days": 2, "gap_steps": 1},
        "xgb": {
            "booster": "gbtree",
            "n_estimators": 5000,
            "early_stopping_rounds": 50,
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "random_state": 42,
        },
    },
    "scaling": {
        "min_instances": 2,
        "max_instances": 50,
        "cost_per_instance_per_hour": 0.05,
        "window_minutes": {"1m": 1, "5m": 5, "15m": 15},
        "safety_buffer_by_metric": {"hits": 0.3, "bytes_sum": 0.3},
        "capacity_per_instance": {
            "hits__1m": 20,
            "hits__5m": 100,
            "hits__15m": 350,
            "bytes_sum__1m": 350_000,
            "bytes_sum__5m": 1_200_000,
            "bytes_sum__15m": 3_500_000,
        },
        "max_step_change_by_window": {"1m": 6, "5m": 10, "15m": 15},
        "hysteresis_by_window": {
            "1m": {"high": 2, "low": 6, "in_margin": 0.18},
            "5m": {"high": 2, "low": 4, "in_margin": 0.18},
            "15m": {"high": 1, "low": 2, "in_margin": 0.12},
        },
        "cooldown_minutes": {"base": 15, "spike": 15},
        "provisioning_by_window": {
            "1m": {"warmup_windows": 1, "min_uptime_windows": 6},
            "5m": {"warmup_windows": 1, "min_uptime_windows": 4},
            "15m": {"warmup_windows": 0, "min_uptime_windows": 2},
        },
        "slo": {
            "base_latency_ms": 80.0,
            "alpha_latency_per_unit_queue": 0.15,
            "p95_latency_target_ms": 300.0,
        },
        "latency": {"queue_decay": 0.02},
        "anomaly": {
            "enabled": True,
            "method": "mad",
            "lookback_hours": 2,
            "mad_k": 6.0,
            "min_points": 10,
            "max_flag_rate": 0.30,
        },
        "ddos_mode": {
            "enabled": True,
            "consecutive_windows": 3,
            "force_scale_out_step_by_window": {"1m": 6, "5m": 10, "15m": 12},
            "max_instances_during_ddos": 50,
        },
    },
}


# -----------------------------
# Utils
# -----------------------------
def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _abs_path(project_root: str, p: str) -> str:
    if p is None:
        return p
    p = str(p)
    if os.path.isabs(p):
        return p
    return str(Path(project_root) / p)


def normalize_capacity_keys(cap_dict: Mapping[Any, Any]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    if not isinstance(cap_dict, Mapping):
        return out

    for k, v in cap_dict.items():
        if isinstance(k, tuple) and len(k) == 2:
            out[(str(k[0]), str(k[1]))] = float(v)
            continue

        if isinstance(k, str):
            ks = k.strip()

            if "__" in ks:
                a, b = ks.split("__", 1)
                out[(a.strip(), b.strip())] = float(v)
                continue

            if "|" in ks:
                a, b = ks.split("|", 1)
                out[(a.strip(), b.strip())] = float(v)
                continue

            if "," in ks and not ks.startswith("("):
                a, b = ks.split(",", 1)
                out[(a.strip(), b.strip())] = float(v)
                continue

            if ks.startswith("(") and ks.endswith(")") and "," in ks:
                ks2 = ks.strip("()")
                a, b = ks2.split(",", 1)
                a = a.strip().strip("'").strip('"')
                b = b.strip().strip("'").strip('"')
                out[(a, b)] = float(v)
                continue

    return out


def _parse_ts(x: Any) -> pd.Timestamp:
    try:
        return pd.Timestamp(x)
    except Exception as e:
        raise ValueError(f"Invalid timestamp in config: {x!r}") from e


def _find_repo_root(cfg_path: Path) -> Path:
    """
    Find repo root by walking upward from config location.
    Repo root is assumed to contain both 'src' and 'configs' directories.
    """
    start = cfg_path.parent.resolve()
    for candidate in [start] + list(start.parents):
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate.resolve()
    # fallback: configs/ is usually directly under repo root
    return cfg_path.parent.parent.resolve()


def _validate_cfg(cfg: Dict[str, Any]) -> None:
    tags = cfg.get("tags", [])
    for t in tags:
        if t not in ("1m", "5m", "15m"):
            raise ValueError(f"Unsupported tag {t!r}. Allowed: 1m,5m,15m")

    _ = _parse_ts(cfg["gaps"]["storm_start"])
    _ = _parse_ts(cfg["gaps"]["storm_end"])

    cap = cfg["scaling"].get("capacity_per_instance", {})
    norm = normalize_capacity_keys(cap)
    for metric in cfg.get("targets", ["hits", "bytes_sum"]):
        for w in tags:
            if (metric, w) not in norm:
                raise ValueError(
                    f"scaling.capacity_per_instance missing key for ({metric},{w}). "
                    f"Provide e.g. '{metric}__{w}: <value>' in config.yaml"
                )

    mi = int(cfg["scaling"]["min_instances"])
    ma = int(cfg["scaling"]["max_instances"])
    if mi <= 0 or ma <= 0 or mi > ma:
        raise ValueError(f"Invalid scaling min/max: min={mi} max={ma}")


# -----------------------------
# Public API
# -----------------------------
def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load YAML config, merge with defaults, and normalize paths based on REPO ROOT.

    Key fix:
    - project_root="." now means repo root (folder containing /src and /configs),
      NOT the configs/ folder.
    """
    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = _deep_merge(DEFAULT_CFG, user_cfg)

    # Determine repo root first (anchor for relative paths)
    repo_root = _find_repo_root(cfg_path)

    # Resolve project_root:
    # - if absolute, keep
    # - if relative, resolve relative to repo_root (NOT configs/)
    project_root = str(cfg.get("project_root", "."))
    if os.path.isabs(project_root):
        project_root_abs = Path(project_root).resolve()
    else:
        project_root_abs = (repo_root / project_root).resolve()

    cfg["project_root"] = str(project_root_abs)

    # Normalize paths to absolute using project_root
    for k, p in list(cfg.get("paths", {}).items()):
        cfg["paths"][k] = _abs_path(cfg["project_root"], p)

    # Ensure some dir paths exist as strings
    for dk in ["data_interim", "data_processed", "artifacts_dir", "reports_dir", "reports_eda", "reports_figures"]:
        cfg["paths"][dk] = _abs_path(cfg["project_root"], cfg["paths"][dk])

    # Normalize raw paths too
    cfg["paths"]["raw_train"] = _abs_path(cfg["project_root"], cfg["paths"]["raw_train"])
    cfg["paths"]["raw_test"] = _abs_path(cfg["project_root"], cfg["paths"]["raw_test"])

    # parse gaps timestamps
    cfg["gaps"]["storm_start"] = str(_parse_ts(cfg["gaps"]["storm_start"]))
    cfg["gaps"]["storm_end"] = str(_parse_ts(cfg["gaps"]["storm_end"]))

    # normalize scaling capacity map for runtime: store BOTH forms
    cap_raw = cfg["scaling"].get("capacity_per_instance", {})
    cfg["scaling"]["capacity_per_instance_raw"] = dict(cap_raw)
    cfg["scaling"]["capacity_per_instance"] = normalize_capacity_keys(cap_raw)

    _validate_cfg(cfg)
    return cfg