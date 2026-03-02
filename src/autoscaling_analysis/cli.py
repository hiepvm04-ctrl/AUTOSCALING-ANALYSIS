# src/autoscaling_analysis/cli.py

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _run(cmd: List[str], *, env: Optional[dict] = None) -> None:
    """
    Run a subprocess command, fail-fast with clear output.
    """
    print(f"[cli] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def _default_config_path() -> str:
    # Allow override via env for UI / CI convenience
    return os.environ.get("AUTOSCALING_CONFIG", "configs/config.yaml")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autoscaling-analysis",
        description="Forecast + autoscaling analysis pipeline (raw logs -> features -> models -> scaling sim -> UI)",
    )

    p.add_argument(
        "--config",
        default=_default_config_path(),
        help="Path to configs/config.yaml (default: env AUTOSCALING_CONFIG or configs/config.yaml)",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # preprocess
    sp = sub.add_parser("preprocess", help="Parse raw logs and build TS3 parquet (train/test x tags)")
    sp.add_argument("--chunk-lines", type=int, default=None, help="Override streaming chunk size (lines)")

    # features
    sub.add_parser("features", help="Build segment-safe ML features from TS3")

    # train
    sub.add_parser("train", help="Train models (xgb + seasonal_naive), export preds + metrics")

    # benchmark
    sb = sub.add_parser("benchmark", help="Create benchmark table from metrics_forecast.csv (long -> wide)")
    sb.add_argument("--split", default="test", help="Which split to benchmark (default=test)")

    # simulate scaling
    ss = sub.add_parser("simulate", help="Run autoscaling simulation from prediction artifacts")
    ss.add_argument("--metric", default="hits", choices=["hits", "bytes_sum"])
    ss.add_argument("--window", default="5m", choices=["1m", "5m", "15m"])
    ss.add_argument("--model", default="xgb", choices=["xgb", "seasonal_naive"])
    ss.add_argument("--test-start", default="1995-08-23 00:00:00")
    ss.add_argument("--test-end", default="1995-09-01 00:00:00")

    # ui
    su = sub.add_parser("ui", help="Run Streamlit UI")
    su.add_argument("--host", default=None, help="Streamlit server address (optional)")
    su.add_argument("--port", type=int, default=None, help="Streamlit server port (optional)")

    # all
    sa = sub.add_parser("all", help="Run full pipeline: preprocess -> features -> train -> benchmark -> simulate")
    sa.add_argument("--chunk-lines", type=int, default=None, help="Override chunk size for preprocess")
    sa.add_argument("--metric", default="hits", choices=["hits", "bytes_sum"])
    sa.add_argument("--window", default="5m", choices=["1m", "5m", "15m"])
    sa.add_argument("--model", default="xgb", choices=["xgb", "seasonal_naive"])
    sa.add_argument("--test-start", default="1995-08-23 00:00:00")
    sa.add_argument("--test-end", default="1995-09-01 00:00:00")
    sa.add_argument("--bench-split", default="test")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Ensure config path exists early (nicer error than downstream)
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[cli] ❌ Config not found: {cfg_path}", file=sys.stderr)
        return 2

    # Base env: propagate config path for UI, and for any code that chooses to read env
    env = dict(os.environ)
    env["AUTOSCALING_CONFIG"] = str(cfg_path)

    try:
        if args.command == "preprocess":
            cmd = ["python", "scripts/preprocess.py", "--config", str(cfg_path)]
            if args.chunk_lines is not None:
                cmd += ["--chunk-lines", str(int(args.chunk_lines))]
            _run(cmd, env=env)

        elif args.command == "features":
            _run(["python", "scripts/features.py", "--config", str(cfg_path)], env=env)

        elif args.command == "train":
            _run(["python", "scripts/train.py", "--config", str(cfg_path)], env=env)

        elif args.command == "benchmark":
            _run(
                ["python", "scripts/benchmark.py", "--config", str(cfg_path), "--split", str(args.split)],
                env=env,
            )

        elif args.command == "simulate":
            _run(
                [
                    "python",
                    "scripts/simulate_scaling.py",
                    "--config",
                    str(cfg_path),
                    "--metric",
                    str(args.metric),
                    "--window",
                    str(args.window),
                    "--model",
                    str(args.model),
                    "--test-start",
                    str(args.test_start),
                    "--test-end",
                    str(args.test_end),
                ],
                env=env,
            )

        elif args.command == "ui":
            # Use scripts/run_ui.sh to keep single source of truth
            cmd = ["bash", "scripts/run_ui.sh", str(cfg_path)]
            # Optional streamlit flags: pass via STREAMLIT_SERVER_* env for compatibility
            if args.host:
                env["STREAMLIT_SERVER_ADDRESS"] = str(args.host)
            if args.port:
                env["STREAMLIT_SERVER_PORT"] = str(int(args.port))
            _run(cmd, env=env)

        elif args.command == "all":
            # 1) preprocess
            cmd = ["python", "scripts/preprocess.py", "--config", str(cfg_path)]
            if args.chunk_lines is not None:
                cmd += ["--chunk-lines", str(int(args.chunk_lines))]
            _run(cmd, env=env)

            # 2) features
            _run(["python", "scripts/features.py", "--config", str(cfg_path)], env=env)

            # 3) train
            _run(["python", "scripts/train.py", "--config", str(cfg_path)], env=env)

            # 4) benchmark
            _run(
                ["python", "scripts/benchmark.py", "--config", str(cfg_path), "--split", str(args.bench_split)],
                env=env,
            )

            # 5) simulate (default hits/5m/xgb)
            _run(
                [
                    "python",
                    "scripts/simulate_scaling.py",
                    "--config",
                    str(cfg_path),
                    "--metric",
                    str(args.metric),
                    "--window",
                    str(args.window),
                    "--model",
                    str(args.model),
                    "--test-start",
                    str(args.test_start),
                    "--test-end",
                    str(args.test_end),
                ],
                env=env,
            )

        else:
            parser.print_help()
            return 2

        print("[cli] ✅ done")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"[cli] ❌ command failed with exit code {e.returncode}", file=sys.stderr)
        return int(e.returncode)
    except Exception as e:
        print(f"[cli] ❌ error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())