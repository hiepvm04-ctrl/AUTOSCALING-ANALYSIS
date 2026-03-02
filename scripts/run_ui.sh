#!/usr/bin/env bash
set -euo pipefail

# scripts/run_ui.sh
# Run Streamlit dashboard from the installed package/module

CONFIG_PATH="${1:-configs/config.yaml}"

echo "[run_ui] Using config: ${CONFIG_PATH}"
export AUTOSCALING_CONFIG="${CONFIG_PATH}"

# Option A: run module (recommended if pip install -e .)
streamlit run -m autoscaling_analysis.ui.streamlit_app