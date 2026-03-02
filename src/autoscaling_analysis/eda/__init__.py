# src/autoscaling_analysis/eda/__init__.py

"""
EDA layer:
- Reusable plots
- EDA report generator (compact, report-ordered)
"""

from .plots import (
    plot_timeseries,
    plot_hist,
    plot_heatmap_dow_hour,
    plot_mean_by_hour,
    plot_mean_by_dow,
    plot_corr_lower_triangle,
    plot_acf_simple,
    plot_overlay_anomalies,
)
from .eda_report import (
    run_eda_compact,
)

__all__ = [
    "plot_timeseries",
    "plot_hist",
    "plot_heatmap_dow_hour",
    "plot_mean_by_hour",
    "plot_mean_by_dow",
    "plot_corr_lower_triangle",
    "plot_acf_simple",
    "plot_overlay_anomalies",
    "run_eda_compact",
]