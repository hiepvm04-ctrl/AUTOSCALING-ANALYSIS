# src/autoscaling_analysis/features/__init__.py

"""
Feature engineering layer (segment-safe, no leakage):
- segment id building based on gaps
- time/cyclic features
- lags (daily)
- rolling stats (shifted to avoid leakage)
- labels (next-step)
- tz-safe join back to test slice
"""

from .segments import build_segment_id
from .transforms import (
    assert_required_cols,
    add_ratio_features,
    create_time_features,
    add_lags,
    add_rolling,
    add_labels,
    to_tz_naive,
    build_time_key,
)
from .make_features import (
    build_feature_frames_for_tag,
    build_features_all_tags,
)

__all__ = [
    "build_segment_id",
    "assert_required_cols",
    "add_ratio_features",
    "create_time_features",
    "add_lags",
    "add_rolling",
    "add_labels",
    "to_tz_naive",
    "build_time_key",
    "build_feature_frames_for_tag",
    "build_features_all_tags",
]