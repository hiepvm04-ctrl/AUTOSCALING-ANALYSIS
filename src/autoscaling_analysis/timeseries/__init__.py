"""
Timeseries layer:
- Gap detection
- Storm masking
- Timeline rebuilding
- TS3 dataset construction
"""

from .gaps import (
    GapConfig,
    label_gaps_ts3,
    mark_storm_gap,
    detect_unknown_gaps,
)

from .build_ts3 import (
    build_ts3_for_split,
)

__all__ = [
    "GapConfig",
    "label_gaps_ts3",
    "mark_storm_gap",
    "detect_unknown_gaps",
    "build_ts3_for_split",
]