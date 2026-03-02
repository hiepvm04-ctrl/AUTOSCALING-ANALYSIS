import pandas as pd
import numpy as np
from autoscaling_analysis.features.transforms import add_rolling

def test_rolling_uses_shift_prevent_leakage():
    df = pd.DataFrame({
        "bucket_start": pd.date_range("2020-01-01", periods=20, freq="5min"),
        "segment_id": [1]*20,
        "hits": np.arange(20).astype(float),
    })
    out, cols = add_rolling(df, "bucket_start", "segment_id", "5m", "hits", ["1h"], roll_use_std=False)
    # For 5m, 1h window = 12 steps; first valid should be at index 12, computed on shifted history (ends at t-1)
    col = cols[0]
    # when it first becomes non-nan, it should NOT include current value at that index
    idx = out[col].first_valid_index()
    assert idx is not None
    # the rolling mean at idx should equal mean of hits[0:12] not hits[1:13] depending on shift
    expected = df["hits"].shift(1).rolling(12, min_periods=12).mean().iloc[idx]
    assert float(out[col].iloc[idx]) == float(expected)