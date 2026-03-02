from autoscaling_analysis.config import load_config, get_capacity

def test_capacity_key_exists():
    cfg = load_config("configs/config.yaml")
    assert get_capacity(cfg, "hits", "5m") > 0
    assert get_capacity(cfg, "bytes_sum", "15m") > 0