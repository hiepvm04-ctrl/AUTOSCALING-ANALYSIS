from autoscaling_analysis.config import load_config
from autoscaling_analysis.scaling.policy import required_instances

def test_required_instances_nonnegative():
    cfg = load_config("configs/config.yaml")
    x = required_instances(cfg, demand=0, metric="hits", window="5m")
    assert x >= cfg["scaling"]["min_instances"]

def test_required_instances_increases():
    cfg = load_config("configs/config.yaml")
    a = required_instances(cfg, demand=100, metric="hits", window="5m")
    b = required_instances(cfg, demand=100000, metric="hits", window="5m")
    assert b >= a