# src/autoscaling_analysis/ui/__init__.py

"""
UI layer (Streamlit):
- Load forecast artifacts
- Run autoscaling simulation
- Visualize KPIs, cost, reliability
"""

from .streamlit_app import main

__all__ = ["main"]