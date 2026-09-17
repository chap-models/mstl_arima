"""MSTL + AutoARIMA baseline for chap.

A two-step pipeline: STL decomposition strips seasonality, AutoARIMA
forecasts the trend+remainder, and the last observed seasonal cycle is
re-attached. Calibrated to small, noisy, highly-seasonal disease
surveillance data.
"""

from chap_mstl_arima.config import ModelConfig
from chap_mstl_arima.model import MSTLArimaModel

__all__ = ["ModelConfig", "MSTLArimaModel"]
