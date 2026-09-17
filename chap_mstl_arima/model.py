"""MSTL + AutoARIMA wrapper that produces sample-based forecasts for chap.

Design notes
------------
- Fit happens at predict time on ``historic_data``: chap's backtester
  retrains per split, so persisting state buys nothing and StatsForecast
  models are not always trivially pickled across versions.
- We fit on ``log1p(y)`` by default. Probabilistic samples are drawn
  from a Normal at the per-step (μ, σ) returned by the model, with σ
  reconstructed from the 68 % predictive interval. Samples are then
  mapped back via ``expm1`` and clipped at zero.
- No future covariates are required. Seasonality is captured by the
  STL component inside MSTL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, MSTL

from chap_mstl_arima.config import ModelConfig
from chap_mstl_arima.io_utils import (
    detect_frequency,
    period_to_timestamp,
    to_long_panel,
)

# 68.27 % interval ≈ ±1σ under Gaussian; use this to back out σ.
SIGMA_LEVEL = 68


@dataclass
class _FitInputs:
    panel: pd.DataFrame
    freq: str
    season_length: int


def _season_length(cfg: ModelConfig, freq: str) -> int:
    return cfg.season_length_weekly if freq.startswith("W") else cfg.season_length_monthly


def _build_model(cfg: ModelConfig, season_length: int) -> MSTL:
    return MSTL(
        season_length=season_length,
        trend_forecaster=AutoARIMA(
            approximation=cfg.arima_approximation,
            stepwise=cfg.arima_stepwise,
        ),
    )


def _prepare(historic_df: pd.DataFrame, cfg: ModelConfig) -> _FitInputs:
    freq = detect_frequency(historic_df)
    panel = to_long_panel(historic_df, freq)
    if cfg.treat_missing_as_zero:
        # DHIS2 does not store zero values, so missing weeks are usually zero cases.
        panel["y"] = panel["y"].fillna(0.0)
    panel = panel.dropna(subset=["y"])
    if cfg.log_transform:
        panel["y"] = np.log1p(panel["y"].clip(lower=0))
    return _FitInputs(panel=panel, freq=freq, season_length=_season_length(cfg, freq))


class MSTLArimaModel:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def predict(
        self,
        historic_df: pd.DataFrame,
        future_df: pd.DataFrame,
    ) -> pd.DataFrame:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.random_seed)

        inputs = _prepare(historic_df, cfg)
        model = _build_model(cfg, inputs.season_length)

        horizons = future_df.groupby("location").size()
        horizon = int(horizons.max())

        model_col = type(model).__name__  # "MSTL"
        lo_col = f"{model_col}-lo-{SIGMA_LEVEL}"
        hi_col = f"{model_col}-hi-{SIGMA_LEVEL}"

        # MSTL extrapolates the last seasonal cycle and crashes on series
        # shorter than the horizon; such locations use the unseen-location
        # fallback below instead.
        n_obs = inputs.panel.groupby("unique_id")["y"].transform("size")
        panel = inputs.panel[n_obs >= horizon]
        if panel.empty:
            fcst = pd.DataFrame(columns=["unique_id", "ds", model_col])
        else:
            sf = StatsForecast(models=[model], freq=inputs.freq, n_jobs=1)
            fcst = sf.forecast(df=panel, h=horizon, level=[SIGMA_LEVEL])

        fcst_idx = fcst.copy()
        fcst_idx["unique_id"] = fcst_idx["unique_id"].astype(str)
        fcst_idx = fcst_idx.set_index(["unique_id", "ds"])

        future = future_df.copy()
        future["unique_id"] = future["location"].astype(str)
        future["ds"] = future["time_period"].apply(
            lambda p: period_to_timestamp(p, inputs.freq)
        )

        z = norm.ppf(0.5 + SIGMA_LEVEL / 200.0)
        n_samples = cfg.n_samples
        out_rows: list[dict] = []
        for _, row in future.iterrows():
            key = (row["unique_id"], row["ds"])
            if key in fcst_idx.index:
                fr = fcst_idx.loc[key]
                mu = float(fr[model_col])
                if lo_col in fcst_idx.columns:
                    sd = float((fr[hi_col] - fr[lo_col]) / (2.0 * z))
                else:
                    sd = 1e-3
                if not np.isfinite(sd) or sd <= 0:
                    sd = 1e-3
            else:
                # Location unseen in training: fall back to the historic
                # mean across locations.
                mu_raw = float(
                    pd.to_numeric(historic_df["disease_cases"], errors="coerce").mean()
                )
                mu = float(np.log1p(max(0.0, mu_raw))) if cfg.log_transform else mu_raw
                sd = 1.0

            draws = rng.normal(loc=mu, scale=sd, size=n_samples)
            if cfg.log_transform:
                draws = np.expm1(draws)
            draws = np.clip(draws, a_min=0.0, a_max=None)

            entry = {"time_period": row["time_period"], "location": row["location"]}
            for i, v in enumerate(draws):
                entry[f"sample_{i}"] = float(v)
            out_rows.append(entry)

        return pd.DataFrame(out_rows)
