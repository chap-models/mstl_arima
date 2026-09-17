# MSTL + AutoARIMA in `chap_mstl_arima`

This is the model class that consistently wins on the datasets we've evaluated
(LAO admin1, VNM admin1, level-5 sectors with spray). It's pulled together
from two off-the-shelf StatsForecast components but the composition is worth
understanding in detail because it explains both the wins and the
limitations.

```
y(t)  ──►  STL decomposition  ──►  seasonal_12(t) + trend(t) + remainder(t)
                                   │              │
                                   │              ▼
                                   │     trend_forecaster  (AutoARIMA)
                                   │              │
                                   │              ▼
                                   │     trend_forecast(t+h)
                                   │              │
                                   ▼              ▼
                         seasonal_12(t+h)  ─►  ŷ(t+h)
```

The forecast is *not* a single statistical model. It's a pipeline: a
deterministic seasonal extractor in front of a stochastic trend model, with
seasonality re-attached after the fact.

## What our wrapper does

Source: `chap_mstl_arima/model.py` (`_build_model`).

```python
return MSTL(
    season_length=season_length,         # 12 for monthly, 52 for weekly
    trend_forecaster=AutoARIMA(
        approximation=cfg.arima_approximation,   # default: False
        stepwise=cfg.arima_stepwise,             # default: True
    ),
)
```

We fit per-location on `log1p(disease_cases)`, request `level=[68]` from
`forecast()` to get a 68 % predictive interval, recover σ as
`(hi − lo) / (2·Φ⁻¹(0.84))`, sample `n_samples` draws (default 100) from
`Normal(μ, σ)` per step, map back with `expm1`, clip at zero. (See
`chap_mstl_arima/model.py`.)

The seasonal period `p` is fixed by the detected frequency: 52 for weekly data,
12 for monthly (`_season_length` in `chap_mstl_arima/model.py`). It is deliberately not
configurable.

The knobs named in this document (`arima_stepwise`, `arima_approximation`, `n_samples`,
`log_transform`, `random_seed`, `treat_missing_as_zero`) are the fields of the service
config:
`POST /api/v1/configs`, schema at `GET /api/v1/configs/$schema`. They were the
`user_options` block of the repository's old `MLproject` file; see
[`migration-to-chapkit.md`](migration-to-chapkit.md).

The rest of this document explains *what* MSTL and AutoARIMA actually do.

---

## Step 1 — STL: separating seasonal from non-seasonal

STL stands for **Seasonal-Trend decomposition using LOESS** (Cleveland et al.,
1990). Given a series `y(t)` and a stated seasonal period `p` (12 for
monthly), STL produces three components such that

```
y(t) = seasonal(t) + trend(t) + remainder(t)
```

This is **purely a smoothing procedure** — no statistical model, no
likelihood, no parameters to learn beyond a few smoother bandwidths.

**Algorithm sketch:**

1. Start with seasonal and trend components initialised to zero.
2. Repeat:
   a. **Detrend**: subtract current trend from `y`.
   b. **Cycle-subseries smoothing**: collect every `p`-th observation into
      `p` subseries (e.g. all Januaries form one subseries, all Februaries
      another …) and apply a LOESS smoother independently to each. This
      gives a slowly-varying seasonal pattern that may evolve year to year.
   c. **Low-pass filtering**: apply a moving average then a LOESS pass over
      the smoothed seasonal to extract its trend-component, then subtract
      that from the smoothed seasonal so it has zero mean over each cycle.
   d. **Re-extract trend**: subtract the new seasonal from `y` and LOESS-smooth
      the result to get an updated trend.
3. Compute `remainder(t) = y(t) − seasonal(t) − trend(t)`.

The key properties that make this useful for disease surveillance:

- **Non-parametric.** No distributional assumption on `y`, no fitted
  coefficients to overfit, no MLE convergence to fail.
- **Robust outlier handling.** LOESS gives weighted local regressions where
  outliers can be down-weighted (depending on the robustness option).
- **Allows the seasonal pattern to drift.** Unlike Fourier dummies, STL's
  seasonal component can slowly change shape over the years.

The deseasonalised series fed to the trend forecaster is
`x_sa(t) = trend(t) + remainder(t)`, i.e. `y` minus the extracted seasonal.

## Step 2 — AutoARIMA on the deseasonalised series

The deseasonalised series goes into a `trend_forecaster`. In our config
that's **AutoARIMA**, an automated Box-Jenkins selector based on the
Hyndman & Khandakar algorithm (the `auto.arima()` from R's `forecast`).

### ARIMA in 60 seconds

An **ARIMA(p, d, q)** model writes the differenced series as a linear
combination of its own lags and past noise terms:

```
(1 - φ₁B - φ₂B² - … - φ_p Bᵖ) (1-B)^d x(t)  =  (1 + θ₁B + … + θ_q B^q) ε(t)
```

where `B` is the backshift operator (`B x(t) = x(t-1)`), `d` is the order of
non-seasonal differencing needed to make the series stationary, `p` is the
AR (autoregressive) order, `q` is the MA (moving-average) order, and `ε(t)`
are i.i.d. Gaussian innovations with variance `σ²`.

AutoARIMA can also include a **seasonal** component `(P, D, Q)_m`, but in
MSTL we deliberately suppress that: the seasonal piece has already been
removed by STL. MSTL's constructor enforces this by either:

- Refusing to accept a trend forecaster that has its own seasonality
  (e.g. AutoETS with seasonal != 'N'), or
- Requiring `season_length=1` on the trend forecaster.

So in our pipeline the "AutoARIMA" inside MSTL is really searching over
non-seasonal `(p, d, q)` triples on a series whose seasonality is already
gone.

### How AutoARIMA picks `(p, d, q)`

1. **Determine `d`** with a KPSS-style stationarity test. Difference the
   series until the test says it's stationary.
2. **Initial grid**. Start from a small set of candidate `(p, q)` pairs
   (e.g. `(2, 2)`, `(0, 0)`, `(1, 0)`, `(0, 1)`).
3. **Stepwise search** (`stepwise=True`, our default). For each model fit,
   compute AICc (small-sample-corrected AIC). Then for each candidate vary
   `p` or `q` by ±1 and refit; keep the best AICc; repeat until no
   neighbouring step improves. This is fast (O(10-30) ARIMA fits) but can
   get stuck in local minima.
4. **Full search** (`stepwise=False`, i.e. `arima_stepwise: false` in the service config).
   Try every `(p, q)` up to `max_p, max_q` (default 5 each). Slow but
   exhaustive — our HPO sweep showed this gave roughly +0.03 log-CRPS on
   the small `laos_subset.csv` but was 7× slower on full LAO and 30× slower
   on VNM, with marginal accuracy gain.
5. **Parameter estimation** for each candidate is by conditional or exact
   maximum likelihood (depending on series length and order).
6. **Return** the model with the lowest AICc.

`approximation=True` switches the fitting from exact MLE to a faster CSS
(conditional sum of squares) approximation. We default to `False`.

### How AutoARIMA forecasts and produces uncertainty

Once fit, ARIMA gives a **closed-form Gaussian predictive distribution**
at each horizon `h`:

```
x̂(t+h) ~ Normal(μ_h, σ_h²)
```

where `μ_h` is the recursion of the AR equation under E[ε]=0, and `σ_h²`
is the variance of accumulated innovations — it grows monotonically with
`h` because ARIMA propagates the innovation noise through the AR
recursion. This is the standard textbook ARIMA prediction interval.

In our wrapper we request `level=[68]`, get back the 16 % and 84 %
quantiles of that Gaussian, and back out σ from `(hi − lo) / (2·Φ⁻¹(0.84))`.

## Step 3 — Reattaching the seasonal

MSTL's predict step is just:

```
ŷ(t+h)  =  trend_forecast(t+h)  +  seasonal_extrapolation(t+h)
```

`seasonal_extrapolation(t+h)` is **deterministic**: it takes the last
fitted seasonal component and continues it periodically. Concretely
StatsForecast wraps `_predict_mstl_seas`, which holds the last cycle of
the smoothed seasonal and repeats it across the horizon.

The predictive variance is **only** the trend forecaster's variance. The
seasonal piece contributes zero forecast uncertainty.

### What this means in practice

- **Pro.** The forecast variance is dramatically lower than seasonal-ARIMA
  would produce on the same series, because seasonal ARIMA's variance
  includes the seasonal innovation. When seasonality is genuinely
  deterministic (or near-deterministic, as in our climate-driven disease
  series), this is *correct* — we should not be uncertain about something
  we know with high confidence.
- **Con — known coverage issue.** When the seasonal amplitude changes
  year-to-year, MSTL's intervals can be **under-covered** during high-
  amplitude seasons. The trend forecaster only sees the deseasonalised
  remainder, so it has no way to express "next April's peak might be
  unusually high or unusually low".
- **Implication for log-CRPS.** Our log-CRPS on LAO/VNM looks favourable
  partly because intervals are narrow *and* well-calibrated on average.
  If we cared specifically about the highest-amplitude seasons, MSTL would
  systematically under-warn.

## Why this combination works so well here

The data have three properties that are perfectly aligned with MSTL's
inductive bias:

1. **Strong, stable, low-dimensional seasonality.** Monthly malaria-style
   counts in tropical climates have a dominant annual cycle. STL extracts
   it once and gets it right; we don't burn model capacity re-learning it
   per series.
2. **Short series with many locations.** Per-series we have ~150 monthly
   observations. Seasonal ARIMA's seasonal differencing eats 12 observations
   immediately, leaving very little to estimate the seasonal terms. STL
   uses the whole series for seasonal extraction (and is non-parametric
   anyway), so we don't lose any data.
3. **Noisy non-seasonal residual.** After STL removes seasonality, what's
   left is a fairly low-frequency trend with noise. AutoARIMA on such a
   series usually picks small orders (ARIMA(0,1,1), ARIMA(1,1,0)), which
   are exactly the well-behaved models that perform well on small samples.

The flip side is that MSTL has **no native way to use covariates** — see
the note below.

## Limitations and natural next steps

- **No exogenous variables out of the box.** MSTL passes `X` through to the
  trend forecaster via its `fit(y, X)` / `forecast(y, h, X, X_future)`
  signatures, so an AutoARIMA trend can in principle become ARIMAX. Useful
  only for covariates with non-seasonal information (anomalies, intervention
  dates), since the seasonal component is already removed.
- **Deterministic seasonal extrapolation.** No uncertainty about future
  seasonal shape.
- **Per-series fits, no pooling.** Each location is fit independently. On
  the spray dataset (406 sectors) that's 406 STL+ARIMA fits per backtest
  split — still fast, but it means a sector with little data cannot
  borrow strength from its neighbours. A hierarchical reconciliation step
  could help here.
- **Gaussian PI on log-counts.** Our 68 %-interval-to-σ trick is exact only
  when the underlying predictive distribution is Gaussian. It is, on log
  scale, for AutoARIMA; the `expm1` mapping produces a log-normal predictive
  distribution on counts, which can over-estimate the upper tail. For raw
  counts a Negative-Binomial reconstruction would be more faithful.

## References

- Cleveland, R. B., Cleveland, W. S., McRae, J. E. & Terpenning, I. (1990).
  *STL: A seasonal-trend decomposition procedure based on LOESS*. Journal
  of Official Statistics, 6(1), 3-73.
- Hyndman, R. J. & Khandakar, Y. (2008). *Automatic time series forecasting:
  the `forecast` package for R*. Journal of Statistical Software, 27(3).
- Bandara, K., Hyndman, R. J. & Bergmeir, C. (2021). *MSTL: A seasonal-
  trend decomposition algorithm for time series with multiple seasonal
  patterns*. International Journal of Operational Research.
- Nixtla StatsForecast: `MSTL` at
  https://nixtlaverse.nixtla.io/statsforecast/docs/models/mstl.html
