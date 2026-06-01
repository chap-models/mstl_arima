# chap_mstl_arima

Chap-compatible, uv-managed MLproject providing the **MSTL + AutoARIMA**
baseline model — a two-step pipeline that consistently wins on small,
noisy, highly-seasonal disease surveillance data (LAO admin1, VNM
admin1, Rwanda level-5 sectors).

This repo is the extraction of the `mstl_arima` model class from
[`chap_nixtla`](https://github.com/knutdrand/chap_nixtla) into a
focused, minimal-dependency baseline.

## What it does

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

Per-location fit on `log1p(disease_cases)`, samples drawn from a Normal
at the per-step (μ, σ) where σ is reconstructed from a 68 % predictive
interval. Samples are `expm1`'d and clipped at zero. See
[`docs/mstl_arima.md`](docs/mstl_arima.md) for the long-form theory.

## Layout

```
chap_mstl_arima/
├── MLproject            # train / predict entry points + user_options schema
├── pyproject.toml       # uv environment
├── main.py              # CLI: `train`, `predict`
├── chap_mstl_arima/
│   ├── config.py        # ModelConfig dataclass
│   ├── io_utils.py      # frequency detection + period parsing
│   └── model.py         # MSTL + AutoARIMA wrapper + sample generation
├── configurations/
│   └── mstl_arima.yaml  # default config (n_samples=100, log_transform=true)
└── docs/
    └── mstl_arima.md    # theory + practice walkthrough
```

## Setup

```bash
uv venv --python 3.11
uv sync
```

## Running

```bash
# Sanity check (uses the model's own uv environment)
chap sanity-check-model . --dataset-path PATH/TO/dataset.csv

# Evaluation with MLflow tracking
chap eval . PATH/TO/dataset.csv eval.nc \
    --model-configuration-yaml configurations/mstl_arima.yaml \
    --run-config.track \
    --backtest-params.n-splits 12 \
    --backtest-params.n-periods 3
```

For monthly chap data, prefer `--backtest-params.n-splits 12` (one year
of held-out windows). Keep `--backtest-params.n-periods 3` and stride 1
unless you have a reason not to.

## Configuration knobs

All exposed via `MLproject` `user_options` (passable from
`--model-configuration-yaml` or as `--user-option-values` overrides):

| Knob | Default | Notes |
|---|---|---|
| `n_samples` | 100 | Number of forecast paths per (location, time_period) |
| `log_transform` | true | Fit on `log1p(y)` |
| `season_length_monthly` | 12 | STL seasonal period for monthly data |
| `season_length_weekly` | 52 | STL seasonal period for weekly data |
| `random_seed` | 42 | Sample reproducibility |
| `arima_approximation` | false | Faster but drops the MA term — less calibrated |
| `arima_stepwise` | true | Hyndman-Khandakar stepwise order selection |

## Why this baseline

Naive ARIMA on chap data over-parameterises the seasonal block relative
to the few seasonal cycles in the training window. STL strips the
seasonal pattern using LOESS (no fitted parameters), then ARIMA models
only the trend + remainder, which is well within its statistical
budget. The seasonal is extrapolated forward by re-attaching the last
observed cycle.

The pipeline produces calibrated 80 / 50 % predictive intervals on the
test datasets it was tuned for. It does *not* use exogenous
covariates — those are deliberately routed through different models in
the wider `chap_nixtla` family.
