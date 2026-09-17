# chap_mstl_arima

The **MSTL + AutoARIMA** baseline for [CHAP](https://github.com/dhis2-chap), packaged as a
[chapkit](https://github.com/dhis2-chap/chapkit) ML service: a FastAPI application with
`POST /api/v1/ml/$train` and `$predict`, an artifact store, and self-registration with
chap-core.

A two-step pipeline that does well on small, noisy, highly-seasonal disease surveillance
data (LAO admin1, VNM admin1, Rwanda level-5 sectors):

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

Per-location fit on `log1p(disease_cases)`, samples drawn from a Normal at the per-step
(μ, σ) where σ is reconstructed from a 68 % predictive interval; samples are `expm1`'d and
clipped at zero. The model uses **no covariates** and self-forecasts seasonality, so it
needs nothing but `disease_cases`, `location` and `time_period`. Both monthly (`YYYY-MM`)
and weekly (`YYYY-Wnn` or `YYYY-MM-DD/YYYY-MM-DD`) period formats are detected
automatically.

See [`docs/mstl_arima.md`](docs/mstl_arima.md) for the long-form theory and
[`docs/migration-to-chapkit.md`](docs/migration-to-chapkit.md) for how this repo was
converted from an `MLproject` to a chapkit service (including the numeric parity
procedure).

## Quickstart

```bash
uv sync
uv run python main.py     # serves on http://localhost:9090
```

Then, in another shell:

```bash
curl -s http://localhost:9090/health
curl -s http://localhost:9090/api/v1/info
```

The interactive API docs are at `http://localhost:9090/docs`, and chapkit's web console at
`http://localhost:9090/`.

## Train and predict over HTTP

The full cycle with `example_data/monthly/`. This is the chap-core-shaped request body:
the tunables go under `user_option_values`, which the service hoists onto its config
fields.

Dataframes travel as `{"columns": [...], "data": [[...], ...]}` with `null` for missing
values, so first turn the CSVs into request bodies:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
import pandas as pd

def payload(name):
    df = pd.read_csv(f"example_data/monthly/{name}_data.csv")
    return {"columns": df.columns.tolist(), "data": df.where(df.notna(), None).values.tolist()}

Path("/tmp/train.json").write_text(json.dumps({"config_id": "REPLACE_ME", "data": payload("training")}))
Path("/tmp/predict.json").write_text(json.dumps({
    "artifact_id": "REPLACE_ME", "historic": payload("historic"), "future": payload("future"),
}))
PY
```

```bash
BASE=http://localhost:9090

# 1. Create a config. This is the chap-core shape; flat fields work too.
CONFIG_ID=$(curl -s -X POST $BASE/api/v1/configs \
  -H 'content-type: application/json' \
  -d '{"name": "demo", "data": {"user_option_values": {"n_samples": 50, "random_seed": 42}}}' \
  | jq -r .id)

# 2. Submit a training job -> {"job_id": ..., "artifact_id": ...}, HTTP 202
TRAIN=$(jq --arg c "$CONFIG_ID" '.config_id = $c' /tmp/train.json | \
  curl -s -X POST "$BASE/api/v1/ml/\$train" -H 'content-type: application/json' -d @-)

# 3. Poll until the job leaves "running"
curl -s $BASE/api/v1/jobs/$(echo "$TRAIN" | jq -r .job_id) | jq '{status, error}'

# 4. Submit a prediction job against the training artifact
PREDICT=$(jq --arg a "$(echo "$TRAIN" | jq -r .artifact_id)" '.artifact_id = $a' /tmp/predict.json | \
  curl -s -X POST "$BASE/api/v1/ml/\$predict" -H 'content-type: application/json' -d @-)
curl -s $BASE/api/v1/jobs/$(echo "$PREDICT" | jq -r .job_id) | jq '{status, error}'

# 5. Download the predictions (time_period, location, sample_0 .. sample_N)
curl -s "$BASE/api/v1/artifacts/$(echo "$PREDICT" | jq -r .artifact_id)/\$download" | jq '.[0]'
```

For a readable, runnable version of exactly this sequence see
[`scripts/parity.py`](scripts/parity.py), which does the same thing in Python and then
diffs the result against the legacy predictions.

## Smoke tests

chapkit ships a synthetic end-to-end tester:

```bash
uv run chapkit test --url http://localhost:9090 --timeout 300
uv run chapkit test --url http://localhost:9090 --period-type weekly --rows 520 --predict-rows 300 --timeout 300
```

The weekly invocation needs the larger row counts: with the default 250 training rows a
weekly panel has under a year of history, which is less than the 52-period season length.

The repo's own tests drive the app in-process and include the golden parity checks:

```bash
make test        # uv run pytest -v
```

## Evaluation with chap-core

```bash
chap eval \
  --model-name http://localhost:9090 \
  --dataset-csv example_data/monthly/training_data.csv \
  --output-file eval.nc \
  --run-config.is-chapkit-model \
  --backtest-params.n-splits 2 \
  --backtest-params.n-periods 3
```

> **Note.** chap-core 2.3.0 cannot read a service that advertises
> `period_type: "any"` - its REST client enum only accepts `weekly` and `monthly`, and
> `chap eval` fails while validating `/api/v1/info`. This service advertises `any` because
> it genuinely handles both, matching the old `MLproject`'s `supported_period_type: any`
> and chapkit's own `PeriodType.any`. Until chap-core's client catches up, evaluate against
> a build with `period_type=PeriodType.monthly` in `main.py`, or use a newer chap-core.
> Everything else in the chap-core path (config POST, `$train`, job polling, `$predict`,
> artifact download, `.nc` export) is verified working.

## Docker

```bash
make build         # docker build
make test-docker   # build, start on :9000, run chapkit test (monthly + weekly), stop
docker compose up --build              # :9090 -> :8000, hardened, tmpfs /tmp
docker compose -f compose.ghcr.yml up  # prebuilt image from GHCR
```

The container runs as the unprivileged `chapkit` user with a read-only root filesystem.
Only `/work/data` (SQLite) and `/tmp` (ML workspaces, plus any JIT or library cache) are
writable.

## Configuration

`POST /api/v1/configs` accepts these fields, flat or nested under `user_option_values`.
The full JSON schema is at `GET /api/v1/configs/$schema`.

| Field | Default | Meaning |
|---|---|---|
| `prediction_periods` | 3 | Number of periods to predict into the future |
| `n_samples` | 100 | Number of forecast paths per (location, time_period) |
| `log_transform` | true | Fit on `log1p(disease_cases)` |
| `random_seed` | 42 | Sample reproducibility |
| `arima_approximation` | false | Faster AutoARIMA fitting, drops the MA term - less calibrated |
| `arima_stepwise` | true | Hyndman-Khandakar stepwise order selection |
| `treat_missing_as_zero` | false | Treat NaN targets as 0 reported cases instead of dropping them |

`treat_missing_as_zero` is worth knowing about: DHIS2 does not store zeros, so a week with
no reported cases often arrives as a gap rather than a `0`.

The STL seasonal period is not configurable: it is 52 for weekly data and 12 for monthly,
chosen from the detected `time_period` format. It used to be exposed as
`season_length_weekly` / `season_length_monthly`; those were debug knobs and were removed.
A config that still sets them is accepted and ignored.

## Numeric parity with the pre-chapkit model

The service produces **the same numbers** as the `MLproject` CLI it replaced - not
statistically similar, identical. `tests/golden/` holds predictions captured from the
legacy code at commit `20ef2f6`, and the test suite reproduces them:

| Dataset | Rows | Sample columns | Cells exactly equal | Max relative difference |
|---|---|---|---|---|
| monthly (LAO admin1) | 216 | 25 | 5400 / 5400 | 0 |
| weekly (Nicaragua) | 36 | 25 | 900 / 900 | 0 |

This holds because `ShellModelRunner` invokes the original CLI verbatim and
`chap_mstl_arima/{model,io_utils,config}.py` are byte-identical to their pre-conversion
state.

- `tests/golden/VERSIONS.md` - how the fixtures were produced and with which library versions.
- `scripts/parity.py` - re-run the comparison against a live service:

```bash
uv run python scripts/parity.py --url http://localhost:9090 --kind monthly
uv run python scripts/parity.py --url http://localhost:9090 --kind weekly
make parity   # both
```

Comparison is exact on shape, column list and `(time_period, location)` row order;
`sample_*` values are compared with `|new - old| <= PARITY_ATOL + PARITY_RTOL * |old|`
(both default `1e-6`) so a CI runner with a different BLAS or libm cannot fail on a
last-ulp difference. The absolute part covers samples of a few thousandths of a case,
where a 5e-8 platform difference is a 1e-5 relative one. On the machine the goldens were
produced on every cell is exactly equal.

## Legacy CLI

The original file-in / file-out entry points are still available - the service shells out
to them:

```bash
uv run python -m chap_mstl_arima train   <training.csv> <model.json> [config.yml]
uv run python -m chap_mstl_arima predict <model.json> <historic.csv> <future.csv> <out.csv> [config.yml]
```

`train` only writes a config marker; the model is fit at predict time against
`historic.csv`, because CHAP's backtester retrains per split.

## Layout

```
main.py                   # the chapkit service (config class, runner, service info)
chap_mstl_arima/
├── __main__.py           # python -m chap_mstl_arima
├── cli.py                # the file-in / file-out CLI the shell runner invokes
├── config.py             # ModelConfig dataclass
├── io_utils.py           # frequency detection + period parsing
└── model.py              # MSTL + AutoARIMA wrapper + sample generation
example_data/{monthly,weekly}/   # CHAP-shaped training / historic / future CSVs
tests/golden/             # legacy predictions + the procedure that produced them
scripts/parity.py         # legacy vs service numeric comparison
```

## Example data

| File | Use as | Shape |
|---|---|---|
| `example_data/monthly/training_data.csv` | `$train` body | 18 Lao provinces x 152 months |
| `example_data/monthly/historic_data.csv` | `$predict` historic | first 140 months |
| `example_data/monthly/future_data.csv` | `$predict` future | last 12 months, target dropped |
| `example_data/weekly/training_data.csv` | `$train` body | 3 Nicaraguan departments x 148 weeks |
| `example_data/weekly/historic_data.csv` | `$predict` historic | same 148 weeks |
| `example_data/weekly/future_data.csv` | `$predict` future | next 12 weeks, target dropped |
