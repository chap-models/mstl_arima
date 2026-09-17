# Migrating `mstl_arima` from an MLproject to a chapkit service

This is the running step log of the conversion of `chap-models/mstl_arima` from a
CHAP `MLproject` (cyclopts CLI, `chap eval .` against a local checkout) into a
[chapkit](https://github.com/dhis2-chap/chapkit) 2.x ML service (FastAPI, `POST
/api/v1/ml/$train` and `$predict`, artifact store, self-registration with chap-core).

It is written to be honest rather than tidy: every command that was run, every decision,
every dead end. The primary audience is a future migration of another chap-model repo.

- **Base commit:** `20ef2f6` (`main`, PR #1 merged).
- **Branch:** `feat/chapkit-service`.
- **Target chapkit:** `>=2.0.1,<3`.
- **Runner:** `ShellModelRunner` with `config_format="chap_core"`.
- **Machine:** macOS 27.0 (Darwin), arm64, uv 0.12.0, chap-core 2.3.0.

## Why `ShellModelRunner` and not `FunctionalModelRunner`

The legacy model already has a clean file-in / file-out CLI (`train`, `predict`) that
reads a CHAP-shaped `config.yml`. `ShellModelRunner` invokes exactly that CLI, which
means the numeric core (`chap_mstl_arima/model.py`, `io_utils.py`, `config.py`) is
carried over **byte for byte**. That is the single strongest guarantee of numeric parity:
the new service literally shells out to the same code path the old `MLproject` did.

`FunctionalModelRunner` would have required rewriting the entry points against
`chapkit.data.DataFrame` (a Pydantic schema, not pandas), introducing a conversion layer
between the service and the model, and therefore a place for drift to hide.

Constraint accepted in exchange: a subprocess per job, and the project directory is
copied into a temp workspace for every train and predict call.

## Overview of the conversion

| Legacy | chapkit |
|---|---|
| `MLproject` `entry_points.train.command` | `ShellModelRunner(train_command=...)` |
| `MLproject` `entry_points.predict.command` | `ShellModelRunner(predict_command=...)` |
| `MLproject` `user_options` | `MSTLArimaConfig(BaseConfig)` fields with `Field(description=...)` |
| `MLproject` `meta_data` | `MLServiceInfo(...)` + `ModelMetadata(...)` |
| `MLproject` `supported_period_type: any` | `period_type=PeriodType.any` |
| `MLproject` `required_covariates: []` | `required_covariates=[]` |
| `configurations/mstl_arima.yaml` | `POST /api/v1/configs` body |
| root `main.py` (cyclopts CLI) | `chap_mstl_arima/cli.py` + `chap_mstl_arima/__main__.py` |
| (nothing) | root `main.py` = the chapkit service |

---

## Step 1 - `build: move to uv with chapkit 2 and pin dependencies`

### What was done

- Rewrote `pyproject.toml`:
  - build backend `setuptools` -> `uv_build` (`[tool.uv.build-backend] module-name = "chap_mstl_arima"`, `module-root = ""`, because the package sits at the repo root, not under `src/`).
  - `requires-python` `>=3.11,<3.14` -> `>=3.13` (chapkit 2 requires 3.13, and the `ghcr.io/dhis2-chap/chapkit-py` base image is 3.13).
  - added `chapkit>=2.0.1,<3`; capped `statsforecast>=2.0.0,<3`; raised `pandas>=2.0` -> `>=2.2`.
  - `[project.optional-dependencies] dev` -> PEP 735 `[dependency-groups] dev` (what `uv sync --all-groups` and `uv sync --no-dev` understand), adding `pytest>=8` and `httpx>=0.28` (needed by `fastapi.testclient.TestClient`).
  - added `[tool.pytest.ini_options]` with `pythonpath = ["."]` so `tests/` can `from main import app` (the service module is a root-level file, not a package member).
  - added the ruff config used across the chapkit reference repos (`py313`, line length 120, `select = ["E", "W", "F", "I"]`).
- Added `.python-version` = `3.13`.
- Fixed `.gitignore`. The old file ignored `uv.lock`, `*.csv` and `.python-version`, all
  three of which this conversion must commit (`uv.lock` for reproducible Docker builds,
  `*.csv` for `example_data/` and `tests/golden/`). Added `data/` (the SQLite dir the
  service creates), `*.db*`, `.pytest_cache/` and `.ruff_cache/`.

### Commands

```
uv lock            # 1.5 s, resolved 102 packages
uv sync --all-groups   # 7.3 s
uv run python -c "from statsforecast import StatsForecast"
```

### Resolved versions (macOS arm64, this is the venv every golden file below was produced with)

| Package | Version |
|---|---|
| python | 3.13.14 (CPython, clang 22.1.3) |
| chapkit | 2.0.1 |
| servicekit | 2.0.2 |
| statsforecast | 2.0.1 |
| statsmodels | 0.15.0 |
| utilsforecast | 0.2.15 |
| numba | 0.67.0 |
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| scipy | 1.18.1 |
| cyclopts | 4.25.2 |
| pydantic | 2.13.5 |
| fastapi | 0.141.1 |

### Decisions and notes

- **pandas 3 was left unpinned.** The plan flagged a risk that `statsforecast` might not
  work under a resolved pandas 3.x, with a fallback of pinning `pandas>=2.2,<3` and
  re-locking *before* producing any golden file. The pin was written first, then backed
  out after an explicit smoke test: a `MSTL(season_length=12, trend_forecaster=AutoARIMA())`
  fit plus `forecast(h=3, level=[68])` on a 60-point synthetic monthly series runs clean
  on statsforecast 2.0.1 + pandas 3.0.5 + numpy 2.5.3. The full golden runs in step 2
  confirm it on real data. So the repo ships with `pandas>=2.2` and no upper bound.
- `statsforecast` resolves to **2.0.1**, not 2.1.x. 2.0.1 is what the index offers for
  this requires-python; no pin was needed to get there.
- `cyclopts` resolves to **4.x** even though the floor is `>=2.9`. The legacy CLI only uses
  `cyclopts.App()` and `@app.command()`, both unchanged in 4.x. Verified by running the
  legacy CLI in step 2 with this venv.

---

## Step 2 - `test: add example data and legacy golden predictions`

This step happens **before** any service code exists. That ordering is deliberate: the
baseline has to be captured with the old code, or there is nothing trustworthy to compare
against later.

### Example data

- `example_data/monthly/{training_data,historic_data,future_data}.csv` copied verbatim
  from `chapkit_simple_multistep_model/example_data/` - 18 Lao admin1 locations, monthly
  `YYYY-MM` periods, 2736 / 2520 / 216 rows. `future_data.csv` has no `disease_cases`
  column, which is how chap-core posts the future frame.
- `example_data/weekly/{training_data,historic_data,future_data}.csv` derived from
  `auto_regressive_weekly_v2/input/trainData.csv` (3 Nicaraguan departments, 160 weekly
  periods each, `YYYY-MM-DD/YYYY-MM-DD` period format). The last 12 periods per location
  become `future_data.csv` with `disease_cases` dropped; the remaining 148 periods per
  location are both `training_data.csv` and `historic_data.csv` (444 rows). Original row
  order is preserved on both sides. The weekly data keeps 3 NaN `disease_cases` values,
  which is useful: it exercises the NaN -> None JSON conversion and the model's dropna.

Weekly data is not optional decoration. `supported_period_type: any` means the service
claims to handle both, and the two code paths differ (`detect_frequency` -> `W-MON` vs
`MS`, `season_length_weekly=52` vs `season_length_monthly=12`, and a different
`period_to_timestamp` branch). A monthly-only parity check would leave half the model
unverified.

### Capturing the baseline

The golden predictions must come from the **legacy** code but the **new** virtualenv.
Mixing those up is the classic way to produce a meaningless parity check: run the old code
with old libraries and the new code with new libraries, and any diff is unattributable.

```
git worktree add ../mstl_arima-baseline 20ef2f6       # untouched legacy code
cd ../mstl_arima-baseline
uv run --project ../mstl_arima python -c \
  "import chap_mstl_arima; print(chap_mstl_arima.__file__)"
# /Users/.../mstl_arima-baseline/chap_mstl_arima/__init__.py    <- correct
```

`uv run --project <dir>` resolves the environment from `<dir>` but keeps the current
working directory, and `''` (cwd) is the first entry on `sys.path`, so the worktree's own
`chap_mstl_arima` shadows the copy installed into the venv. This worked first try; the
documented fallbacks (`PYTHONPATH=$WT`, or `uv run --project $REPO --directory $WT`) were
not needed.

Then, from the worktree:

```
uv run --project $REPO python main.py train \
  $REPO/example_data/monthly/training_data.csv $SCRATCH/monthly_model.json \
  $REPO/tests/golden/config.yaml                                   # 1.3 s

uv run --project $REPO python main.py predict \
  $SCRATCH/monthly_model.json \
  $REPO/example_data/monthly/historic_data.csv \
  $REPO/example_data/monthly/future_data.csv \
  $SCRATCH/monthly_a.csv $REPO/tests/golden/config.yaml            # 3.0 s
```

and the same pair against `example_data/weekly/` (1.2 s train, 1.5 s predict).

### Determinism proof

`predict` was run a second time into `monthly_b.csv` from the same marker and config:

```
cmp $SCRATCH/monthly_a.csv $SCRATCH/monthly_b.csv   # byte-identical
```

This matters because it establishes that *exact* equality is the right bar. The model
seeds `np.random.default_rng(cfg.random_seed)` inside `MSTLArimaModel.predict` and draws
`n_samples` values per `future_df` row in iteration order, so the RNG stream is a pure
function of (config, future row order). Any difference the service introduces - a
reordered future frame, a dropped row, a config value that failed to reach the script -
shows up immediately as a numeric diff rather than as a silent statistical wobble.

### Files added

- `tests/golden/config.yaml` - `user_option_values: {n_samples: 25, random_seed: 42}`.
  25 samples instead of 100 keeps the fixtures at 93 KB / 18 KB while still covering 5400
  and 900 float cells respectively.
- `tests/golden/lao_monthly_predictions.csv` - 216 x 27 (`time_period`, `location`,
  `sample_0` .. `sample_24`).
- `tests/golden/nicaragua_weekly_predictions.csv` - 36 x 27.
- `tests/golden/VERSIONS.md` - the reproduction recipe, the resolved versions, and the
  tolerance policy.
- `scripts/parity.py` - the parity harness (below).

### `scripts/parity.py`

Compares a **reference** against a **candidate**:

- reference = the legacy CLI re-run live (`python -m chap_mstl_arima train|predict`, which
  from step 3 onward is the same code the service shells out to), or, with `--golden`, the
  committed fixture produced by the pre-conversion code;
- candidate = a running chapkit service, driven over HTTP exactly the way chap-core drives
  it: `POST /api/v1/configs` with the chap-core-shaped body
  `{"name": ..., "data": {"user_option_values": {...}}}`, `POST /api/v1/ml/$train`, poll
  `GET /api/v1/jobs/{job_id}`, `POST /api/v1/ml/$predict`, poll again, then
  `GET /api/v1/artifacts/{artifact_id}/$download` using the `artifact_id` that came back in
  the 202 response (both `$train` and `$predict` return `{"job_id", "artifact_id"}`).

Comparison rules: column list, row count and `(time_period, location)` order are compared
**exactly** and abort on any difference; the `sample_*` cells are compared numerically and
the script prints a markdown table (kind, rows, sample columns, exactly-equal cells, max
abs diff, max rel diff) and exits 1 when the max relative difference exceeds `--rtol`
(default `PARITY_RTOL`, default `1e-6`).

Two details worth stealing for other migrations:

- `pd.read_csv(..., float_precision="round_trip")` on both sides. Without it pandas' fast
  float parser can differ in the last ulp and manufacture a parity failure out of nothing.
- Relative difference is only defined where the reference is non-zero. Cells where the
  reference is exactly 0 (the model clips samples at zero, so there are some) count as
  `inf` relative difference if the candidate is non-zero, and 0 otherwise.

---

## Step 3 - `feat: expose MSTL + AutoARIMA as a chapkit service`

### File moves

| From | To | Change |
|---|---|---|
| `main.py` (cyclopts CLI) | `chap_mstl_arima/cli.py` | module docstring rewritten; `if __name__ == "__main__":` body extracted into `main()`. Nothing else. |
| - | `chap_mstl_arima/__main__.py` | 3 lines, calls `cli.main()`. |
| - | `main.py` | the chapkit service. |

`chap_mstl_arima/model.py`, `io_utils.py` and `config.py` are **byte-identical** to
`20ef2f6`. That is checkable:

```
git diff 20ef2f6 -- chap_mstl_arima/model.py chap_mstl_arima/io_utils.py chap_mstl_arima/config.py
# (empty)
```

`python -m` rather than a console script: `ShellModelRunner` copies the project into a
scratch workspace and runs the command there with that workspace as `cwd`. `cwd` is on
`sys.path`, so `python -m chap_mstl_arima` resolves the copied package without the
project having to be installed into the workspace. `python` itself resolves to the
virtualenv interpreter both locally (`uv run python main.py` puts `.venv/bin` on `PATH`)
and in the image (`/app/.venv/bin` is on `PATH` in `ghcr.io/dhis2-chap/chapkit-py`).

Verified immediately after the move, before writing any test:

```
uv run python -m chap_mstl_arima train    example_data/monthly/training_data.csv ... 
uv run python -m chap_mstl_arima predict  ...
cmp <out> tests/golden/lao_monthly_predictions.csv   # byte-identical
```

### The runner

```python
runner: ShellModelRunner[MSTLArimaConfig] = ShellModelRunner(
    train_command="python -m chap_mstl_arima train {data_file} model.json config.yml",
    predict_command=(
        "python -m chap_mstl_arima predict model.json {historic_file} {future_file} "
        "{output_file} config.yml"
    ),
    config_format="chap_core",
)
```

`{data_file}` -> `data.csv`, `{historic_file}` -> `historic.csv`, `{future_file}` ->
`future.csv`, `{output_file}` -> `predictions.csv`, all workspace-relative. `model.json`
and `config.yml` are literal filenames: chapkit always writes `config.yml` into the
workspace, and the whole train workspace (including the `model.json` the train command
wrote) is zipped, stored as the `ml_training_workspace` artifact and restored into the
predict workspace before the predict command runs.

`config_format="chap_core"` is the reason `cli.py` needed no edits. It makes chapkit emit

```yaml
prediction_periods: 3
additional_continuous_covariates: []
user_option_values:
  n_samples: 25
  random_seed: 42
  ...
```

and the legacy `_load_config` already does
`raw.get("user_option_values") or raw.get("user_options") or raw`. The default
(`config_format="flat"`) would have written every tunable at the top level, which
`_load_config` would *also* have accepted via its `or raw` fallback - but then
`prediction_periods` and `additional_continuous_covariates` would have been fed to
`ModelConfig.from_user_options` as unknown keys (harmless, they are filtered) while
diverging from the shape every other chap-models script expects. `chap_core` is the right
default for a migration.

### The `user_option_values` hoisting validator - the single most important gotcha

chap-core creates a config with

```json
{"name": "my-run", "user_option_values": {"n_samples": 25}, "additional_continuous_covariates": []}
```

`BaseConfig` has `model_config = {"extra": "allow"}`. Without a hook, `user_option_values`
is therefore accepted as an unknown extra field and stored verbatim; every declared
tunable keeps its default; and `dump_config_yaml(config, "chap_core")` - which nests
*everything except* `prediction_periods` and `additional_continuous_covariates` under
`user_option_values` - emits

```yaml
user_option_values:
  user_option_values: {n_samples: 25}   # <- the request, buried
  n_samples: 100                        # <- the default, which is what the script reads
```

The service answers 200, the job succeeds, the numbers are quietly wrong. Hence:

```python
@model_validator(mode="before")
@classmethod
def _hoist_user_option_values(cls, data: object) -> object:
    if isinstance(data, dict) and isinstance(data.get("user_option_values"), dict):
        hoisted = {k: v for k, v in data.items() if k != "user_option_values"}
        for key, value in data["user_option_values"].items():
            hoisted.setdefault(key, value)
        return hoisted
    return data
```

`setdefault` means flat keys win over nested ones, so `chapkit test` (which posts flat
fields) and chap-core (which posts nested ones) both land on the same object. Verified:

```
MSTLArimaConfig.model_validate({"name": "x", "user_option_values": {"n_samples": 7}}).n_samples  == 7
MSTLArimaConfig.model_validate({"n_samples": 3, "user_option_values": {"n_samples": 7}}).n_samples == 3
dump_config_yaml(cfg, "chap_core")  ->  user_option_values.n_samples: 7, no nesting
```

The second required default is `prediction_periods`. `BaseConfig` declares it with **no
default** (`prediction_periods: int`), and chap-core never sends it - it carries the
horizon in the future frame instead. Without `Field(default=3, ...)` every chap-core
config POST is a 422.

### Service info

Straight translation of the MLproject `meta_data` block, plus the contract fields. The
only judgement calls: `min_prediction_periods=1` (the model cannot forecast zero periods
usefully) and `max_prediction_periods=104` (two years of weekly periods; beyond that the
seasonal extrapolation is just repeating the same cycle). `author_note` from the MLproject
is carried over into `ModelMetadata.author_note` even though the plan did not list it -
it is an MLproject field and dropping it would lose information. The MLproject title for
`arima_approximation` contained a Unicode em dash; it was replaced with an ASCII hyphen.

### Verified at the end of this step (service running on :9090)

| Comparison | Result |
|---|---|
| golden monthly fixture vs service | 5400 / 5400 cells exactly equal, max abs diff 0.000e+00 |
| golden weekly fixture vs service | 900 / 900 cells exactly equal, max abs diff 0.000e+00 |
| live legacy CLI vs service (monthly) | 5400 / 5400 cells exactly equal |

Exact, not approximate. As expected: the service runs the same code, in the same
interpreter, on the same inputs, in the same order.
