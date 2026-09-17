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
  - added `chapkit>=2.0.1,<3`; capped `statsforecast>=2.0.0,<3`; raised `pandas>=2.0` -> `>=2.2`
    (both revised in step 8 to `statsforecast>=2.1.0,<3` and `pandas>=2.2,<3`).
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

> **Superseded in step 8.** This resolution installs fine on macOS but cannot be built in
> the Docker image: statsforecast 2.0.1 publishes no cp313 wheels. See
> [step 8](#step-8---builddeps-pin-pandas-below-3-so-statsforecast-resolves-to-a-wheel-release)
> for the corrected pins (statsforecast 2.1.1, pandas 2.3.3) and for why the decision
> recorded below was wrong.

### Decisions and notes

- **pandas 3 was left unpinned - and this was the one real mistake of the conversion.**
  The plan flagged a risk that `statsforecast` might not work under a resolved pandas 3.x,
  with a fallback of pinning `pandas>=2.2,<3`. The pin was written first, then backed out
  after an explicit smoke test: a `MSTL(season_length=12, trend_forecaster=AutoARIMA())`
  fit plus `forecast(h=3, level=[68])` on a 60-point synthetic monthly series runs clean
  on statsforecast 2.0.1 + pandas 3.0.5 + numpy 2.5.3, and the golden runs in step 2
  confirmed it on real data. The test answered the question that was asked ("does it
  *run*?") and not the question that mattered ("does it *install from a wheel on linux*?").
  Step 8 undoes this.
- `statsforecast` resolved to **2.0.1**, not 2.1.x, and the reason is the pandas pin:
  statsforecast 2.1.x requires `pandas<3`, so allowing pandas 3 silently held statsforecast
  back two minor versions. A transitive constraint quietly downgrading a direct dependency
  is easy to miss when the only thing you look at is whether the import works.
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

---

## Step 4 - `test: in-process service tests with golden parity`

### Layout

```
tests/
├── __init__.py          # makes conftest import exactly once, as tests.conftest
├── conftest.py          # DATABASE_URL + session TestClient + golden_options
├── helpers.py           # df_payload / wait_for_job / create_config / train / predict
├── test_service.py      # 9 tests
└── golden/              # from step 2
```

`tests/conftest.py` sets `DATABASE_URL` to a temp SQLite **file** before
`from main import app`, because `main.py` reads the env var at module import time and
because an in-memory database would not be shared between the request thread and the
background job worker:

```python
_DB_DIR = Path(tempfile.mkdtemp(prefix="chap_mstl_arima_test_"))
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_DB_DIR}/test.db")

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402
```

The `# noqa: E402` comments are load-bearing: ruff's `E402` (module level import not at top
of file) would otherwise fail CI, and moving the imports up would break the test suite.

`tests/__init__.py` exists so pytest imports `conftest.py` once, as `tests.conftest`.
Without it, `from tests.helpers import ...` in a test module creates a *second* copy of the
package's modules under different names. Shared constants live in `helpers.py` rather than
`conftest.py` for the same reason.

The tests drive the ASGI app in-process through `TestClient`, but the jobs still fork real
`python -m chap_mstl_arima` subprocesses into real temp workspaces. Nothing about the
runner path is mocked.

### The 9 tests

| Test | What it pins down |
|---|---|
| `test_health` | service boots, database and registration subsystems healthy |
| `test_info` | `id`, `display_name`, `period_type == "any"`, `required_covariates == []`, `allow_free_additional_continuous_covariates is False`, author metadata |
| `test_config_schema_defaults` | all 9 config fields present in `GET /api/v1/configs/$schema` with the MLproject defaults and a non-empty description; `required` is empty |
| `test_config_hoists_user_option_values` | chap-core-shaped POST -> `n_samples == 7`, `prediction_periods == 3`, no leftover `user_option_values` extra field |
| `test_config_flat_fields_win_over_nested` | unit test on the class: flat only, nested only, and both (flat wins) |
| `test_monthly_reproduces_legacy_golden` | exact column list, exact `(time_period, location)` order, `assert_allclose(rtol=PARITY_RTOL, atol=0)` over 5400 cells |
| `test_weekly_reproduces_legacy_golden` | same over 900 cells |
| `test_unseen_location_fallback` | a future row for a location absent from training still returns finite, non-negative samples (the `historic mean` fallback in `model.py`) |
| `test_future_row_order_preserved` | future frame shuffled with `random_state=1234`; output row order equals input row order |

`test_future_row_order_preserved` is the one that looks like padding and is not. The model
draws `n_samples` values per future row in iteration order from a single seeded generator,
and chap-core matches predictions to requests positionally. A runner (or a future
refactor) that sorted the future frame would produce plausible-looking, wrong numbers
without failing anything else.

`JOB_TIMEOUT_SECONDS = 600`, generous because a single monthly job fits 18 MSTL +
AutoARIMA models in a subprocess after chapkit has copied the project into a fresh
workspace. (Under statsforecast 2.0.1 it also paid numba's JIT compilation; 2.1.1 does not
depend on numba - see step 8.)

### Results (macOS arm64)

```
9 passed in 17.75s

5.21s  test_monthly_reproduces_legacy_golden
3.66s  test_unseen_location_fallback
3.63s  test_future_row_order_preserved
3.61s  test_weekly_reproduces_legacy_golden
1.56s  setup (the shared monthly training artifact)
```

```
monthly: 5400 / 5400 cells exactly equal to the legacy golden output
weekly:   900 /  900 cells exactly equal to the legacy golden output
```

### Gotcha found in this step: `ruff format` rewrote the frozen numeric core

`uv run ruff format .` (ruff 0.16.8) reformatted `chap_mstl_arima/model.py` - reordering
`from statsforecast.models import AutoARIMA, MSTL` to `import MSTL, AutoARIMA` and joining
three wrapped expressions. All cosmetic, all semantically identical, and all fatal to the
one property this conversion sells: `git diff <base> -- chap_mstl_arima/model.py` must be
empty, so that a reviewer can verify by inspection that the numbers cannot have moved.

Same run also rewrote Python code blocks **inside Markdown files** - ruff >= 0.16 formats
fenced Python in `.md`, which silently reflowed documentation snippets that were wrapped
for reading.

Fix, in `pyproject.toml`:

```toml
[tool.ruff]
extend-exclude = [
    "chap_mstl_arima/config.py",
    "chap_mstl_arima/io_utils.py",
    "chap_mstl_arima/model.py",
    "*.md",
]
```

and `git checkout <base> -- chap_mstl_arima/{config,io_utils,model}.py` to undo the damage.
Worth doing **before** the first `ruff format` on any migration that promises an untouched
model core.

`chap_mstl_arima/cli.py` is deliberately *not* excluded, so it did get reformatted: one
`or` chain in `_load_config` was joined onto a single line. Semantically identical, and
the file is glue rather than numerics - but it means "moved verbatim" is now "moved
verbatim, then formatted".

---

## Step 5 - `ci: add docker packaging, makefile and github workflows`

### `Dockerfile`

Taken from the chapkit `shell-py` scaffold template
(`chapkit/src/chapkit/cli/templates/Dockerfile.jinja2`, the `else` branch), with two
changes:

- `COPY main.py ./` and `COPY chap_mstl_arima/ ./chap_mstl_arima/` instead of the
  template's `COPY scripts/ ./scripts/` - this repo's shell entry point is a package, not
  a `scripts/` directory.
- `NUMBA_CACHE_DIR=/tmp/numba_cache` added next to the template's `HOME`, `MPLCONFIGDIR`
  and `XDG_CACHE_HOME`. statsforecast's numba kernels try to write a JIT cache next to the
  installed package; with `read_only: true` in compose that directory is not writable and
  the first prediction job fails. Any statsforecast / numba model needs this line.

`--no-install-project` is correct here: the project is never pip-installed into the image.
`ShellModelRunner` copies `/work` into a scratch workspace and runs
`python -m chap_mstl_arima` with that workspace as `cwd`, and `cwd` is on `sys.path`.

### `.dockerignore`

The scaffold list plus `example_data/`, `tests/`, `docs/` and `scripts/`. These are not
copied by the Dockerfile anyway, so the entries are belt and braces - but they also
document the intent: the smaller the image's `/work`, the smaller every per-job workspace
copy.

### `compose.yml` / `compose.ghcr.yml`

`compose.yml` follows the ewars reference: host port 9090 -> container 8000, `init: true`,
`read_only: true`, `no-new-privileges`, `cap_drop: ALL`, `user: chapkit:chapkit`, a named
volume at `/work/data` for the SQLite database and a 2 GB tmpfs at `/tmp` for ML workspaces
and the numba cache. The chap-core self-registration environment variables are present but
commented out; note the `$$register` double dollar, which compose needs to emit a literal
`$`. `compose.ghcr.yml` points at `ghcr.io/chap-models/mstl_arima:latest`.

### `Makefile`

`run` (local, port 9090), `build`, `run-ghcr`, `test` (pytest), `test-docker` (build,
start on port 9000, wait for `/health`, then `chapkit test` monthly **and** weekly),
`parity` (both kinds against a running service), `lint` / `check`, `clean`.

### Workflows

`ci.yml` has two jobs, both `timeout-minutes: 30`: `lint-and-test` (uv + Python 3.13 +
`make check` + `make test`, which includes golden parity) and `docker-build` (buildx with
GHA cache, start the container, poll `/health`, `chapkit test --timeout 300 --verbose`,
dump container logs on failure). `publish-docker.yml` is the standard chap-models publish
workflow, pushing to `ghcr.io/${{ github.repository }}` with build provenance attestation
and `GIT_REVISION=${{ github.sha }}`.

### `CLAUDE.md`

The three project rules, verbatim from `chapkit_ewars_model/CLAUDE.md`: no emojis, no
tool attribution in commits or PRs, Conventional Commits for messages, branches and PR
titles.

### Not verified at the time

The Docker daemon was not running when this step was written, so none of it was executed
and the image had never been built. That gap is exactly what hid the dependency bug fixed
in step 8: `uv sync --frozen` tried to compile `statsforecast` from source and there is no
compiler in the base image. Once Docker came up, `make build` and `make test-docker` both
passed - see the verification results below. **Build the image before you trust the
lockfile.**

---

## Step 6 - `docs: describe chapkit service usage and migration`

- `README.md` rewritten: what the model is, `uv sync` / `uv run python main.py` quickstart,
  a runnable curl sequence (config -> `$train` -> poll -> `$predict` -> poll -> `$download`),
  `chapkit test` for both period types, `chap eval`, docker, the 9-field config table, the
  parity section, and the legacy CLI. The curl sequence in the README was executed verbatim
  against a running service before being committed; it is not pseudo-code.
- `docs/mstl_arima.md`: the theory document kept its content. Two references were
  repointed - `configurations/auto_arima_best.yaml` (a file that no longer exists) became
  "`arima_stepwise: false` in the service config", and a paragraph was added mapping the
  knobs it discusses to the service config schema.
- This file finished.

---

## Step 7 - `chore: remove MLproject and legacy entry points`

`git rm MLproject configurations/mstl_arima.yaml`. The root `main.py` was already replaced
in step 3. Everything the two files carried now lives in the service:

| MLproject | Now |
|---|---|
| `meta_data.*` | `MLServiceInfo` / `ModelMetadata` in `main.py` |
| `user_options.*` | `MSTLArimaConfig` fields, with the MLproject titles as `Field(description=...)` |
| `supported_period_type` | `period_type=PeriodType.any` |
| `required_covariates`, `allow_free_additional_continuous_covariates` | same names on `MLServiceInfo` |
| `target: disease_cases` | implicit; chapkit has no per-service target field |
| `uv_env: pyproject.toml` | the Dockerfile's `uv sync --frozen` |
| `entry_points.train.command` | `ShellModelRunner(train_command=...)` |
| `entry_points.predict.command` | `ShellModelRunner(predict_command=...)` |
| `configurations/mstl_arima.yaml` | a `POST /api/v1/configs` body (see the README) |

---

## Step 8 - `build(deps): pin pandas below 3 so statsforecast resolves to a wheel release`

Added after the first `docker build` was attempted. **This step exists because step 1 got
the dependency pins wrong in a way that only a Linux container build could reveal.**

### The failure

```
make test-docker
...
Failed to build `statsforecast==2.0.1`
  ...
  No such file or directory: 'g++'
```

### Root cause

`statsforecast` 2.0.1 and 2.0.2 publish **no cp313 wheels for any platform** - 20 wheels
each, none of them `cp313`. So on Python 3.13 pip/uv always compiles the C++ extension
from source. On macOS that silently works, because clang is present. On the
`ghcr.io/dhis2-chap/chapkit-py` base image there is no compiler, and the build dies.

Why was 2.0.1 selected at all? Because step 1 left pandas unpinned. statsforecast 2.1.x
requires `pandas<3`; the resolver preferred pandas 3.0.5 and therefore backed statsforecast
down to the newest release compatible with it, which is 2.0.1. A pandas preference
silently chose an sdist-only version of a different package.

```
uv pip install --dry-run "statsforecast>=2.1"   # -> statsforecast 2.1.1 + pandas 2.3.3
```

statsforecast 2.1.1 ships cp313 wheels for `macosx_10_13_x86_64`, `macosx_11_0_arm64`,
`manylinux_2_28_aarch64`, `manylinux_2_28_x86_64` and `win_amd64` - every platform this
repo targets.

### The fix

```toml
"statsforecast>=2.1.0,<3",
"pandas>=2.2,<3",
```

The pandas cap exists **only** to keep statsforecast on a wheel release. There is no known
pandas 3 incompatibility in this model's own code; the step 1 smoke test and goldens on
pandas 3.0.5 were genuine. The comment in `pyproject.toml` says so, because otherwise a
future maintainer will "helpfully" lift the cap.

```
uv lock          # Updated pandas 3.0.5 -> 2.3.3, statsforecast 2.0.1 -> 2.1.1,
                 # Removed numba 0.67.0, Removed llvmlite 0.49.0
uv sync --all-groups
```

Note the side effect: **statsforecast 2.1.1 no longer depends on numba**, so numba and
llvmlite left the lock entirely. `NUMBA_CACHE_DIR=/tmp/numba_cache` stays in the Dockerfile
anyway - it costs nothing and numba is a plausible future transitive dependency - but its
comment was corrected so it does not claim a dependency that is not there.

### Updated resolved versions (the ones every golden and every number below now reflect)

| Package | Version |
|---|---|
| python | 3.13.14 (CPython, clang 22.1.3) |
| chapkit | 2.0.1 |
| servicekit | 2.0.2 |
| statsforecast | **2.1.1** |
| statsmodels | 0.15.0 |
| utilsforecast | 0.2.15 |
| numba | **not installed** |
| numpy | 2.5.3 |
| pandas | **2.3.3** |
| scipy | 1.18.1 |
| cyclopts | 4.25.2 |
| pydantic | 2.13.5 |
| fastapi | 0.141.1 |

### Wheel availability audit

To make sure statsforecast was the only offender, the whole runtime closure was exported
and checked against PyPI:

```
uv export --frozen --no-dev --no-hashes -o <scratch>/req.txt
```

then, for each of the 96 pinned runtime packages, the PyPI JSON API
(`https://pypi.org/pypi/<name>/<version>/json`) was queried for a wheel installable under
cp313 on manylinux **x86_64 and aarch64** (a `py3-none-any` wheel counts for both).

```
checked 96 pinned runtime packages
OK: 96   offenders: 0
```

The same script flags `statsforecast==2.0.1` and `==2.0.2` as `MISSING` (0 cp313 wheels),
which is the control that proves it works. This check belongs in every migration, run
**before** goldens are produced.

### Goldens regenerated

Both fixtures were regenerated from the legacy worktree with the identical procedure
(see step 2): `__file__` provenance check, `train` then `predict`, monthly `predict` run
twice and `cmp`'d (byte-identical again).

**The regenerated CSVs are byte-identical to the statsforecast 2.0.1 / pandas 3.0.5 ones.**

```
git diff --stat tests/golden    # (no output)
```

| kind | rows | sample cols | cells identical | changed cells | max abs diff | max rel diff |
|---|---|---|---|---|---|---|
| monthly | 216 | 25 | 5400 / 5400 (100.00 %) | 0 | 0.000e+00 | 0.000e+00 |
| weekly | 36 | 25 | 900 / 900 (100.00 %) | 0 | 0.000e+00 | 0.000e+00 |

So the library bump is numerically a no-op here, and the conversion parity result
(5400 / 5400 and 900 / 900 against the legacy CLI) is unaffected. Those are two separate
claims and they are worth keeping separate in the PR: *statsforecast 2.0.1 -> 2.1.1 changed
nothing*, and *the chapkit conversion changed nothing*.

### Re-verified after the bump

```
uv run ruff format --check .   ->  9 files already formatted
uv run ruff check .            ->  All checks passed!
uv run pytest -v               ->  9 passed in 16.15s
monthly: 5400 / 5400 cells exactly equal to the legacy golden output
weekly:   900 /  900 cells exactly equal to the legacy golden output

docker build ...               ->  succeeded in 20 s, everything installed from wheels
make test-docker               ->  chapkit test monthly and weekly, ALL TESTS PASSED
```


---

## Verification results

Run on macOS 27.0 arm64, against `uv run python main.py` on port 9090, at the tip of the
branch.

### Lint

```
uv run ruff format --check .   ->  9 files already formatted
uv run ruff check .            ->  All checks passed!
```

### Tests

```
uv run pytest -v   ->  9 passed in 16.15s

4.66s  test_monthly_reproduces_legacy_golden
3.67s  test_future_row_order_preserved
3.11s  test_unseen_location_fallback
3.09s  test_weekly_reproduces_legacy_golden
1.55s  setup (shared monthly training artifact)
0.01s  everything else

monthly: 5400 / 5400 cells exactly equal to the legacy golden output
weekly:   900 /  900 cells exactly equal to the legacy golden output
```

### `chapkit test`

| Invocation | Result | Elapsed |
|---|---|---|
| `chapkit test --url http://localhost:9090 --timeout 300` | ALL TESTS PASSED (1 config, 1 training, 1 prediction, 2 validations) | 5.40 s |
| `chapkit test --url http://localhost:9090 --period-type weekly --rows 520 --predict-rows 300 --timeout 300` | ALL TESTS PASSED | 6.45 s |

The weekly run needs the larger row counts: at the default `--rows 250` a weekly panel has
under a year of history, which is shorter than the 52-period season length.

### `scripts/parity.py`

Reference = the legacy CLI re-run live:

| kind | rows | sample cols | exactly equal cells | max abs diff | max rel diff | result |
|---|---|---|---|---|---|---|
| monthly | 216 | 25 | 5400 / 5400 (100.00 %) | 0.000e+00 | 0.000e+00 | PASS |
| weekly | 36 | 25 | 900 / 900 (100.00 %) | 0.000e+00 | 0.000e+00 | PASS |

Reference = the committed pre-conversion golden fixture (`--golden`):

| kind | rows | sample cols | exactly equal cells | max abs diff | max rel diff | result |
|---|---|---|---|---|---|---|
| monthly | 216 | 25 | 5400 / 5400 (100.00 %) | 0.000e+00 | 0.000e+00 | PASS |
| weekly | 36 | 25 | 900 / 900 (100.00 %) | 0.000e+00 | 0.000e+00 | PASS |

### `chap eval` (chap-core 2.3.0)

**Blocked by chap-core, not by this service.** With `period_type=PeriodType.any`:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for MLServiceInfo
period_type
  Input should be 'weekly' or 'monthly' [type=enum, input_value='any', input_type=str]
```

`chap_core/rest_api/services/schemas.py` declares
`class PeriodType(StrEnum): weekly; monthly` - no `any` - and
`chapkit_rest_api_wrapper.info()` validates `/api/v1/info` against it, so `chap eval`
aborts before doing any work. chap-core's *other* period-type enum
(`chap_core/model_spec.py`) does have `any`, and chapkit 2.0.1 offers `PeriodType.any`
with the comment "chap-core then skips its period-type check", so this is a gap in
chap-core 2.3.0's chapkit client rather than a wrong declaration here. chap-core 2.3.0 is
the newest release on PyPI as of this writing.

To confirm nothing else in the chap-core path is broken, the service was temporarily
rebuilt with `period_type=PeriodType.monthly` and the same command re-run:

```
chap eval --model-name http://localhost:9090 \
  --dataset-csv example_data/monthly/training_data.csv \
  --output-file <scratch>/eval.nc --run-config.is-chapkit-model \
  --backtest-params.n-splits 2 --backtest-params.n-periods 3
```

That **succeeded end to end** (`POST /api/v1/configs` 201, two `$train` + `$predict`
rounds, artifact downloads, 193 KB `eval.nc` written). The change was reverted; the branch
ships `PeriodType.any`.

### Docker

```
docker build --build-arg GIT_REVISION=$(git rev-parse HEAD) -t chap-mstl-arima:latest .
```

Succeeds in **20 s** on an arm64 host; `uv sync --frozen --no-dev --no-install-project`
installs the whole closure from wheels in 3.8 s, no compiler involved.

```
make test-docker
```

Builds the image, starts it on port 9000, waits for `/health`, then runs `chapkit test`
against the container for both period types. **Both pass** (monthly and weekly,
`Result: ALL TESTS PASSED`, ~6.5 s each); total 15.7 s.

This is the check that failed before step 8 with
`Failed to build statsforecast==2.0.1 ... No such file or directory: 'g++'`.

---

## Gotchas worth carrying to the next migration

1. **`prediction_periods` needs a default.** `BaseConfig` declares it with none and
   chap-core never sends it. Without `Field(default=...)` every chap-core config POST is a
   422.
2. **Hoist `user_option_values`, or use `config_format="chap_core"` and know what it
   does.** `extra="allow"` means a nested dict is accepted silently, the tunables keep
   their defaults, and the run succeeds with the wrong numbers. This is the failure mode
   that a smoke test will not catch and a parity test will.
3. **Capture the baseline before you write a line of service code**, from a worktree of
   the base commit, using the *new* virtualenv, and assert `__file__` points into the
   worktree.
4. **Prove the baseline is deterministic** (run predict twice and `cmp`) before deciding
   what tolerance the parity test should use. If it is deterministic, demand exact
   equality locally and keep a `PARITY_RTOL` env escape hatch for CI.
5. **`pd.read_csv(..., float_precision="round_trip")`** on both sides of any float
   comparison.
6. **Exclude the frozen numeric core from `ruff`** *before* the first `ruff format .`.
   ruff >= 0.16 also formats Python code blocks inside Markdown.
7. **Point `NUMBA_CACHE_DIR` (and `HOME`, `MPLCONFIGDIR`, `XDG_CACHE_HOME`) at `/tmp`**
   for any numba-backed model, or the first job in a read-only container fails.
8. **Test both period types** when the model declares `supported_period_type: any`, and
   remember `chapkit test --period-type weekly` needs `--rows 520 --predict-rows 300`.
9. **Row order is part of the contract** for sampling models. Write the shuffled-future
   test.
10. **chap-core 2.3.0 cannot consume `period_type: "any"`** from a chapkit service. Check
    this before promising `any` to a deployment.
11. **Check that the resolved lock installs from *wheels* on `linux/amd64` and
    `linux/arm64` before you produce any golden file.** "It imports and runs on my macOS
    laptop" is not the same claim: macOS has a compiler, the chapkit base images do not.
    Export the runtime closure (`uv export --frozen --no-dev --no-hashes`) and check each
    pinned version against the PyPI JSON API for a cp313 manylinux wheel on both
    architectures.
12. **A newer transitive preference can silently downgrade a direct dependency to an
    sdist-only release.** Here, leaving pandas unpinned pulled pandas 3, which is
    incompatible with statsforecast 2.1.x, which pushed statsforecast back to 2.0.1 - the
    last release with no cp313 wheels. Nothing warned; the model imported and produced
    correct numbers. When you pin a version *to work around a different package*, say so in
    a comment, or someone will lift the pin.
13. **Run `docker build` early - before the goldens, not after the PR.** It is the only
    check that exercises the target platform, and a dependency change after the goldens
    exist means regenerating and re-justifying them.
