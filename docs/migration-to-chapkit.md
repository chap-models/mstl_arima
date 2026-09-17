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
