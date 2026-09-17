# Golden prediction fixtures

These CSVs are the **legacy** (pre-chapkit) predictions. They are the reference the
chapkit service is held against by `tests/test_service.py` and `scripts/parity.py`.

## How they were produced

They were produced by the legacy code at commit `20ef2f6` - the `MLproject` cyclopts CLI
in the repo's old root `main.py` - checked out into a separate git worktree so that the
branch's own (converted) code could not be picked up by accident:

```
git worktree add ../mstl_arima-baseline 20ef2f6
```

The legacy code was then run from that worktree, but with **the branch's virtualenv**, so
that library versions are identical on both sides of the comparison and any diff is
attributable to the conversion rather than to a dependency bump:

```
cd ../mstl_arima-baseline
uv run --project ../mstl_arima python -c \
  "import chap_mstl_arima; print(chap_mstl_arima.__file__)"
# must print .../mstl_arima-baseline/chap_mstl_arima/__init__.py
```

`uv run --project <other-dir>` keeps the current working directory, and cwd is first on
`sys.path`, so the worktree's own package shadows the one installed in the venv. The
`__file__` assertion above is mandatory: if it prints the branch path, the "baseline" is
not a baseline. (Fallback if it ever does: `PYTHONPATH=$WT` or
`uv run --project $REPO --directory $WT`.)

### Monthly (`lao_monthly_predictions.csv`)

```
uv run --project ../mstl_arima python main.py train \
  ../mstl_arima/example_data/monthly/training_data.csv \
  <scratch>/monthly_model.json \
  ../mstl_arima/tests/golden/config.yaml

uv run --project ../mstl_arima python main.py predict \
  <scratch>/monthly_model.json \
  ../mstl_arima/example_data/monthly/historic_data.csv \
  ../mstl_arima/example_data/monthly/future_data.csv \
  <scratch>/monthly_a.csv \
  ../mstl_arima/tests/golden/config.yaml
```

216 rows (18 Lao admin1 locations x 12 monthly periods, 2009-09 .. 2009-12 onward),
27 columns (`time_period`, `location`, `sample_0` .. `sample_24`).

### Weekly (`nicaragua_weekly_predictions.csv`)

Same two commands against `example_data/weekly/`. 36 rows (3 Nicaraguan departments x 12
weekly periods, `YYYY-MM-DD/YYYY-MM-DD` period format), 27 columns.

## Determinism

`predict` was run twice, into two different files, from the same model marker and the
same config. `cmp` reports the two files **byte-identical**. The model seeds
`np.random.default_rng(random_seed)` inside `MSTLArimaModel.predict` and draws per
`future_df` row in input order, so the output is reproducible given the same library
versions, the same config, and the same future row order.

## Config

`tests/golden/config.yaml`:

```yaml
user_option_values:
  n_samples: 25
  random_seed: 42
```

`n_samples` is 25 rather than the default 100 purely to keep the fixture CSVs small;
25 columns is still enough to catch any drift in the RNG stream. All other options are
left at their `ModelConfig` defaults.

## Environment

| Item | Value |
|---|---|
| Legacy commit | `20ef2f6` |
| Platform | macOS 27.0 (Darwin), arm64 |
| Python | 3.13.14 (CPython, clang 22.1.3) |
| statsforecast | 2.1.1 |
| statsmodels | 0.15.0 |
| utilsforecast | 0.2.15 |
| numba | not installed (statsforecast 2.1.1 does not depend on it) |
| numpy | 2.5.3 |
| pandas | 2.3.3 |
| scipy | 1.18.1 |
| chapkit | 2.0.1 |

### Note on the statsforecast 2.0.1 -> 2.1.1 bump

These fixtures were first produced with statsforecast 2.0.1 / pandas 3.0.5 / numba 0.67.0,
and regenerated with statsforecast 2.1.1 / pandas 2.3.3 / no numba after the dependency
pin changed (statsforecast 2.0.x ships no cp313 wheels, so the Docker build could not
install it). The regenerated CSVs are **byte-identical** to the originals: 5400 / 5400 and
900 / 900 sample cells unchanged, max absolute difference 0. The library bump did not move
the numbers.

## Tolerance

On this machine and this venv the chapkit service reproduces these numbers **exactly**
(bit-for-bit). The tests nevertheless compare with
`np.testing.assert_allclose(rtol=PARITY_RTOL, atol=PARITY_ATOL)` with both env-driven
defaults at `1e-6`, so that a Linux CI runner with a different BLAS or libm cannot fail on
last-ulp differences. The absolute tolerance was added after the first CI run on
ubuntu x86_64 showed 4 of 5400 monthly cells off by at most 5.2e-8 (values around 0.003
cases, 8.5e-6 relative). Shape, column
list and `(time_period, location)` row order are always compared exactly - those must
never drift.

## Regenerating

Do not regenerate these casually: they are the contract. If a dependency bump genuinely
changes the numbers, regenerate from the legacy worktree with the same procedure, update
the version table above, and say so in the commit message.
