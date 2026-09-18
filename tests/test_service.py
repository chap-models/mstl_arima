"""In-process tests for the chapkit service, including golden parity.

The parity tests are the reason this file exists: `tests/golden/*.csv` were
produced by the pre-chapkit MLproject CLI, and the service must reproduce them.
See `tests/golden/VERSIONS.md` for how they were captured.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import yaml
from chapkit.ml.runner import dump_config_yaml
from fastapi.testclient import TestClient

from chap_mstl_arima.config import ModelConfig
from main import MSTLArimaConfig
from tests.helpers import (
    EXAMPLE_DATA,
    GOLDEN_DIR,
    create_config,
    predict,
    read_csv,
    sample_columns,
    train,
)

# Exact equality holds on the machine the fixtures were produced on. The
# tolerances exist so a Linux CI runner with a different BLAS / libm cannot fail
# on a last-ulp difference. The absolute tolerance matters for samples that are
# a few thousandths of a case: expm1 of a tiny log-scale draw differs by ~5e-8
# across platforms, which is ~1e-5 relative but physically nothing. Shape,
# columns and row order are never tolerant.
PARITY_RTOL = float(os.getenv("PARITY_RTOL", "1e-6"))
PARITY_ATOL = float(os.getenv("PARITY_ATOL", "1e-6"))

KINDS = {
    "monthly": (EXAMPLE_DATA / "monthly", GOLDEN_DIR / "lao_monthly_predictions.csv"),
    "weekly": (EXAMPLE_DATA / "weekly", GOLDEN_DIR / "nicaragua_weekly_predictions.csv"),
}


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def monthly_artifact(client: TestClient, golden_options: dict) -> str:
    """One training artifact for the monthly panel, reused by several tests."""
    config_id = create_config(client, "pytest-monthly", golden_options)
    return train(client, config_id, read_csv(EXAMPLE_DATA / "monthly" / "training_data.csv"))


# --------------------------------------------------------------------------- #
# service contract
# --------------------------------------------------------------------------- #


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_info(client: TestClient) -> None:
    info = client.get("/api/v1/info").json()
    assert info["id"] == "chap-mstl-arima"
    assert info["display_name"] == "MSTL + AutoARIMA"
    # MLproject: supported_period_type: any
    assert info["period_type"] == "any"
    # MLproject: required_covariates: [], allow_free_additional_continuous_covariates: false
    assert info["required_covariates"] == []
    assert info["allow_free_additional_continuous_covariates"] is False
    assert info["model_metadata"]["author"] == "Knut Rand"
    assert info["model_metadata"]["author_assessed_status"] == "yellow"


def test_config_schema_defaults(client: TestClient) -> None:
    """Every surviving MLproject user_option is exposed with its MLproject default."""
    schema = client.get("/api/v1/configs/$schema").json()
    properties = schema["properties"]

    expected = {
        "prediction_periods": 3,
        "n_samples": 100,
        "log_transform": True,
        "random_seed": 42,
        "arima_approximation": False,
        "arima_stepwise": True,
        "treat_missing_as_zero": False,
    }
    for field, default in expected.items():
        assert field in properties, f"{field} missing from config schema"
        assert properties[field]["default"] == default, field
        assert properties[field].get("description"), f"{field} has no description"

    # The season_length_* knobs were debug-only and are gone; the seasonal period
    # is fixed at 52 for weekly data and 12 for monthly.
    assert "season_length_monthly" not in properties
    assert "season_length_weekly" not in properties

    # No field may be required: chap-core posts only user_option_values.
    assert schema.get("required", []) == []


# --------------------------------------------------------------------------- #
# config handling
# --------------------------------------------------------------------------- #


def test_config_hoists_user_option_values(client: TestClient) -> None:
    """A chap-core-shaped POST body must land on the declared fields."""
    response = client.post(
        "/api/v1/configs",
        json={"name": "pytest-hoist", "data": {"user_option_values": {"n_samples": 7}}},
    )
    assert response.status_code in (200, 201), response.text
    config_id = response.json()["id"]

    data = client.get(f"/api/v1/configs/{config_id}").json()["data"]
    assert data["n_samples"] == 7
    # chap-core never sends prediction_periods; BaseConfig declares it without a
    # default, so the service has to supply one or every POST is a 422.
    assert data["prediction_periods"] == 3
    # The nested dict must not survive as an extra field.
    assert "user_option_values" not in data


def test_legacy_season_length_options_are_accepted_and_ignored(client: TestClient) -> None:
    """A stored chap-core configuration may still carry the removed season_length knobs.

    It has to be accepted (`BaseConfig` allows extra fields) and it has to not reach
    the model. This walks the whole config path - HTTP body, stored config, the
    config.yml the shell runner writes, and the ModelConfig the script builds from it -
    without paying for a train/predict job.
    """
    response = client.post(
        "/api/v1/configs",
        json={
            "name": "pytest-legacy-season-length",
            "data": {"user_option_values": {"season_length_monthly": 6, "n_samples": 7}},
        },
    )
    assert response.status_code in (200, 201), response.text

    data = client.get(f"/api/v1/configs/{response.json()['id']}").json()["data"]
    assert data["n_samples"] == 7
    assert data["season_length_monthly"] == 6, "extra fields must survive, not 422"

    # It is carried into config.yml under user_option_values, like any other key ...
    written = yaml.safe_load(dump_config_yaml(MSTLArimaConfig.model_validate(data), "chap_core"))
    assert written["user_option_values"]["season_length_monthly"] == 6

    # ... and ModelConfig.from_user_options drops it, so the model never sees it.
    model_config = ModelConfig.from_user_options(written["user_option_values"])
    assert model_config.n_samples == 7
    assert not hasattr(model_config, "season_length_monthly")


def test_config_flat_fields_win_over_nested() -> None:
    """`chapkit test` posts flat fields; chap-core posts nested ones. Both work."""
    assert MSTLArimaConfig.model_validate({"n_samples": 11}).n_samples == 11
    assert MSTLArimaConfig.model_validate({"user_option_values": {"n_samples": 7}}).n_samples == 7
    both = MSTLArimaConfig.model_validate({"n_samples": 3, "user_option_values": {"n_samples": 7}})
    assert both.n_samples == 3


# --------------------------------------------------------------------------- #
# golden parity
# --------------------------------------------------------------------------- #


def _assert_matches_golden(predictions: pd.DataFrame, golden: pd.DataFrame, label: str) -> None:
    assert list(predictions.columns) == list(golden.columns), f"{label}: column list changed"
    assert len(predictions) == len(golden), f"{label}: row count changed"

    for key in ("time_period", "location"):
        assert predictions[key].astype(str).tolist() == golden[key].astype(str).tolist(), (
            f"{label}: {key} order changed"
        )

    cols = sample_columns(golden)
    assert cols, f"{label}: no sample columns in the golden fixture"

    got = predictions[cols].to_numpy(dtype=float)
    want = golden[cols].to_numpy(dtype=float)

    exact = int(np.count_nonzero(got == want))
    print(f"\n{label}: {exact} / {got.size} cells exactly equal to the legacy golden output")

    np.testing.assert_allclose(got, want, rtol=PARITY_RTOL, atol=PARITY_ATOL)


def _run_golden(client: TestClient, golden_options: dict, kind: str) -> None:
    data_dir, golden_path = KINDS[kind]

    config_id = create_config(client, f"pytest-golden-{kind}", golden_options)
    artifact_id = train(client, config_id, read_csv(data_dir / "training_data.csv"))
    predictions = predict(
        client,
        artifact_id,
        read_csv(data_dir / "historic_data.csv"),
        read_csv(data_dir / "future_data.csv"),
    )

    _assert_matches_golden(predictions, read_csv(golden_path), kind)


def test_monthly_reproduces_legacy_golden(client: TestClient, golden_options: dict) -> None:
    """The service must reproduce the pre-chapkit CLI's monthly predictions."""
    _run_golden(client, golden_options, "monthly")


def test_weekly_reproduces_legacy_golden(client: TestClient, golden_options: dict) -> None:
    """The service must reproduce the pre-chapkit CLI's weekly predictions."""
    _run_golden(client, golden_options, "weekly")


# --------------------------------------------------------------------------- #
# behaviour the shell-runner round trip could plausibly break
# --------------------------------------------------------------------------- #


def test_unseen_location_fallback(client: TestClient, monthly_artifact: str) -> None:
    """A location absent from training still gets finite, non-negative samples."""
    historic = read_csv(EXAMPLE_DATA / "monthly" / "historic_data.csv")
    future = read_csv(EXAMPLE_DATA / "monthly" / "future_data.csv")

    extra = future.iloc[[0]].copy()
    extra["location"] = "ZZ-NEW"
    future = pd.concat([future, extra], ignore_index=True)

    predictions = predict(client, monthly_artifact, historic, future)

    assert len(predictions) == len(future)
    unseen = predictions[predictions["location"] == "ZZ-NEW"]
    assert len(unseen) == 1

    values = unseen[sample_columns(predictions)].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert (values >= 0).all()


def test_future_row_order_preserved(client: TestClient, monthly_artifact: str) -> None:
    """Output rows come back in the order the future frame was posted in.

    This is load bearing for parity: the model draws `n_samples` values per
    future row in iteration order from a single seeded generator, so a reordered
    future frame produces different numbers per row. chap-core matches rows
    positionally, so a runner that sorted the frame would corrupt results
    without failing anything.
    """
    historic = read_csv(EXAMPLE_DATA / "monthly" / "historic_data.csv")
    future = (
        read_csv(EXAMPLE_DATA / "monthly" / "future_data.csv")
        .sample(frac=1.0, random_state=1234)
        .reset_index(drop=True)
    )

    predictions = predict(client, monthly_artifact, historic, future)

    assert predictions["time_period"].astype(str).tolist() == future["time_period"].astype(str).tolist()
    assert predictions["location"].astype(str).tolist() == future["location"].astype(str).tolist()
