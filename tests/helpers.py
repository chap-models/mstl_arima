"""HTTP helpers shared by the service tests.

These mirror what chap-core does over the wire, so the tests fail for the same
reasons a real deployment would.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DATA = REPO_ROOT / "example_data"
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"

# The first job in a session pays numba's JIT compilation on top of one
# AutoARIMA fit per location, so this is generous on purpose.
JOB_TIMEOUT_SECONDS = 600


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV without losing the last ulp of any float."""
    return pd.read_csv(path, float_precision="round_trip")


def df_payload(df: pd.DataFrame) -> dict[str, Any]:
    """Serialize a pandas frame into chapkit's DataFrame wire format.

    NaN is not valid JSON; missing values become None, exactly as chap-core
    sends them.
    """
    rows = [
        [None if isinstance(v, float) and np.isnan(v) else v for v in row]
        for row in df.itertuples(index=False, name=None)
    ]
    return {"columns": df.columns.tolist(), "data": rows}


def wait_for_job(client: TestClient, job_id: str, timeout: int = JOB_TIMEOUT_SECONDS) -> None:
    """Poll `GET /api/v1/jobs/{id}` until the job completes, or fail the test."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        status = job.get("status")
        if status == "completed":
            return
        if status == "failed":
            pytest.fail(f"job {job_id} failed: {job.get('error')}")
        time.sleep(0.5)
    pytest.fail(f"job {job_id} did not complete within {timeout}s")


def create_config(client: TestClient, name: str, user_option_values: dict[str, Any]) -> str:
    """Create a config using the chap-core-shaped (nested) request body."""
    response = client.post(
        "/api/v1/configs",
        json={"name": name, "data": {"user_option_values": user_option_values}},
    )
    assert response.status_code in (200, 201), response.text
    return response.json()["id"]


def train(client: TestClient, config_id: str, data: pd.DataFrame) -> str:
    """Run a training job and return the training artifact id."""
    response = client.post("/api/v1/ml/$train", json={"config_id": config_id, "data": df_payload(data)})
    assert response.status_code in (200, 202), response.text
    body = response.json()
    wait_for_job(client, body["job_id"])
    return body["artifact_id"]


def predict(client: TestClient, artifact_id: str, historic: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Run a prediction job and return the downloaded predictions."""
    response = client.post(
        "/api/v1/ml/$predict",
        json={
            "artifact_id": artifact_id,
            "historic": df_payload(historic),
            "future": df_payload(future),
        },
    )
    assert response.status_code in (200, 202), response.text
    body = response.json()
    wait_for_job(client, body["job_id"])

    download = client.get(f"/api/v1/artifacts/{body['artifact_id']}/$download")
    assert download.status_code == 200, download.text
    return pd.DataFrame(download.json())


def sample_columns(df: pd.DataFrame) -> list[str]:
    """Return the `sample_*` columns, in order."""
    return [c for c in df.columns if c.startswith("sample_")]
