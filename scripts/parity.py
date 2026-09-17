#!/usr/bin/env python
"""Numeric parity harness: legacy CLI predictions vs the chapkit service.

The chapkit service shells out to the very same CLI the legacy MLproject used
(`python -m chap_mstl_arima predict ...`), so the two sides should agree
bit-for-bit on the same machine and virtualenv. This script proves it, cell by
cell, and is the gate that a conversion PR has to pass.

Usage:

    # service must be running, e.g. `uv run python main.py`
    uv run python scripts/parity.py --url http://localhost:9090 --kind monthly
    uv run python scripts/parity.py --url http://localhost:9090 --kind weekly

    # compare against the committed legacy fixtures instead of re-running the CLI
    uv run python scripts/parity.py --url http://localhost:9090 --kind monthly --golden

Exits 1 when any cell violates |candidate - reference| <= atol + rtol * |reference|
(--rtol / --atol, env PARITY_RTOL / PARITY_ATOL, both default 1e-6).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DATA = REPO_ROOT / "example_data"
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"

KINDS: dict[str, dict[str, Path]] = {
    "monthly": {
        "training": EXAMPLE_DATA / "monthly" / "training_data.csv",
        "historic": EXAMPLE_DATA / "monthly" / "historic_data.csv",
        "future": EXAMPLE_DATA / "monthly" / "future_data.csv",
        "golden": GOLDEN_DIR / "lao_monthly_predictions.csv",
    },
    "weekly": {
        "training": EXAMPLE_DATA / "weekly" / "training_data.csv",
        "historic": EXAMPLE_DATA / "weekly" / "historic_data.csv",
        "future": EXAMPLE_DATA / "weekly" / "future_data.csv",
        "golden": GOLDEN_DIR / "nicaragua_weekly_predictions.csv",
    },
}

KEY_COLUMNS = ["time_period", "location"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV without losing the last ulp of any float."""
    return pd.read_csv(path, float_precision="round_trip")


def df_payload(df: pd.DataFrame) -> dict[str, Any]:
    """Serialize a pandas DataFrame the way chapkit's DataFrame schema expects.

    NaN is not valid JSON, so missing values become None - which is exactly what
    chap-core sends too.
    """
    rows = [
        [None if isinstance(v, float) and np.isnan(v) else v for v in row]
        for row in df.itertuples(index=False, name=None)
    ]
    return {"columns": df.columns.tolist(), "data": rows}


def user_option_values(config_path: Path) -> dict[str, Any]:
    """Read the tunables out of a chap-core style config.yaml."""
    raw = yaml.safe_load(config_path.read_text()) or {}
    return raw.get("user_option_values") or raw.get("user_options") or raw


# --------------------------------------------------------------------------- #
# reference side: the legacy CLI
# --------------------------------------------------------------------------- #


def legacy_predictions(kind: str, config_path: Path) -> pd.DataFrame:
    """Run the legacy CLI (`python -m chap_mstl_arima`) and return its predictions."""
    paths = KINDS[kind]
    with tempfile.TemporaryDirectory(prefix=f"parity_{kind}_") as tmp:
        tmpdir = Path(tmp)
        model = tmpdir / "model.json"
        out = tmpdir / "predictions.csv"
        for cmd in (
            [sys.executable, "-m", "chap_mstl_arima", "train", str(paths["training"]), str(model), str(config_path)],
            [
                sys.executable,
                "-m",
                "chap_mstl_arima",
                "predict",
                str(model),
                str(paths["historic"]),
                str(paths["future"]),
                str(out),
                str(config_path),
            ],
        ):
            subprocess.run(cmd, check=True, cwd=REPO_ROOT, capture_output=True, text=True)
        return read_csv(out)


# --------------------------------------------------------------------------- #
# candidate side: the chapkit service over HTTP
# --------------------------------------------------------------------------- #


def wait_for_job(client: httpx.Client, job_id: str, timeout: float) -> None:
    """Poll GET /api/v1/jobs/{id} until the job leaves the running state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        status = job.get("status")
        if status == "completed":
            return
        if status == "failed":
            raise SystemExit(f"job {job_id} failed: {job.get('error')}")
        time.sleep(1.0)
    raise SystemExit(f"job {job_id} did not complete within {timeout:.0f}s")


def service_predictions(url: str, kind: str, config_path: Path, timeout: float) -> pd.DataFrame:
    """Drive the running chapkit service end to end and return its predictions."""
    paths = KINDS[kind]
    training = read_csv(paths["training"])
    historic = read_csv(paths["historic"])
    future = read_csv(paths["future"])

    with httpx.Client(base_url=url, timeout=60.0) as client:
        # chap-core posts configs in this shape: a name plus a user_option_values
        # dict. chapkit 2.1.0 hoists those keys onto the config, so this must behave exactly like
        # posting them flat.
        config = client.post(
            "/api/v1/configs",
            json={"name": f"parity-{kind}", "data": {"user_option_values": user_option_values(config_path)}},
        )
        config.raise_for_status()
        config_id = config.json()["id"]

        train = client.post(
            "/api/v1/ml/$train",
            json={"config_id": config_id, "data": df_payload(training)},
        )
        train.raise_for_status()
        train_body = train.json()
        wait_for_job(client, train_body["job_id"], timeout)
        training_artifact_id = train_body["artifact_id"]

        predict = client.post(
            "/api/v1/ml/$predict",
            json={
                "artifact_id": training_artifact_id,
                "historic": df_payload(historic),
                "future": df_payload(future),
            },
        )
        predict.raise_for_status()
        predict_body = predict.json()
        wait_for_job(client, predict_body["job_id"], timeout)

        download = client.get(f"/api/v1/artifacts/{predict_body['artifact_id']}/$download")
        download.raise_for_status()
        return pd.DataFrame(download.json())


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #


def compare(reference: pd.DataFrame, candidate: pd.DataFrame, kind: str, rtol: float, atol: float) -> tuple[str, bool]:
    """Compare two prediction frames and return a markdown row plus a pass flag."""
    if list(reference.columns) != list(candidate.columns):
        raise SystemExit(
            f"[{kind}] column mismatch\n  reference: {list(reference.columns)}\n  candidate: {list(candidate.columns)}"
        )
    if len(reference) != len(candidate):
        raise SystemExit(f"[{kind}] row count mismatch: reference {len(reference)}, candidate {len(candidate)}")

    for key in KEY_COLUMNS:
        ref_keys = reference[key].astype(str).tolist()
        cand_keys = candidate[key].astype(str).tolist()
        if ref_keys != cand_keys:
            first = next(i for i, (a, b) in enumerate(zip(ref_keys, cand_keys)) if a != b)
            raise SystemExit(
                f"[{kind}] {key} order mismatch at row {first}: reference {ref_keys[first]!r}, "
                f"candidate {cand_keys[first]!r}"
            )

    sample_cols = [c for c in reference.columns if c.startswith("sample_")]
    if not sample_cols:
        raise SystemExit(f"[{kind}] no sample_* columns found")

    ref = reference[sample_cols].to_numpy(dtype=float)
    cand = candidate[sample_cols].to_numpy(dtype=float)

    diff = np.abs(cand - ref)
    denom = np.abs(ref)
    rel = np.zeros_like(diff)
    nonzero = denom > 0
    rel[nonzero] = diff[nonzero] / denom[nonzero]
    rel[(~nonzero) & (diff > 0)] = np.inf

    total = ref.size
    exact = int(np.count_nonzero(cand == ref))
    max_abs = float(diff.max()) if total else 0.0
    max_rel = float(rel.max()) if total else 0.0

    # Same rule as numpy.testing.assert_allclose: |cand - ref| <= atol + rtol * |ref|.
    ok = bool(np.all(diff <= atol + rtol * denom))
    row = (
        f"| {kind} | {len(reference)} | {len(sample_cols)} | {exact} / {total} "
        f"({100.0 * exact / total:.2f} %) | {max_abs:.3e} | {max_rel:.3e} | {'PASS' if ok else 'FAIL'} |"
    )
    return row, ok


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:9090", help="base URL of the running chapkit service")
    parser.add_argument("--kind", choices=sorted(KINDS), default="monthly", help="which example dataset to use")
    parser.add_argument(
        "--golden",
        action="store_true",
        help="compare against the committed legacy fixture instead of re-running the legacy CLI",
    )
    parser.add_argument("--config", type=Path, default=GOLDEN_DIR / "config.yaml", help="model config YAML")
    parser.add_argument(
        "--rtol",
        type=float,
        default=float(os.getenv("PARITY_RTOL", "1e-6")),
        help="maximum tolerated relative difference (env PARITY_RTOL)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=float(os.getenv("PARITY_ATOL", "1e-6")),
        help="absolute tolerance added to the relative one, in cases (env PARITY_ATOL)",
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="per-job timeout in seconds")
    args = parser.parse_args()

    if args.golden:
        reference = read_csv(KINDS[args.kind]["golden"])
        source = f"legacy golden fixture ({KINDS[args.kind]['golden'].relative_to(REPO_ROOT)})"
    else:
        reference = legacy_predictions(args.kind, args.config)
        source = "legacy CLI (python -m chap_mstl_arima)"

    candidate = service_predictions(args.url, args.kind, args.config, args.timeout)

    row, ok = compare(reference, candidate, args.kind, args.rtol, args.atol)

    print(f"reference: {source}")
    print(f"candidate: chapkit service at {args.url}")
    print(f"rtol: {args.rtol:g}  atol: {args.atol:g}")
    print()
    print("| kind | rows | sample cols | exactly equal cells | max abs diff | max rel diff | result |")
    print("|---|---|---|---|---|---|---|")
    print(row)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
