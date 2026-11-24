"""Shared fixtures.

The chapkit app is driven in-process through Starlette's `TestClient`: no port,
no container, no `uv run python main.py` in another terminal. The train and
predict jobs still spawn real `python -m chap_mstl_arima` subprocesses, which is
the whole point - these tests exercise the same shell-runner path production
uses.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

# Must be set BEFORE `main` is imported: main.py reads DATABASE_URL at module
# import time. A file-backed SQLite database (not :memory:) so that the
# background job worker and the polling requests share one database.
_DB_DIR = Path(tempfile.mkdtemp(prefix="chap_mstl_arima_test_"))
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_DB_DIR}/test.db")

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402
from tests.helpers import GOLDEN_DIR  # noqa: E402


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def golden_options() -> dict:
    """The `user_option_values` block the golden fixtures were produced with."""
    raw = yaml.safe_load((GOLDEN_DIR / "config.yaml").read_text())
    return raw["user_option_values"]
