"""Command line entry points for the MSTL + AutoARIMA baseline.

The ``train`` command is a no-op marker file — StatsForecast models are
cheap to fit and chap's backtester retrains per split, so we refit
inside ``predict`` against ``historic_data``. The model file is still
written so chap sees the expected artifact.

This module is the legacy MLproject CLI, moved here verbatim from the
repository's old root ``main.py``. It is what the chapkit service in the
new root ``main.py`` shells out to via ``ShellModelRunner``:

    python -m chap_mstl_arima train {data_file} model.json config.yml
    python -m chap_mstl_arima predict model.json {historic_file} \
        {future_file} {output_file} config.yml

It also remains usable on its own: ``uv run python -m chap_mstl_arima``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cyclopts
import pandas as pd
import yaml

from chap_mstl_arima.config import ModelConfig
from chap_mstl_arima.model import MSTLArimaModel

logger = logging.getLogger("chap_mstl_arima")
logger.setLevel(logging.INFO)

app = cyclopts.App()


def _load_config(model_config: str | None) -> ModelConfig:
    if not model_config:
        return ModelConfig()
    p = Path(model_config)
    if not p.exists() or p.stat().st_size == 0:
        return ModelConfig()
    text = p.read_text()
    if p.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        return ModelConfig()
    user_opts = raw.get("user_option_values") or raw.get("user_options") or raw
    return ModelConfig.from_user_options(user_opts or {})


@app.command()
def train(train_data: str, model: str, model_config: str = "") -> None:
    cfg = _load_config(model_config or None)
    Path(model).write_text(json.dumps(cfg.__dict__))
    logger.info("Wrote model marker to %s (config=%s)", model, cfg.__dict__)


@app.command()
def predict(
    model: str,
    historic_data: str,
    future_data: str,
    out_file: str,
    model_config: str = "",
) -> None:
    cfg_from_train = ModelConfig()
    mp = Path(model)
    if mp.exists() and mp.stat().st_size > 0:
        try:
            cfg_from_train = ModelConfig(**json.loads(mp.read_text()))
        except Exception:
            pass
    cfg = _load_config(model_config or None) if model_config else cfg_from_train

    historic_df = pd.read_csv(historic_data)
    future_df = pd.read_csv(future_data)
    logger.info(
        "predict: historic=%d rows, future=%d rows, locations=%d, cfg=%s",
        len(historic_df),
        len(future_df),
        historic_df["location"].nunique(),
        cfg.__dict__,
    )

    out = MSTLArimaModel(cfg).predict(historic_df, future_df)
    out.to_csv(out_file, index=False)
    logger.info("Wrote %d prediction rows to %s", len(out), out_file)


def main() -> None:
    """Configure logging and dispatch to the cyclopts app."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app()


if __name__ == "__main__":
    main()
