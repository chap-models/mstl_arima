"""Chapkit ML service for the MSTL + AutoARIMA baseline.

This file replaces the repository's old `MLproject` + cyclopts CLI boot path.
The modelling code is untouched: `ShellModelRunner` shells out to the very same
entry points the MLproject declared, now reachable as
`python -m chap_mstl_arima`. See `docs/migration-to-chapkit.md`.
"""

import os
from pathlib import Path

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo, ModelMetadata, PeriodType
from chapkit.artifact import ArtifactHierarchy
from chapkit.ml import ShellModelRunner
from pydantic import Field, model_validator


class MSTLArimaConfig(BaseConfig):
    """Tunables for a single train/predict run.

    The field set is the `user_options` block of the old `MLproject`, minus the
    two `season_length_*` debug knobs (the seasonal period is now fixed at 52 for
    weekly data and 12 for monthly), plus `prediction_periods`, which chapkit's
    `BaseConfig` declares as a required CHAP-interpreted field.

    `BaseConfig` allows extra fields, so a stored chap-core configuration that
    still carries `season_length_monthly` is accepted; `ModelConfig.from_user_options`
    filters unknown keys, so it never reaches the model.
    """

    prediction_periods: int = Field(
        default=3,
        description="Number of periods to predict into the future",
    )
    n_samples: int = Field(
        default=100,
        description="Number of probabilistic samples",
    )
    log_transform: bool = Field(
        default=True,
        description="Fit on log1p(disease_cases)",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for sampling",
    )
    arima_approximation: bool = Field(
        default=False,
        description="AutoARIMA approximation (faster, drops MA - less calibrated)",
    )
    arima_stepwise: bool = Field(
        default=True,
        description="AutoARIMA stepwise search",
    )
    treat_missing_as_zero: bool = Field(
        default=False,
        description="Treat missing (NaN) target values as 0 reported cases instead of dropping them",
    )

    @model_validator(mode="before")
    @classmethod
    def _hoist_user_option_values(cls, data: object) -> object:
        """Accept chap-core's nested `user_option_values` payload as flat fields.

        chap-core posts a config as `{"name": ..., "user_option_values": {...}}`.
        `BaseConfig` sets `extra="allow"`, so without this hook the dict would be
        stored verbatim as an unknown extra field, every declared tunable would
        keep its default, and `dump_config_yaml(..., "chap_core")` would then emit
        `user_option_values: {user_option_values: {...}, n_samples: 100, ...}` -
        the script would read the defaults and silently ignore what was requested.

        Flat keys win over nested ones, so `chapkit test` (which posts flat
        fields) and the chap-core shape both behave identically.
        """
        if isinstance(data, dict) and isinstance(data.get("user_option_values"), dict):
            hoisted = {k: v for k, v in data.items() if k != "user_option_values"}
            for key, value in data["user_option_values"].items():
                hoisted.setdefault(key, value)
            return hoisted
        return data


# The commands below are the MLproject entry points verbatim, with the MLproject
# parameter names replaced by the placeholders chapkit substitutes. `model.json`
# and `config.yml` are plain workspace-relative filenames: chapkit always writes
# config.yml into the workspace, and the train workspace (model.json included) is
# restored before predict runs.
#
# config_format="chap_core" nests the tunables under `user_option_values:` in
# config.yml, which is exactly the layout the legacy `_load_config` already
# understands - so chap_mstl_arima/cli.py needed no changes at all.
runner: ShellModelRunner[MSTLArimaConfig] = ShellModelRunner(
    train_command="python -m chap_mstl_arima train {data_file} model.json config.yml",
    predict_command=(
        "python -m chap_mstl_arima predict model.json {historic_file} {future_file} {output_file} config.yml"
    ),
    config_format="chap_core",
)

# Straight translation of the MLproject `meta_data` block.
info = MLServiceInfo(
    id="chap-mstl-arima",
    display_name="MSTL + AutoARIMA",
    version="0.2.0",
    description=(
        "MSTL + AutoARIMA baseline for chap. Designed for small, noisy, highly-seasonal "
        "surveillance data. Self-forecasts seasonality and does not depend on future covariates."
    ),
    model_metadata=ModelMetadata(
        author="Knut Rand",
        author_note=(
            "Two-step pipeline. STL (LOESS-smoothed seasonal-trend decomposition) strips the "
            "seasonal component, an AutoARIMA model forecasts the trend+remainder, and the last "
            "observed seasonal cycle is extrapolated forward. Models are fit on "
            "log1p(disease_cases) and probabilistic samples are drawn from the parametric "
            "predictive Normal."
        ),
        author_assessed_status=AssessedStatus.yellow,
        contact_email="knutdrand@gmail.com",
        organization="HISP Centre, University of Oslo",
        organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
        citation_info=(
            'Climate Health Analytics Platform. 2026. "MSTL + AutoARIMA Model". HISP Centre, University of Oslo.'
        ),
    ),
    # MLproject: supported_period_type: any. The model detects monthly vs weekly
    # from the time_period format and picks season_length accordingly.
    period_type=PeriodType.any,
    min_prediction_periods=1,
    # 104 = two years of weekly periods; beyond that the seasonal extrapolation
    # is repeating the same cycle too many times to be meaningful.
    max_prediction_periods=104,
    # MLproject: allow_free_additional_continuous_covariates: false, required_covariates: [].
    # The model is univariate on disease_cases and ignores covariate columns entirely.
    allow_free_additional_continuous_covariates=False,
    required_covariates=[],
)

hierarchy = ArtifactHierarchy(
    name="chap_mstl_arima",
    level_labels={0: "ml_training_workspace", 1: "ml_prediction"},
)

# Uses the DATABASE_URL environment variable or defaults to data/chapkit.db,
# creating the directory when it does not exist.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///data/chapkit.db")
if DATABASE_URL.startswith("sqlite") and ":///" in DATABASE_URL:
    db_path = Path(DATABASE_URL.split("///")[1])
    db_path.parent.mkdir(parents=True, exist_ok=True)

app = (
    MLServiceBuilder(
        info=info,
        config_schema=MSTLArimaConfig,
        hierarchy=hierarchy,
        runner=runner,
        database_url=DATABASE_URL,
    )
    .with_monitoring()
    # See compose.yml for the chap-core self-registration environment variables.
    .with_registration(keepalive_interval=15)
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    # Port 9090 matches the compose host port and avoids the usual busy ports.
    run_app("main:app", reload=False, port=9090)
