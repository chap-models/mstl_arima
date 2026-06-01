from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_samples: int = 100
    log_transform: bool = True
    season_length_monthly: int = 12
    season_length_weekly: int = 52
    random_seed: int = 42
    arima_approximation: bool = False
    arima_stepwise: bool = True

    @classmethod
    def from_user_options(cls, opts: dict) -> "ModelConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in (opts or {}).items() if k in known}
        return cls(**filtered)
