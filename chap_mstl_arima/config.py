from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    n_samples: int = 100
    log_transform: bool = True
    random_seed: int = 42
    arima_approximation: bool = False
    arima_stepwise: bool = True
    treat_missing_as_zero: bool = False

    @classmethod
    def from_user_options(cls, opts: dict) -> "ModelConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in (opts or {}).items() if k in known}
        return cls(**filtered)
