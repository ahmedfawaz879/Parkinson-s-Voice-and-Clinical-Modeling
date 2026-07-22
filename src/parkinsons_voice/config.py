"""Configuration management - load YAML configs with optional overrides.

Centralizes hyperparameters (n_estimators, test_size, random_state, SNR
levels, etc.) so they are defined once in configs/default.yaml rather than
duplicated as magic numbers across modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Locate the built-in default config shipped with the package.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "default.yaml"


class Config:
    """Nested attribute-access wrapper around a plain dict.

    Supports ``cfg.model.classifier.n_estimators`` style access without
    losing the ability to iterate / serialize as a dict.
    """

    def __init__(self, mapping: dict[str, Any]) -> None:
        for key, value in mapping.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert back to a plain dict."""
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def get(self, key: str, default: Any = None) -> Any:
        return self.__dict__.get(key, default)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in-place) and return base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path | None = None) -> Config:
    """Load the default config and optionally merge a user-supplied YAML.

    Parameters
    ----------
    config_path:
        Path to an override YAML file. Values specified there take
        precedence over configs/default.yaml.

    Returns
    -------
    Config
        A nested, attribute-accessible configuration object.
    """
    with open(_DEFAULT_CONFIG, encoding="utf-8") as fh:
        base: dict[str, Any] = yaml.safe_load(fh)

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, encoding="utf-8") as fh:
            overrides: dict[str, Any] = yaml.safe_load(fh) or {}
        base = _deep_merge(base, overrides)

    return Config(base)


def set_global_seed(seed: int) -> None:
    """Seed numpy (and Python's ``random``) so pipeline runs are reproducible."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(cfg: Config) -> None:
    """Create the data/output directories declared in config if missing."""
    for path in (cfg.data.raw_dir, cfg.data.processed_dir, cfg.data.output_dir, cfg.output.artifacts_dir):
        os.makedirs(path, exist_ok=True)
