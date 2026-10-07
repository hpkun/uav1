"""Configuration loading for the paper-constrained combat environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

from .models import AircraftSpec


ENVIRONMENT_VERSION = "2.3"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the active V2.3 environment configuration."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {
        "environment_version", "simulation", "action", "aircraft", "arena",
        "scenario", "weapon", "reward", "observation", "blue_policy",
    }
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"configuration must contain: {sorted(required)}")
    if str(config["environment_version"]) != ENVIRONMENT_VERSION:
        raise ValueError(
            f"environment_version must be {ENVIRONMENT_VERSION}, got "
            f"{config['environment_version']}"
        )
    return config


def aircraft_spec(config: dict[str, Any]) -> AircraftSpec:
    return AircraftSpec(**config["aircraft"])


__all__ = ["ENVIRONMENT_VERSION", "aircraft_spec", "load_config"]
