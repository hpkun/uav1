"""Configuration loading for the paper-constrained combat environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

from .models import AircraftSpec


ENVIRONMENT_VERSION = "2.3"
SUPPORTED_ENVIRONMENT_VERSIONS = frozenset({"2.3", "2.4"})


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the active V2.3 environment configuration."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return validate_config(config)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Reject unsupported schemas before constructing a combat environment."""
    required = {
        "environment_version", "simulation", "action", "aircraft", "arena",
        "scenario", "weapon", "reward", "observation", "blue_policy",
    }
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"configuration must contain: {sorted(required)}")
    unknown = set(config) - required
    if unknown:
        raise ValueError(f"unknown combat configuration fields: {sorted(unknown)}")
    observation_fields = {
        "horizontal_position_scale", "altitude_scale", "speed_center", "speed_scale",
        "relative_position_scale",
    }
    unknown_observation = set(config["observation"]) - observation_fields
    if unknown_observation:
        raise ValueError(f"unknown observation fields: {sorted(unknown_observation)}")
    version = str(config["environment_version"])
    if version not in SUPPORTED_ENVIRONMENT_VERSIONS:
        raise ValueError(
            f"environment_version must be 2.3 or 2.4, got "
            f"{config['environment_version']}"
        )
    expected_team_size = {"2.3": 4, "2.4": 8}[version]
    if config["scenario"].get("team_size") != expected_team_size:
        raise ValueError(f"environment_version {version} requires team_size={expected_team_size}")
    if len(config["scenario"].get("formation_offsets", [])) != expected_team_size:
        raise ValueError("formation_offsets length must equal version-specific team_size")
    weapon_fields = {"range_min", "range_max", "off_boresight_angle_max",
                     "effective_hit_distance", "attack_noise_scale", "height_noise_scale"}
    if version == "2.4":
        weapon_fields.add("target_aspect_angle_max")
    if set(config["weapon"]) != weapon_fields:
        raise ValueError(f"weapon schema mismatch for environment_version {version}")
    if version == "2.4" and not 0 < float(config["weapon"]["target_aspect_angle_max"]) <= 3.141592653589793:
        raise ValueError("target_aspect_angle_max must be in (0, pi]")
    return config


def environment_dimensions(config: dict[str, Any]) -> tuple[int, int, int]:
    """Return obs/action/agents from the validated instance protocol."""
    from .observation import observation_dim_for_team_size
    validate_config(config)
    agents = int(config["scenario"]["team_size"])
    return observation_dim_for_team_size(agents), 3, agents


def aircraft_spec(config: dict[str, Any]) -> AircraftSpec:
    return AircraftSpec(**config["aircraft"])


__all__ = ["ENVIRONMENT_VERSION", "SUPPORTED_ENVIRONMENT_VERSIONS", "environment_dimensions",
           "aircraft_spec", "load_config", "validate_config"]
