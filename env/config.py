"""Configuration loading for the paper-constrained combat environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

from .models import AircraftSpec


ENVIRONMENT_VERSION = "2.3"
SUPPORTED_ENVIRONMENT_VERSIONS = frozenset({"2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "3.0", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14", "3.15", "3.16", "3.17", "3.18", "3.19", "3.20", "3.21", "3.22", "3.23", "3.24", "3.25", "3.26", "3.27", "3.28", "3.29", "3.30", "3.31", "3.32", "3.33", "3.34", "3.35", "3.36"})


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the active V2.3 environment configuration."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return validate_config(config)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Reject unsupported schemas before constructing a combat environment."""
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.36':
        from .v336_config import validate_v336
        return validate_v336(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.35':
        from .v335_config import validate_v335
        return validate_v335(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.34':
        from .v334_config import validate_v334
        return validate_v334(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.33':
        from .v333_config import validate_v333
        return validate_v333(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.32':
        from .v332_config import validate_v332
        return validate_v332(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.31':
        from .v331_config import validate_v331
        return validate_v331(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.30':
        from .v330_config import validate_v330
        return validate_v330(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.29':
        from .v329_config import validate_v329
        return validate_v329(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.28':
        from .v328_config import validate_v328
        return validate_v328(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.27':
        from .v327_config import validate_v327
        return validate_v327(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.26':
        from .v326_config import validate_v326
        return validate_v326(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.25':
        from .v325_config import validate_v325
        return validate_v325(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.24':
        from .v324_config import validate_v324
        return validate_v324(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.23':
        from .v323_config import validate_v323
        return validate_v323(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.22':
        from .v322_config import validate_v322
        return validate_v322(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.21':
        from .v321_config import validate_v321
        return validate_v321(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.20':
        from .v320_config import validate_v320
        return validate_v320(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.19':
        from .v319_config import validate_v319
        return validate_v319(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.17':
        from .v317_config import validate_v317
        return validate_v317(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.18':
        from .v318_config import validate_v318
        return validate_v318(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.16':
        from .v316_config import validate_v316
        return validate_v316(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.15':
        from .v315_config import validate_v315
        return validate_v315(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.14':
        from .v314_config import validate_v314
        return validate_v314(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.13':
        from .v313_config import validate_v313
        return validate_v313(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.12':
        from .v312_config import validate_v312
        return validate_v312(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.11':
        from .v311_config import validate_v311
        return validate_v311(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.10':
        from .v310_config import validate_v310
        return validate_v310(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.9':
        from .v39_config import validate_v39
        return validate_v39(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.8':
        from .v38_config import validate_v38
        return validate_v38(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.7':
        from .v37_config import validate_v37
        return validate_v37(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.6':
        from .v36_config import validate_v36
        return validate_v36(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.5':
        from .v35_config import validate_v35
        return validate_v35(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.4':
        from .v34_config import validate_v34
        return validate_v34(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.3':
        from .v33_config import validate_v33
        return validate_v33(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.2':
        from .v32_config import validate_v32
        return validate_v32(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.1':
        from .v31_config import validate_v31
        return validate_v31(config)
    if isinstance(config, dict) and str(config.get('environment_version')) == '3.0':
        from .v30_config import validate_v30
        return validate_v30(config)
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
            f"environment_version must be 2.3, 2.4, 2.5, 2.6, 2.7, 2.8 or 2.9, got "
            f"{config['environment_version']}"
        )
    expected_team_size = {"2.3": 4, "2.4": 8, "2.5": 5, "2.6": 8, "2.7": 5, "2.8": 5, "2.9": 5}[version]
    if config["scenario"].get("team_size") != expected_team_size:
        raise ValueError(f"environment_version {version} requires team_size={expected_team_size}")
    if len(config["scenario"].get("formation_offsets", [])) != expected_team_size:
        raise ValueError("formation_offsets length must equal version-specific team_size")
    weapon_fields = {"range_min", "range_max", "off_boresight_angle_max",
                     "effective_hit_distance", "attack_noise_scale", "height_noise_scale"}
    if version in {"2.4", "2.5", "2.6", "2.7", "2.8", "2.9"}:
        weapon_fields.add("target_aspect_angle_max")
    if set(config["weapon"]) != weapon_fields:
        raise ValueError(f"weapon schema mismatch for environment_version {version}")
    if version in {"2.4", "2.5", "2.6", "2.7", "2.8", "2.9"} and not 0 < float(config["weapon"]["target_aspect_angle_max"]) <= 3.141592653589793:
        raise ValueError("target_aspect_angle_max must be in (0, pi]")
    if version in {"2.6", "2.7", "2.8", "2.9"}:
        import math
        fields = {"desired_speed", "turn_angle", "turn_range", "required_lateral_displacement", "conversion_range"}
        blue = config["blue_policy"]
        if not isinstance(blue, dict) or set(blue) != fields:
            raise ValueError(f"v{version} blue_policy schema mismatch")
        values = {k: float(blue[k]) for k in fields}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f"v{version} blue_policy values must be finite")
        if not (150 <= values["desired_speed"] <= 300
                and 0 < values["turn_angle"] < math.pi/2
                and values["turn_range"] > 0
                and values["required_lateral_displacement"] > 0
                and values["required_lateral_displacement"] < values["conversion_range"] <= float(config["weapon"]["range_max"])):
            raise ValueError(f"invalid v{version} blue_policy geometry/speed parameters")
    return config


def environment_dimensions(config: dict[str, Any]) -> tuple[int, int, int]:
    """Return obs/action/agents from the validated instance protocol."""
    from .observation import observation_dim_for_team_size
    validate_config(config)
    if config['environment_version'] in ('3.30','3.31','3.32','3.33','3.34','3.35','3.36'):
        return 45, 3, 4
    agents = int(config["scenario"]["team_size"])
    if str(config['environment_version']) in ('3.12', '3.13', '3.14', '3.15', '3.16', '3.17', '3.18', '3.19', '3.20', '3.21', '3.22', '3.23', '3.24', '3.25', '3.26', '3.27', '3.28', '3.29'):
        return 56, 3, agents
    if str(config['environment_version']) in ('3.0', '3.1', '3.2', '3.3', '3.4', '3.5', '3.6', '3.7', '3.8','3.9','3.10','3.11'):
        from .v30_observation import observation_dim_for_v30
        return observation_dim_for_v30(agents), 3, agents
    return observation_dim_for_team_size(agents), 3, agents


def aircraft_spec(config: dict[str, Any]) -> AircraftSpec:
    return AircraftSpec(**config["aircraft"])


__all__ = ["ENVIRONMENT_VERSION", "SUPPORTED_ENVIRONMENT_VERSIONS", "environment_dimensions",
           "aircraft_spec", "load_config", "validate_config"]
