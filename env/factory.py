"""Construct the project's 4v4 combat environment."""
from pathlib import Path
from typing import Any
from .combat_env import DEFAULT_COMBAT_CONFIG, MultiUAVCombatEnv


def make_combat_environment(config: str | Path | dict[str, Any] = DEFAULT_COMBAT_CONFIG) -> MultiUAVCombatEnv:
    return MultiUAVCombatEnv(config)


__all__ = ["make_combat_environment"]
