"""Configuration-driven combat-environment construction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config
from .combat_env import MultiUAVCombatEnv
from .persistent_env import PERSISTENT_WAVE_VARIANTS, PersistentWaveCombatEnv


def make_combat_environment(config: str | Path | dict[str, Any]) -> MultiUAVCombatEnv:
    loaded = load_config(config) if not isinstance(config, dict) else config
    variant = loaded.get("environment_variant", "direct_v2_3")
    if variant in PERSISTENT_WAVE_VARIANTS:
        return PersistentWaveCombatEnv(loaded)
    if variant == "direct_v2_3":
        return MultiUAVCombatEnv(loaded)
    raise ValueError(f"unknown environment_variant: {variant!r}")


__all__ = ["make_combat_environment"]
