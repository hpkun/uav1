"""Small common interface for modular MAPPO capability modules."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


class CapabilityModule:
    name = "module"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = deepcopy(config or {})
        self.enabled = bool(self.config.get("enabled", False))

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "enabled": self.enabled, "config": deepcopy(self.config)}


def enabled_module_names(config: dict[str, Any]) -> list[str]:
    return sorted(name for name, value in config.items() if bool(value.get("enabled", False)))


__all__ = ["CapabilityModule", "enabled_module_names"]
