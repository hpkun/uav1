"""Subprocess entry point for one persistent combat environment."""
from __future__ import annotations

import os
import traceback
from typing import Any

from .factory import make_combat_environment


def combat_environment_worker(connection: Any, config: Any) -> None:
    """Own one environment and serve reset/step commands over a pipe."""
    try:
        environment = make_combat_environment(config)
        connection.send(("ready", {
            "pid": os.getpid(),
            "environment_class": environment.__class__.__name__,
            "environment_variant": getattr(
                environment, "environment_variant", "direct_v2_3"
            ),
            "fixed_policy_class": environment.fixed_policy.__class__.__name__,
            "observation_dim": int(environment.observation_dim),
            "action_dim": int(environment.action_dim),
            "team_size": int(environment.team_size),
        }))
        while True:
            command, payload = connection.recv()
            if command == "reset":
                observation, _ = environment.reset(int(payload))
                connection.send(("ok", (observation, environment.red_alive_mask)))
            elif command == "step":
                connection.send(("ok", environment.step(payload)))
            elif command == "export_curriculum_state":
                exporter = getattr(environment, "export_curriculum_state", None)
                if exporter is None:
                    raise RuntimeError("environment does not support curriculum snapshots")
                connection.send(("ok", exporter()))
            elif command == "restore_curriculum_state":
                restorer = getattr(environment, "restore_curriculum_state", None)
                if restorer is None:
                    raise RuntimeError("environment does not support curriculum snapshots")
                connection.send(("ok", restorer(payload)))
            elif command == "close":
                connection.send(("ok", None))
                break
            else:
                raise RuntimeError(f"unknown environment-worker command: {command}")
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


__all__ = ["combat_environment_worker"]
