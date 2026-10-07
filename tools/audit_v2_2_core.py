"""Read-only V2.2 geometry and MAPPO probability diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from algorithm.mappo.networks import SharedMAPPOActor
from env.geometry import engagement_geometry
from env.models import AircraftState
from env.weapon import WeaponEnvelope


ROOT = PROJECT_ROOT


def state(*, x=0.0, altitude=3000.0, theta=0.0) -> AircraftState:
    return AircraftState(x=x, y=0.0, z=-altitude, v=225.0,
                         theta=theta, psi=0.0)


def geometry_audit(samples: int) -> dict:
    config = yaml.safe_load((ROOT / "configs/combat_environment.yaml").read_text())
    weapon = WeaponEnvelope(**config["weapon"])
    distance = 3000.0
    cases = {
        "A_horizontal_true_pointing": (0.0, 0.0),
        "B_climb_30_true_pointing": (30.0, 30.0),
        "C_dive_30_true_pointing": (-30.0, -30.0),
        "D_climb_60_horizontal_los": (60.0, 0.0),
        "E_dive_60_horizontal_los": (-60.0, 0.0),
    }
    rows = {}
    for index, (name, (pitch_deg, los_deg)) in enumerate(cases.items()):
        pitch = np.deg2rad(pitch_deg)
        elevation = np.deg2rad(los_deg)
        horizontal = distance * np.cos(elevation)
        target_altitude = 3000.0 + distance * np.sin(elevation)
        geometry = engagement_geometry(
            state(theta=pitch), state(x=horizontal, altitude=target_altitude)
        )
        threshold = weapon.hit_threshold(geometry.distance)
        # Common random numbers make the rotation-equivalence comparison exact.
        rng = np.random.default_rng(20230817)
        noise = rng.normal(size=(samples, 2))
        legacy_hits = (
            np.abs(geometry.ata + weapon.attack_noise_scale * noise[:, 0]) <= threshold
        ) & (
            np.abs(geometry.ha + weapon.height_noise_scale * noise[:, 1]) <= threshold
        )
        active_hits = (
            np.abs(geometry.boresight_azimuth_error
                   + weapon.attack_noise_scale * noise[:, 0]) <= threshold
        ) & (
            np.abs(geometry.boresight_elevation_error
                   + weapon.height_noise_scale * noise[:, 1]) <= threshold
        )
        rows[name] = {
            "pitch_deg": pitch_deg,
            "los_elevation_deg": los_deg,
            "distance": geometry.distance,
            "fire_window": weapon.in_fire_window(geometry),
            "off_boresight_deg": float(np.rad2deg(geometry.off_boresight)),
            "ata_deg": float(np.rad2deg(geometry.ata)),
            "ha_deg": float(np.rad2deg(geometry.ha)),
            "v2_2_legacy_hit_probability": float(legacy_hits.mean()),
            "active_hit_probability": float(active_hits.mean()),
            "samples": samples,
        }
    return rows


def percentile_summary(values: torch.Tensor) -> dict:
    values = values.detach().float().cpu().flatten()
    q = torch.quantile(values, torch.tensor([0.5, 0.95, 0.99]))
    return {
        "mean": float(values.mean()), "median": float(q[0]),
        "p95": float(q[1]), "p99": float(q[2]), "max": float(values.max()),
    }


def mappo_audit(checkpoint: Path, samples: int) -> dict:
    config = yaml.safe_load((ROOT / "configs/mappo.yaml").read_text())
    impl, network = config["implementation"], config["network"]
    actor = SharedMAPPOActor(
        observation_dim=int(network["observation_dim"]),
        action_dim=int(network["action_dim"]),
        hidden_dim=int(network["actor_hidden_layers"][0]),
        log_std_min=float(impl["log_std_min"]),
        log_std_max=float(impl["log_std_max"]),
        activation=str(impl["actor_activation"]),
    )
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(state_dict["actor"])
    actor.eval()
    generator = torch.Generator().manual_seed(20230817)
    observations = torch.randn(samples, 52, generator=generator)
    with torch.no_grad():
        distribution = actor.distribution(observations)
        raw = distribution.rsample()
        actions = torch.tanh(raw)
        sampled = actor._squashed_log_prob(distribution, raw, actions)
        reconstructed_raw = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
        reconstructed = actor._squashed_log_prob(
            distribution, reconstructed_raw, actions.clamp(-1 + 1e-6, 1 - 1e-6)
        )
        exact = actor._squashed_log_prob(distribution, raw, actions)
        error = (sampled - reconstructed).abs()
        exact_error = (sampled - exact).abs()
        max_abs = actions.abs().amax(dim=-1)
        groups = {
            "lt_0.9": max_abs < .9,
            "0.9_to_0.99": (max_abs >= .9) & (max_abs < .99),
            "0.99_to_0.999": (max_abs >= .99) & (max_abs < .999),
            "ge_0.999": max_abs >= .999,
        }
        grouped = {}
        for name, mask in groups.items():
            grouped[name] = {
                "count": int(mask.sum()),
                "error": percentile_summary(error[mask]) if mask.any() else None,
            }
        log_std = distribution.scale.log()
    return {
        "checkpoint": str(checkpoint), "samples": samples,
        "raw_action_recompute_error": percentile_summary(exact_error),
        "atanh_reconstruction_error": percentile_summary(error),
        "by_saturation_group": grouped,
        "log_std_mean_per_dimension": log_std.mean(0).tolist(),
        "std_mean_per_dimension": distribution.scale.mean(0).tolist(),
        "fraction_abs_action_gt_0.9_per_dimension": (actions.abs() > .9).float().mean(0).tolist(),
        "fraction_abs_action_gt_0.99_per_dimension": (actions.abs() > .99).float().mean(0).tolist(),
        "fraction_abs_action_gt_0.999_per_dimension": (actions.abs() > .999).float().mean(0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mappo-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100_000)
    args = parser.parse_args()
    checkpoint = (
        args.mappo_checkpoint if args.mappo_checkpoint.is_absolute()
        else PROJECT_ROOT / args.mappo_checkpoint
    )
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    report = {
        "environment_geometry": geometry_audit(args.samples),
        "mappo_probability": mappo_audit(checkpoint, args.samples),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
