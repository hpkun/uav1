"""Read-only, paired-seed mean-policy versus sampled-policy diagnostics."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from algorithm.common.checkpoint import validate_checkpoint_for_evaluation
from algorithm.common.critic_protocol import checkpoint_widths
from algorithm.common.evaluator import aggregate_combat_records, episode_return_metrics
from algorithm.common.protocol import config_sha256
from algorithm.mappo.factory import build_mappo_trainer
from algorithm.stea_mappo.factory import build_stea_mappo_trainer
from algorithm.stea_mappo.protocol import require_cuda, validate_checkpoint
from env.config import environment_dimensions, load_config
from env.factory import make_combat_environment

ACTION_DIMENSIONS = ("heading", "pitch", "speed")
GAP_FIELDS = ("win_rate", "average_return", "average_red_loss", "average_blue_loss",
              "average_episode_length", "timeout_rate")


@contextmanager
def policy_rng(device, seed):
    """A fresh episode sampling stream, independent of environment RNG/state."""
    device = torch.device(device)
    indices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=indices):
        torch.random.default_generator.manual_seed(seed)
        torch.cuda.default_generators[indices[0]].manual_seed(seed)
        yield


class PolicyExecution:
    """Observe the actual executed distribution in either Gaussian protocol."""
    def __init__(self, actor, recurrent, device):
        self.actor, self.recurrent, self.device = actor, recurrent, torch.device(device)
        self.hidden = None
        self.start = True
        self.clamped_log_std = None

    def reset(self, agents):
        self.start = True
        if self.recurrent:
            self.hidden = torch.zeros(agents, self.actor.gru_hidden_dim, device=self.device)

    @torch.no_grad()
    def decide(self, observations, alive, mode):
        if mode not in ("deterministic", "stochastic"):
            raise ValueError("unknown execution mode")
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        alive = torch.as_tensor(alive, dtype=torch.float32, device=self.device)
        if self.recurrent:
            distribution, self.hidden, _ = self.actor.distribution_step(
                observations, self.hidden, alive,
                torch.tensor(float(self.start), device=self.device))
        else:
            distribution = self.actor.distribution(observations)
        raw = distribution.mean if mode == "deterministic" else distribution.rsample()
        actions = raw.tanh() * alive[:, None]
        self.clamped_log_std = distribution.scale.log()
        if not torch.allclose(distribution.scale, self.clamped_log_std.exp(), rtol=1e-6, atol=0.):
            raise RuntimeError("captured sigma does not match actual actor distribution")
        return actions, raw, distribution, self.clamped_log_std

    def after_transition(self, alive, done):
        self.start = False
        if self.recurrent:
            self.hidden *= torch.as_tensor(alive, device=self.device)[:, None]
            if done:
                self.hidden.zero_()

    def close(self):
        pass


class VarianceStatistics:
    """Equal weight per live Red-agent decision, separately for each mode."""
    def __init__(self):
        self.rows = []

    def add(self, distribution, clamped_log_std, raw, alive):
        live = torch.as_tensor(alive, device=raw.device) > .5
        values = (clamped_log_std, distribution.scale, distribution.mean.abs(),
                  (raw - distribution.mean).abs())
        if live.any():
            self.rows.append(np.stack([x[live].detach().cpu().numpy() for x in values], axis=1))

    def summarize(self):
        if not self.rows:
            return {name: {"live_decisions": 0} for name in ACTION_DIMENSIONS}
        rows = np.concatenate(self.rows).astype(np.float64)
        result = {}
        for dim, name in enumerate(ACTION_DIMENSIONS):
            log_std, sigma, abs_mu, deviation = rows[:, :, dim].T
            result[name] = dict(
                live_decisions=len(log_std), mean_log_std=float(log_std.mean()),
                median_log_std=float(np.median(log_std)), std_log_std=float(log_std.std(ddof=0)),
                mean_sigma=float(sigma.mean()), median_sigma=float(np.median(sigma)),
                p10_sigma=float(np.quantile(sigma, .1)), p90_sigma=float(np.quantile(sigma, .9)),
                mean_absolute_mu=float(abs_mu.mean()),
                mean_absolute_sampled_raw_action_minus_mu=float(deviation.mean()))
        return result


def load_policy(algorithm, checkpoint, env_config, algorithm_config, device):
    require_cuda(device)
    if str(env_config["environment_version"]) != "2.5" or environment_dimensions(env_config) != (65, 3, 5):
        raise ValueError("this diagnostic requires v2.5 / 65D / 3 actions / 5 agents")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if algorithm == "mappo":
        validate_checkpoint_for_evaluation(state, env_config, algorithm_config)
        extra = state.get("extra", {})
        for field in ("environment_version", "observation_dim", "action_dim", "num_agents",
                      "environment_config_sha256", "algorithm_config_sha256", "training_seed",
                      "training_total_sampled_steps", "training_num_envs", "effective_hidden_dim",
                      "training_gamma", "training_smoke"):
            if field not in extra:
                raise RuntimeError(f"incomplete checkpoint contract: missing {field}")
        actor_width, critic_width = checkpoint_widths(state)
        trainer = build_mappo_trainer(algorithm_config, device, hidden_dim=actor_width, critic_hidden_dim=critic_width)
    elif algorithm == "stea-mappo":
        extra = validate_checkpoint(state, env_config, algorithm_config)
        trainer = build_stea_mappo_trainer(algorithm_config, device,
            seed=extra["training_seed"], smoke=extra["training_smoke"])
    else:
        raise ValueError("algorithm must be mappo or stea-mappo")
    for field, expected in (("environment_config_sha256", config_sha256(env_config)),
                            ("algorithm_config_sha256", config_sha256(algorithm_config))):
        if extra[field] != expected:
            raise RuntimeError(f"checkpoint {field} mismatch")
    trainer.load(checkpoint)
    trainer.actor.eval()
    return trainer, state


def evaluate_mode(actor, recurrent, env_config, seeds, device, mode, policy_seed):
    statistics, records = VarianceStatistics(), []
    execution = PolicyExecution(actor, recurrent, device)
    try:
        for index, seed in enumerate(seeds):
            env = make_combat_environment(env_config)
            observations, _ = env.reset(seed)
            execution.reset(env.team_size)
            returns = np.zeros(env.team_size)
            digest = hashlib.sha256()
            with policy_rng(device, policy_seed + index):
                while True:
                    alive = env.red_alive_mask
                    actions, raw, distribution, log_std = execution.decide(observations, alive, mode)
                    statistics.add(distribution, log_std, raw, alive)
                    actions = actions.cpu().numpy()
                    digest.update(actions.tobytes())
                    observations, reward, terminated, truncated, info = env.step(actions)
                    returns += reward
                    execution.after_transition(env.red_alive_mask, terminated or truncated)
                    if terminated or truncated:
                        team, agent = episode_return_metrics(returns)
                        scalars = {key: value.item() if isinstance(value, np.generic) else value
                                   for key, value in info.items()
                                   if isinstance(value, (str, int, float, bool, type(None), np.generic))}
                        records.append(dict(episode_return=team, mean_agent_episode_return=agent,
                            **scalars, environment_seed=seed,
                            policy_sampling_seed=policy_seed + index if mode == "stochastic" else None,
                            executed_action_sha256=digest.hexdigest()))
                        break
    finally:
        execution.close()
    metrics = aggregate_combat_records(records)
    metrics.update({name.upper(): metrics[f"average_episode_{name}_total"] for name in ("r1", "r2", "r3", "r4")})
    return dict(metrics=metrics, policy_statistics=statistics.summarize(), episodes=records)


def evaluate_policy_modes(algorithm, checkpoint, env_config, algorithm_config, seed_base,
                          episodes, device="cuda", stochastic_policy_seed=12345):
    if episodes <= 0 or seed_base < 0 or stochastic_policy_seed < 0:
        raise ValueError("episodes must be positive and seeds nonnegative")
    trainer, state = load_policy(algorithm, checkpoint, env_config, algorithm_config, device)
    extra = state["extra"]
    seeds = list(range(seed_base, seed_base + episodes))
    training_end = int(extra["training_seed"]) + int(extra["training_total_sampled_steps"]) + int(extra["training_num_envs"])
    if any(int(extra["training_seed"]) <= seed <= training_end for seed in seeds):
        raise ValueError("environment seeds overlap the conservative training seed range")
    with torch.backends.cudnn.flags(benchmark=False, deterministic=True):
        modes = {mode: evaluate_mode(trainer.actor, algorithm == "stea-mappo", env_config,
                 seeds, device, mode, stochastic_policy_seed)
                 for mode in ("deterministic", "stochastic")}
    return dict(algorithm=state["algorithm"], checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sampled_steps=int(state["sampled_steps"]), checkpoint_metadata=extra,
        implementation_version=state.get("mappo_impl_version", state.get("stea_mappo_impl_version")),
        environment_config_sha256=config_sha256(env_config), algorithm_config_sha256=config_sha256(algorithm_config),
        environment_seeds=seeds, stochastic_policy_seed=stochastic_policy_seed, device=str(device),
        statistics_weighting="one sample per live Red-agent decision; modes have their own visited states",
        log_std_definition="actual actor head clamped to its configured bounds; sigma=exp(clamped_log_std)",
        mode_gap={key: modes["stochastic"]["metrics"][key] - modes["deterministic"]["metrics"][key]
                  for key in GAP_FIELDS}, **modes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("mappo", "stea-mappo"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--algorithm-config", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stochastic-policy-seed", type=int, default=12345)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    protected = {p.resolve() for p in (args.checkpoint, args.env_config, args.algorithm_config)}
    if output in protected or output.suffix.lower() != ".json":
        raise ValueError("output must be a separate .json file")
    result = evaluate_policy_modes(args.algorithm, args.checkpoint, load_config(args.env_config),
        yaml.safe_load(args.algorithm_config.read_text(encoding="utf-8")), args.seed_base,
        args.episodes, args.device, args.stochastic_policy_seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise FileExistsError("output already contains different results; choose a new output path")
    else:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), mode_gap=result["mode_gap"]), indent=2))


if __name__ == "__main__":
    main()
