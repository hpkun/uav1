#!/usr/bin/env python3
"""Read-only deterministic/stochastic deployment evaluation for MARC checkpoints."""
from __future__ import annotations
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.common.evaluator import episode_return_metrics
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.trainer import MODULAR_MAPPO_IMPL_VERSION, MAPPO_IMPL_VERSION
from algorithm.modules.wave_survival_pbrs import mission_context_numpy
from algorithm.train_modular_mappo import load_config
from env.factory import make_combat_environment

def episode_policy_seed(base,environment_seed,repeat):
    payload=f"{int(base)}:{int(environment_seed)}:{int(repeat)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],"little")%(2**31-1)


def validate_method_version(state):
    method = state.get("extra", {}).get("development_method")
    version = state.get("development_feature_versions", {}).get("milestone_aware_retention_credit")
    valid = {"marc_mappo_v1": 1, "marc_mappo_v2": 2}
    if state.get("algorithm") != "modular_mappo" or method not in valid or version != valid[method]:
        raise RuntimeError(
            f"Invalid MARC checkpoint: development_method={method!r}, checkpoint MARC version={version!r}; "
            "expected valid pair: (marc_mappo_v1, 1) or (marc_mappo_v2, 2), algorithm=modular_mappo"
        )
    return method, version


def environment_identity(config):
    return {"environment_variant": config.get("environment_variant"),
            "red_agents": config.get("scenario", {}).get("team_size"),
            "total_waves": config.get("persistent_waves", {}).get("total_waves"),
            "blue_units_per_wave": config.get("persistent_waves", {}).get("blue_units_per_wave"),
            "max_steps": config.get("simulation", {}).get("max_steps")}


def validate_environment(state, config, env_config):
    expected = {"environment_variant": "persistent_wave_v2", "red_agents": 4,
                "total_waves": 3, "blue_units_per_wave": [4, 3, 3], "max_steps": 3000}
    extra = state.get("extra", {})
    training_env = extra.get("runtime_environment_config") or extra.get("environment_config")
    if not isinstance(training_env, dict):
        raise RuntimeError("checkpoint lacks training environment identity")
    for label, value in (("diagnostic", env_config), ("checkpoint training", training_env)):
        identity = environment_identity(value)
        if identity != expected:
            raise RuntimeError(f"433 environment identity mismatch ({label}): {identity}; expected {expected}")
    for key, expected_value in (("environment_variant", "persistent_wave_v2"),
                                ("observation_dim", 52), ("action_dim", 3), ("num_agents", 4)):
        if extra.get(key) != expected_value:
            raise RuntimeError(f"checkpoint {key} mismatch: {extra.get(key)!r}; expected {expected_value}")
    for key, expected_value in (("observation_dim", 52), ("action_dim", 3), ("num_agents", 4)):
        if config["network"].get(key) != expected_value:
            raise RuntimeError(f"algorithm config {key} mismatch; expected {expected_value}")
    # This diagnostic only supports the unchanged, feed-forward MARC deployment topology.
    allowed = {"actor_lr_decay", "milestone_aware_retention_credit"}
    if "milestone_aware_retention_credit" not in state.get("enabled_modules", []) or set(state["enabled_modules"]) - allowed:
        raise RuntimeError("unsupported deployment modules: expected feed-forward MARC without test-time corrections")
    return {**expected, "observation_dim": 52, "action_dim": 3}


def load_deployment_checkpoint(checkpoint, env_config):
    """Strict actor-only restoration: no optimizer/RNG/elite-bank restoration."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory; no CPU fallback")
    checkpoint = Path(checkpoint).resolve()
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    method, version = validate_method_version(state)
    if state.get("modular_mappo_impl_version") != MODULAR_MAPPO_IMPL_VERSION or state.get("baseline_mappo_impl_version") != MAPPO_IMPL_VERSION:
        raise RuntimeError("checkpoint MAPPO implementation version mismatch")
    config = load_config(checkpoint.parent / "algorithm_config.yaml")
    identity = validate_environment(state, config, env_config)
    marc = config.get("modules", {}).get("milestone_aware_retention_credit", {})
    if config.get("development_method") != method or not marc.get("enabled") or marc.get("version", 1) != version:
        raise RuntimeError(f"run-local config does not match checkpoint method/version: {method}/{version}")
    # Construction seeds Torch. Preserve CPU and every CUDA RNG, including on multi-GPU hosts.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        trainer = build_modular_mappo_trainer(config, "cuda", total_sampled_steps=config["training"]["total_sampled_steps"])
    if trainer.module_protocol()["module_config_sha256"] != state.get("module_config_sha256"):
        raise RuntimeError("checkpoint module protocol mismatch")
    trainer.actor.load_state_dict(state["actor"], strict=True)
    trainer.actor.eval()
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest.hexdigest(),
                "checkpoint_sampled_steps": int(state["sampled_steps"]),
                "training_seed": int(state["extra"]["training_seed"]),
                "development_method": method, "marc_version": version,
                "environment_variant": identity["environment_variant"], "environment_identity": identity,
                "optimizer_restored": False, "marc_module_state_restored": False}
    return trainer, metadata


def evaluation_seed_bank(seed_base, episodes):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    end = seed_base + episodes - 1
    if seed_base <= 45_000_199 and end >= 45_000_000:
        raise ValueError("reserved 45000000..45000199 final-test bank must not be used")
    return list(range(seed_base, end + 1))

def one_episode(trainer,env_config,seed,deterministic,policy_seed):
    env=make_combat_environment(env_config);obs,_=env.reset(seed);alive=env.red_alive_mask.copy();ah,ch=trainer.initial_hidden(1);ep=np.zeros(1,np.float32);ret=np.zeros(4);wave=1
    if np.asarray(obs).shape != (4, 52):raise RuntimeError(f"observation shape mismatch: {np.asarray(obs).shape}")
    devices=list(range(torch.cuda.device_count())) if trainer.device.type=="cuda" else []
    with torch.random.fork_rng(devices=devices):
        if not deterministic:
            torch.manual_seed(policy_seed)
            if trainer.device.type=="cuda":torch.cuda.manual_seed_all(policy_seed)
        while True:
            ctx=mission_context_numpy(trainer,np.asarray([wave]),np.asarray([3]),env.blue_alive_mask[None],np.asarray([env.steps]),env.max_steps)
            actions,ah=trainer.act(obs[None],alive[None],deterministic,False,ctx,ah,ep,wave_indices=np.asarray([wave]))
            if actions.shape != (1,4,3) or not np.isfinite(actions).all():raise RuntimeError("invalid deployment action shape/values")
            obs,reward,terminated,truncated,info=env.step(actions[0]);ret+=reward;alive=np.asarray(info["red_alive_mask"],np.float32)
            ah=trainer.recurrent.apply_alive(ah,alive[None]);ch=trainer.recurrent.apply_alive(ch,alive[None]);ep[:]=1;wave=int(info.get("wave_index",1))
            if terminated or truncated:
                team,_=episode_return_metrics(ret)
                return {"seed":seed,"W1":float(info["waves_cleared"]>=1),"W2":float(info["waves_cleared"]>=2),"W3":float(info["waves_cleared"]>=3),"AW":float(info["waves_cleared"]),"Return":team,
                    "Boundary":float(info["red_boundary_exits"]),"Ground":float(info["red_ground_losses"]),"EpisodeLength":float(info["episode_length"])}

def aggregate(rows):
    result={key:float(np.mean([r[key] for r in rows])) for key in ("W1","W2","W3","AW","Return","Boundary","Ground","EpisodeLength")}
    result["Q2"]=None if result["W1"]==0 else result["W2"]/result["W1"];result["Q3"]=None if result["W2"]==0 else result["W3"]/result["W2"]
    return result


def evaluate_deployment(trainer,metadata,env_config,env_path,episodes,seed_base,policy_mode,stochastic_repeats=3,policy_seed_base=91_000_000):
    if stochastic_repeats < 1 or policy_mode not in ("deterministic", "stochastic"):
        raise ValueError("invalid deployment mode/repeats")
    seeds = evaluation_seed_bank(seed_base, episodes)
    repeats = 1 if policy_mode == "deterministic" else stochastic_repeats
    results = []
    for repeat in range(repeats):
        policy_seeds = [episode_policy_seed(policy_seed_base, seed, repeat) for seed in seeds]
        rows = [one_episode(trainer, env_config, seed, policy_mode == "deterministic", policy_seed)
                for seed, policy_seed in zip(seeds, policy_seeds)]
        results.append({"repeat": repeat, "metrics": aggregate(rows),
                        "environment_seed_range": [seeds[0], seeds[-1]], "episode_policy_seeds": policy_seeds})
    metrics = {key: {"mean": float(np.mean([r["metrics"][key] for r in results])),
                     "std": float(np.std([r["metrics"][key] for r in results]))}
               for key in results[0]["metrics"] if all(r["metrics"][key] is not None for r in results)}
    report = {**metadata, "environment_config": str(Path(env_path).resolve()), "policy_mode": policy_mode,
              "environment_seed_range": [seeds[0], seeds[-1]], "episodes_per_repeat": episodes,
              "stochastic_repeats": repeats, "per_repeat": results, "aggregate": metrics,
              "policy_rng_provenance": {"base_seed": policy_seed_base,
                  "derivation": "SHA256(base:environment_seed:repeat), first 8 bytes little-endian modulo 2**31-1",
                  "scope": "independent per environment seed and repeat; fork_rng preserves CPU/all CUDA RNG",
                  "sampling_used": policy_mode == "stochastic"}, "training_performed": False}
    # None conditional ratios remain null; nonfinite metrics must fail rather than produce invalid JSON.
    json.dumps(report, allow_nan=False)
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--checkpoint",type=Path,required=True);parser.add_argument("--env-config",type=Path,required=True);parser.add_argument("--episodes",type=int,required=True);parser.add_argument("--seed-base",type=int,required=True)
    parser.add_argument("--policy-mode",choices=("deterministic","stochastic"),required=True);parser.add_argument("--stochastic-repeats",type=int,default=3);parser.add_argument("--policy-seed-base",type=int,default=91_000_000);parser.add_argument("--output",type=Path)
    args=parser.parse_args();checkpoint=args.checkpoint if args.checkpoint.is_absolute() else ROOT/args.checkpoint;env_path=args.env_config if args.env_config.is_absolute() else ROOT/args.env_config
    if not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory; no CPU fallback")
    if args.episodes<1 or args.stochastic_repeats<1:raise ValueError("episodes/repeats must be positive")
    evaluation_seed_bank(args.seed_base,args.episodes)
    output=None if args.output is None else (args.output if args.output.is_absolute() else ROOT/args.output)
    if output is not None and output.exists():raise FileExistsError(output)
    env_config=load_config(env_path);trainer,metadata=load_deployment_checkpoint(checkpoint,env_config)
    report=evaluate_deployment(trainer,metadata,env_config,env_path,args.episodes,args.seed_base,args.policy_mode,args.stochastic_repeats,args.policy_seed_base)
    if args.output:
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
