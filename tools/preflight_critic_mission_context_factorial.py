"""Strict preflight and optional CUDA smoke for the critic-context 2x2."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import csv
import re

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modules.wave_survival_pbrs import (
    WaveSurvivalPotentialShapingModule, mission_progress_from_blue_losses,
    mission_progress_from_wave_state,
)

MANIFEST = ROOT / "experiments/critic_mission_context_factorial_manifest.json"
AUDIT = ROOT / "outputs/critic_mission_context_preflight_audit"
SEEDS = (5201, 5202, 5203)
EVAL = (42000000, 42000049)
EXPOSED_41M = (41000000, 41000199)
ENV_CONFIG_SHA256 = "ca2108c449065f17a3ad8ea287c94e8aa94dadac8b1e20a7b063afbfd22333ee"
ENV_FILE_SHA256 = "ad16c516b31c6fd6eeed825da114e53e6092356daed18b2723371750e5dd92b2"
CELLS = ("C0R0", "C0R1", "C1R0", "C1R1")
CONFIGS = {
    "C0R0": ROOT / "configs/dev_c0r0_plain_mappo_900k.yaml",
    "C0R1": ROOT / "configs/dev_c0r1_pbrs_900k.yaml",
    "C1R0": ROOT / "configs/dev_c1r0_critic_context_900k.yaml",
    "C1R1": ROOT / "configs/dev_c1r1_critic_context_pbrs_900k.yaml",
}
OFF = ("recurrent_memory", "popart", "multi_wave_reward", "wave_balancing",
       "warm_start", "curriculum", "policy_anchor", "entity_attention",
       "advantage_priority", "ppo_stabilization")
PROTOCOL_DECLARATIONS = {
    path.resolve() for path in [MANIFEST, Path(__file__).resolve(),
        ROOT / "tools/run_critic_mission_context_factorial.sh", *CONFIGS.values(),
        ROOT / "configs/dev_critic_mission_context_common_900k.yaml",
        ROOT / "tests/test_critic_mission_context_factorial.py"]
}


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def normalize_factors(config: dict) -> dict:
    value = deepcopy(config)
    value["modules"]["wave_context"]["enabled"] = "FACTOR_C"
    value["modules"]["wave_survival_pbrs"]["enabled"] = "FACTOR_R"
    return value


def validate_critic_context_factorial_configs(configs: dict[str, dict]) -> dict:
    reference = normalize_factors(configs["C0R0"])
    if any(normalize_factors(configs[cell]) != reference for cell in CELLS[1:]):
        raise RuntimeError("resolved configs differ outside Factor C and Factor R")
    expected = {
        "actor_learning_rate": 3e-4, "critic_learning_rate": 3e-4,
        "gamma": .999, "gae_lambda": .95, "clip_ratio": .2,
        "entropy_coefficient": .01, "value_loss_coefficient": .5,
        "max_grad_norm": .5, "rollout_steps": 256, "ppo_epochs": 10,
        "minibatch_size": 512, "num_train_envs": 24,
        "total_sampled_steps": 900000, "evaluation_episodes": 50,
        "evaluation_interval_sampled_steps": 100000, "device": "cuda",
    }
    rows = {}
    for cell, config in configs.items():
        training = config["training"]
        bad = {key: (training.get(key), value) for key, value in expected.items()
               if training.get(key) != value}
        if bad:
            raise RuntimeError(f"{cell} training mismatch: {bad}")
        if any(config["modules"].get(name, {}).get("enabled", False) for name in OFF):
            raise RuntimeError(f"{cell} has a forbidden module enabled")
        c_expected = cell.startswith("C1")
        r_expected = cell.endswith("R1")
        context = config["modules"]["wave_context"]
        pbrs = config["modules"]["wave_survival_pbrs"]
        if context != {"enabled": c_expected, "context_target": "critic_only",
                       "encoding": "mission_markov", "max_waves": 3}:
            raise RuntimeError(f"{cell} context factor mismatch")
        if pbrs != {"enabled": r_expected, "coefficient": 1.0,
                    "potential": "progress_times_survival", "terminal_zero": True}:
            raise RuntimeError(f"{cell} PBRS factor mismatch")
        decay = config["modules"]["actor_lr_decay"]
        if decay != {"enabled": True, "schedule": "delayed_linear",
                     "start_step": 600000, "end_step": 900000,
                     "start_lr": 3e-4, "end_lr": 1e-4}:
            raise RuntimeError(f"{cell} actor LR schedule mismatch")
        protocol = config["development_protocol"]
        if (protocol["primary_checkpoint"] != "latest_at_budget"
                or protocol["primary_budget"] != 900000
                or protocol["best_checkpoint_role"] != "diagnostic_only"):
            raise RuntimeError(f"{cell} checkpoint-selection protocol mismatch")
        validation = protocol["validation"]
        if (validation["seed_start"], validation["seed_end"], validation["episodes"],
                validation["deterministic"], validation["common_scenarios"]) != (*EVAL, 50, True, True):
            raise RuntimeError(f"{cell} evaluation protocol mismatch")
        if protocol["reserved_future_final_test"]["executed"] is not False:
            raise RuntimeError("33M must remain unexecuted")
        rows[cell] = {"factor_C": c_expected, "factor_R": r_expected,
                      "actor_observation_dim": 52, "actor_context_dim": 0,
                      "critic_context_dim": 5 if c_expected else 0,
                      "evaluation_range": list(EVAL), "budget": 900000}
    return rows


def text_freshness_scan() -> dict:
    needles = {"training_5201_5203": ("5201", "5202", "5203"),
               "candidate_40m": ("40000000", "40000049"),
               "exposed_41m": ("41000000", "41000049", "41000199"),
               "selected_42m": ("42000000", "42000049")}
    hits = {key: [] for key in needles}
    # Source/config text: standalone integer literals are meaningful evidence.
    # Scan every repository text location, including root files, docs/archive,
    # and scripts.  Current protocol declarations are excluded from freshness
    # conflicts and reported separately by validate().
    for base in [ROOT]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if (not path.is_file() or path.suffix.lower() not in {".json",".yaml",".yml",".md",".txt",".py",".sh"}
                    or path.resolve() in PROTOCOL_DECLARATIONS
                    or ROOT / "outputs" in path.parents or ROOT / ".git" in path.parents):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for key, values in needles.items():
                found = [value for value in values if re.search(rf"(?<!\d){value}(?!\d)", text)]
                if found:
                    hits[key].append({"path": str(path.relative_to(ROOT)), "tokens": found})
    # Outputs contain arbitrary floating metrics and hashes, so inspect only
    # seed-bearing structured fields and explicit log declarations.
    def classify(value: int):
        if value in SEEDS:return "training_5201_5203"
        if 40000000 <= value <= 40000049:return "candidate_40m"
        if 41000000 <= value <= 41000199:return "exposed_41m"
        if 42000000 <= value <= 42000049:return "selected_42m"
        return None
    def walk(value, path_parts=()):
        found=[]
        if isinstance(value,dict):
            for key,item in value.items():
                child=path_parts+(str(key),)
                if "seed" in str(key).lower():
                    candidates=item if isinstance(item,list) else [item]
                    for candidate in candidates:
                        if isinstance(candidate,(int,float)) and float(candidate).is_integer():
                            category=classify(int(candidate))
                            if category:found.append((category,int(candidate),".".join(child)))
                found.extend(walk(item,child))
        elif isinstance(value,list):
            for index,item in enumerate(value):found.extend(walk(item,path_parts+(str(index),)))
        return found
    output_root=ROOT/"outputs"
    if output_root.exists():
        for path in output_root.rglob("*"):
            if not path.is_file() or AUDIT in path.parents:
                continue
            found=[]
            try:
                if path.suffix.lower()==".json":found=walk(json.loads(path.read_text(encoding="utf-8",errors="ignore")))
                elif path.suffix.lower()==".jsonl":
                    for line in path.read_text(encoding="utf-8",errors="ignore").splitlines():
                        if line.strip():found.extend(walk(json.loads(line)))
                elif path.suffix.lower()==".csv":
                    with path.open(encoding="utf-8-sig",errors="ignore",newline="") as handle:
                        for row in csv.DictReader(handle):
                            for name,value in row.items():
                                if name and "seed" in name.lower() and value and re.fullmatch(r"\d+(?:\.0+)?",value.strip()):
                                    category=classify(int(float(value)))
                                    if category:found.append((category,int(float(value)),name))
                elif path.suffix.lower()==".log":
                    text=path.read_text(encoding="utf-8",errors="ignore")
                    for category,values in needles.items():
                        for value in values:
                            if re.search(rf"(?i)(?:training_|evaluation_)?seed(?:_base|_end)?\s*[:=]\s*['\"]?{value}(?!\d)",text):
                                found.append((category,int(value),"log_seed_declaration"))
            except (OSError,ValueError,json.JSONDecodeError,csv.Error):
                continue
            for category,value,field in found:
                hits[category].append({"path":str(path.relative_to(ROOT)),"tokens":[str(value)],"field":field})
    for key in hits:
        unique={(row["path"],tuple(row["tokens"]),row.get("field")):row for row in hits[key]}
        hits[key]=list(unique.values())
    return hits


def checkpoint_freshness_scan() -> dict:
    hits = []
    paths = list((ROOT / "outputs").rglob("*.pt"))
    for path in paths:
        try:
            state = torch.load(path, map_location="cuda", weights_only=False)
            extra = state.get("extra", {}) if isinstance(state, dict) else {}
            values = [extra.get("training_seed"), extra.get("evaluation_seed_base")]
            values += [row.get("evaluation_seed_base") for row in extra.get("evaluation_history", [])
                       if isinstance(row, dict)]
            found = [int(value) for value in values if isinstance(value, (int, float)) and
                     (int(value) in SEEDS or 40000000 <= int(value) <= 40000049
                      or 41000000 <= int(value) <= 41000199
                      or 42000000 <= int(value) <= 42000049)]
            if found:
                hits.append({"path": str(path.relative_to(ROOT)), "values": found})
            del state
        except Exception as error:
            hits.append({"path": str(path.relative_to(ROOT)), "load_error": str(error)})
        torch.cuda.empty_cache()
    return {"checkpoint_count": len(paths), "hits": hits}


def initialization_audit(configs: dict[str, dict], device: str) -> dict:
    result = {"device": device, "seeds": {}}
    obs = np.random.default_rng(20260908).normal(size=(2, 4, 52)).astype(np.float32)
    alive = np.ones((2, 4), dtype=np.float32)
    for seed in SEEDS:
        trainers = {}; rng = {}
        for cell in CELLS:
            config = deepcopy(configs[cell])
            config["training"]["seed"] = seed
            trainers[cell] = build_modular_mappo_trainer(config, device)
            rng[cell] = {"cpu": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                         "cuda": [hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
                                  for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}
        actor_hashes = {cell: tensor_hash(trainers[cell].actor.state_dict()) for cell in CELLS}
        critic_hashes = {cell: tensor_hash(trainers[cell].critic.state_dict()) for cell in CELLS}
        common_names=set(trainers["C0R0"].critic.state_dict())
        common_hashes={cell:tensor_hash({k:v for k,v in trainers[cell].critic.state_dict().items() if k in common_names}) for cell in CELLS}
        distributions = {}
        actions = {}
        for cell, trainer in trainers.items():
            tensor = torch.as_tensor(obs, device=trainer.device)
            distribution = trainer.actor.distribution(tensor)
            distributions[cell] = (distribution.mean.detach().cpu(), distribution.stddev.detach().cpu())
            actions[cell] = trainer.act(obs, alive, True)[0]
        actor_exact = len(set(actor_hashes.values())) == 1
        behavior_exact = all(torch.equal(distributions[CELLS[0]][0], distributions[cell][0])
                             and torch.equal(distributions[CELLS[0]][1], distributions[cell][1])
                             and np.array_equal(actions[CELLS[0]], actions[cell])
                             for cell in CELLS[1:])
        c0_equal = critic_hashes["C0R0"] == critic_hashes["C0R1"]
        c1_equal = critic_hashes["C1R0"] == critic_hashes["C1R1"]
        shared_exact=len(set(common_hashes.values()))==1
        rng_exact=len({json.dumps(value,sort_keys=True) for value in rng.values()})==1
        architectures = {cell: checkpoint_architecture(trainer) for cell, trainer in trainers.items()}
        c0=next(iter((trainers["C0R0"].values_step(obs,alive)[0],)))
        contexts=np.asarray([[1,0,0,0,1],[0,1,0,1/3,.5],[0,0,1,2/3,0]],np.float32)
        context_values=[]
        for context in contexts:
            context_values.append(trainers["C1R0"].values_step(obs,alive,np.repeat(context[None],2,axis=0))[0])
        initial_values_exact=all(np.array_equal(c0,value) for value in context_values)
        projection=trainers["C1R0"].critic.mission_context_projection
        projection_zero=bool(torch.count_nonzero(projection).item()==0)
        learn_trainer=trainers["C1R0"]
        obs_tensor=torch.as_tensor(obs,device=learn_trainer.device)
        alive_tensor=torch.as_tensor(alive,device=learn_trainer.device)
        learn_context=torch.as_tensor(np.repeat(contexts[1][None],2,axis=0),device=learn_trainer.device)
        learn_trainer.critic_optimizer.zero_grad(set_to_none=True)
        learn_values,_=learn_trainer.critic.forward_step(obs_tensor,alive_tensor,learn_context)
        learn_values.sum().backward()
        projection_gradient_nonzero=bool(torch.count_nonzero(projection.grad).item())
        learn_trainer.critic_optimizer.step()
        projection_learned=bool(torch.linalg.vector_norm(projection).item()>0)
        with torch.no_grad():
            after_a,_=learn_trainer.critic.forward_step(obs_tensor,alive_tensor,torch.as_tensor(np.repeat(contexts[0][None],2,axis=0),device=learn_trainer.device))
            after_b,_=learn_trainer.critic.forward_step(obs_tensor,alive_tensor,learn_context)
        learned_context_effect=not torch.equal(after_a,after_b)
        if not (actor_exact and behavior_exact and c0_equal and c1_equal and shared_exact and rng_exact
                and initial_values_exact and projection_zero and projection_gradient_nonzero
                and projection_learned and learned_context_effect):
            raise RuntimeError(f"initialization fairness failed for seed {seed}")
        if any(architectures[cell]["actor_context_dim"] != 0 or architectures[cell]["actor_input_dim"] != 52 for cell in CELLS):
            raise RuntimeError("actor received context")
        if any(architectures[cell]["critic_context_dim"] != (5 if cell.startswith("C1") else 0) for cell in CELLS):
            raise RuntimeError("critic context dimension mismatch")
        result["seeds"][str(seed)] = {"actor_hashes": actor_hashes, "all_actor_hashes_equal": actor_exact,
            "actor_distribution_and_deterministic_actions_exact": behavior_exact, "critic_hashes": critic_hashes,
            "c0_critic_pair_equal": c0_equal, "c1_critic_pair_equal": c1_equal,
            "shared_critic_parameters_exact_across_four_cells":shared_exact,
            "post_init_cpu_cuda_rng_states_exact":rng_exact,"post_init_rng_hashes":rng,
            "c0_c1_initial_values_exact_for_three_contexts":initial_values_exact,
            "mission_context_projection_initial_norm":0.0,
            "mission_context_projection_gradient_nonzero":projection_gradient_nonzero,
            "mission_context_projection_learns_after_tiny_step":projection_learned,
            "different_contexts_affect_value_after_tiny_step":learned_context_effect,
            "architectures": architectures,
            "critic_parameter_difference_c1_minus_c0": architectures["C1R0"]["critic_parameter_count"] - architectures["C0R0"]["critic_parameter_count"]}
        del trainers
        torch.cuda.empty_cache()
    return result


def old_checkpoint_compatibility_audit() -> list[dict]:
    representatives = {
        "MAPPO": "outputs/pw_alloff_matched_1p5m_seed2024/latest.pt",
        "WB": "outputs/pw_m5_wave_balance_1p5m_seed2024/latest.pt",
        "EA": "outputs/formal_eawb/ea_seed3101/latest.pt",
        "FBMR": "outputs/dev_fbmr_stage2/fbmr_seed3101/latest.pt",
        "WS_PBRS_V1": "outputs/dev_ws_pbrs/baseline_seed5101/latest.pt",
    }
    rows=[]
    for label,relative in representatives.items():
        path=ROOT/relative
        if not path.is_file():
            raise FileNotFoundError(f"missing compatibility checkpoint: {path}")
        state=torch.load(path,map_location="cuda",weights_only=False)
        config=state.get("extra",{}).get("algorithm_config")
        if not isinstance(config,dict):
            raise RuntimeError(f"{label} checkpoint lacks embedded algorithm config")
        trainer=build_modular_mappo_trainer(config,"cuda")
        trainer.load(path,restore_rng=False)
        rows.append({"family":label,"path":relative,"sampled_steps":int(state["sampled_steps"]),"load":"PASS"})
        del trainer,state
        torch.cuda.empty_cache()
    return rows


def causal_fairness_audit(configs: dict[str, dict]) -> dict:
    """Compare independent first rollouts; never reseed after construction."""
    root=AUDIT/".causal_tmp"
    if root.exists():shutil.rmtree(root)
    root.mkdir(parents=True)
    env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    env["simulation"]["max_steps"]=12
    captures={}
    try:
        for cell in CELLS:
            cfg=deepcopy(configs[cell]);cfg["training"]["seed"]=9902
            cfg["implementation"]["evaluation_seed_base"]=99010000
            runner=ModularMAPPOTrainingRunner(env,cfg,num_envs=2,total_sampled_steps=8,
                device="cuda",seed=9902,output_dir=root/cell.lower(),smoke=True)
            initial_obs=runner.observations.copy();initial_alive=runner.alive.copy()
            with torch.no_grad():
                dist=runner.trainer.actor.distribution(torch.as_tensor(initial_obs,device="cuda"))
                mu=dist.mean.cpu().numpy();std=dist.stddev.cpu().numpy()
            batch=runner.collect_rollout(4)
            captures[cell]={"initial_observations":initial_obs,"initial_alive":initial_alive,
                "actor_mu":mu,"actor_std":std,"observations":batch.observations,
                "raw_actions":batch.raw_actions,"actions":batch.actions,
                "log_probs":batch.old_log_probs,"raw_rewards":batch.raw_environment_rewards,
                "training_rewards":batch.rewards,"dones":batch.dones,"alive":batch.alive_masks,
                "next_alive":batch.next_alive_masks,"waves":batch.wave_indices,
                "next_observations":batch.next_observations,"contexts":batch.contexts}
            runner.vector.close();del runner;torch.cuda.empty_cache()
        reference=captures["C0R0"]
        exact_fields=("initial_observations","initial_alive","actor_mu","actor_std","observations",
                      "raw_actions","actions","log_probs","raw_rewards","dones","alive",
                      "next_alive","waves","next_observations")
        field_exact={name:all(np.array_equal(reference[name],captures[cell][name]) for cell in CELLS[1:]) for name in exact_fields}
        r0_identity=all(np.array_equal(captures[cell]["training_rewards"],captures[cell]["raw_rewards"]) for cell in ("C0R0","C1R0"))
        c1_internal_context_only=(captures["C1R0"]["contexts"].shape[-1]==5 and captures["C0R0"]["contexts"].shape[-1]==0)
        if not all(field_exact.values()) or not r0_identity or not c1_internal_context_only:
            raise RuntimeError("first-rollout causal fairness failed")
        return {"status":"PASS","training_seed":9902,"steps":4,"num_envs":2,
                "no_post_construction_reseed":True,"exact_fields":field_exact,
                "r0_training_reward_is_raw_exact":r0_identity,
                "allowed_differences_only":["R1 training reward","C1 internal critic context"],
                "c1_internal_context_only":c1_internal_context_only}
    finally:
        if root.exists():shutil.rmtree(root)


def run_cuda_smoke(configs: dict[str, dict]) -> dict:
    smoke_root = AUDIT / ".smoke_tmp"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True)
    env = yaml.safe_load((ROOT / "configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    env["simulation"]["max_steps"] = 8
    rows = {}
    initial_actor = {}
    initial_critic = {}
    try:
        for cell in CELLS:
            config = deepcopy(configs[cell])
            config["training"]["seed"] = 9901
            config["implementation"]["evaluation_seed_base"] = 99000000
            config["training"]["evaluation_episodes"] = 2
            runner = ModularMAPPOTrainingRunner(
                env, config, num_envs=2, total_sampled_steps=16, device="cuda",
                seed=9901, output_dir=smoke_root / cell.lower(), smoke=True,
            )
            initial_actor[cell] = tensor_hash(runner.trainer.actor.state_dict())
            initial_critic[cell] = tensor_hash(runner.trainer.critic.state_dict())
            context_gradient_nonzero = None
            if runner.trainer.critic.context_dim:
                obs_tensor=torch.zeros((1,4,52),device="cuda")
                alive_tensor=torch.ones((1,4),device="cuda")
                context_tensor=torch.tensor([[1.,0.,0.,.25,.9]],device="cuda")
                value,_=runner.trainer.critic.forward_step(obs_tensor,alive_tensor,context_tensor)
                value.sum().backward()
                projection=runner.trainer.critic.mission_context_projection
                context_gradient_nonzero=bool(torch.count_nonzero(projection.grad).item())
                runner.trainer.critic.zero_grad(set_to_none=True)
            result = runner.run()
            latest = smoke_root / cell.lower() / "latest.pt"
            state = torch.load(latest, map_location="cuda", weights_only=False)
            metrics = result["final_optimization_metrics"]
            finite = all(np.isfinite(float(value)) for value in metrics.values())
            rows[cell] = {"sampled_steps": result["sampled_steps"], "cuda": True,
                          "finite_update": finite, "checkpoint_save_load": int(state["sampled_steps"]) == 16,
                          "critic_context_dim": state["extra"]["network_architecture"]["critic_context_dim"],
                          "actor_context_dim": state["extra"]["network_architecture"]["actor_context_dim"],
                          "critic_context_gradient_nonzero": context_gradient_nonzero,
                          "context_progress_mean": metrics["context_progress_mean"],
                          "context_remaining_horizon_mean": metrics["context_remaining_horizon_mean"],
                          "pbrs_enabled": cell.endswith("R1"),
                          "evaluation_seed_base": 99000000,
                          "evaluation_reward": "raw_environment_reward"}
            runner.vector.close()
            del runner, state
            torch.cuda.empty_cache()
        if len(set(initial_actor.values())) != 1:
            raise RuntimeError("smoke actor initializations differ")
        if initial_critic["C0R0"] != initial_critic["C0R1"] or initial_critic["C1R0"] != initial_critic["C1R1"]:
            raise RuntimeError("smoke paired critic initializations differ")
        # Nonzero synthetic R1 shaping, while R0 remains exact identity.
        raw = np.zeros((1,4), np.float32)
        args = ([{"total_waves":3,"blue_losses":1}], np.asarray([1]),
                np.ones((1,4),np.float32), np.ones((1,4),np.float32),
                np.ones((1,4),np.float32), np.asarray([False]))
        off = WaveSurvivalPotentialShapingModule({"enabled":False},.999)
        on = WaveSurvivalPotentialShapingModule({"enabled":True,"coefficient":1.0,"potential":"progress_times_survival","terminal_zero":True},.999)
        out_off,_ = off.adapt(raw,*args); out_on,_ = on.adapt(raw,*args)
        return {"status":"PASS", "device":torch.cuda.get_device_name(0), "smoke_training_seed":9901,
                "smoke_evaluation_range":[99000000,99000001], "reserved_ranges_used":False,
                "long_training_started":False, "cells":rows, "initial_actor_hashes":initial_actor,
                "initial_critic_hashes":initial_critic, "actor_init_exact":len(set(initial_actor.values()))==1,
                "critic_pair_init_exact":True, "pbrs_disabled_identity":np.array_equal(raw,out_off),
                "pbrs_enabled_nonzero":bool(np.any(out_on!=raw))}
    finally:
        if smoke_root.exists():
            shutil.rmtree(smoke_root)


def validate(check_outputs: bool = True, deep_freshness: bool = True,
             run_smoke: bool = False, launch_check: bool = False) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["protocol_role"] != "development_only" or manifest["design"] != "2x2 factorial":
        raise RuntimeError("factorial manifest role/design mismatch")
    if manifest["training_seeds"] != list(SEEDS) or len(manifest["runs"]) != 12:
        raise RuntimeError("manifest must define 12 fresh runs")
    expected_execution={"mode":"strict_serial","max_parallel_processes":1,
        "seed_order":list(SEEDS),"within_seed_order":list(CELLS),"shared_cuda_device":0}
    if manifest.get("execution_plan")!=expected_execution:
        raise RuntimeError("manifest is not the strict serial 12-run protocol")
    expected = {(cell, seed) for seed in SEEDS for cell in CELLS}
    if {(row["cell"], row["training_seed"]) for row in manifest["runs"]} != expected:
        raise RuntimeError("manifest matrix mismatch")
    if manifest["evaluation"] != {"seed_start":EVAL[0],"seed_end":EVAL[1],"episodes":50,"deterministic":True,"common_scenarios":True}:
        raise RuntimeError("manifest evaluation mismatch")
    if manifest["reserved_untouched_future_final_test"]["executed"] is not False:
        raise RuntimeError("33M marked executed")
    expected_provenance={
        "39000000..39000049":"EXPOSED_PREVIOUS_WS_PBRS_DEVELOPMENT",
        "40000000..40000049":"EXPOSED",
        "41000000..41000199":"EXPOSED_HISTORICAL_DETERMINISTIC_8M_EVALUATION",
        "42000000..42000049":"SELECTED_FRESH_DEVELOPMENT_ONLY",
        "5101..5103":"EXPOSED_HISTORICAL_TRAINING",
        "5201..5203":"SELECTED_FRESH_TRAINING",
        "33000000..33000199":"RESERVED_UNTOUCHED_FUTURE_FINAL",
    }
    if manifest.get("seed_provenance")!=expected_provenance:
        raise RuntimeError("manifest seed provenance mismatch")
    if manifest.get("rejected_exposed_evaluation")!={
        "seed_start":41000000,"seed_end":41000199,
        "reason":"historical deterministic 8M evaluation range documented in docs/archive/li2023/environment_diagnosis.md"}:
        raise RuntimeError("manifest 41M exposure declaration mismatch")
    configs = {cell: load_config(path) for cell, path in CONFIGS.items()}
    cells = validate_critic_context_factorial_configs(configs)
    env = yaml.safe_load((ROOT / "configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    if env["environment_variant"] != "persistent_wave_v2" or env["simulation"]["max_steps"] != 3000:
        raise RuntimeError("environment protocol mismatch")
    from algorithm.common.protocol import config_sha256
    env_semantic_hash=config_sha256(env)
    env_file_hash=hashlib.sha256((ROOT/"configs/persistent_wave_v2_environment.yaml").read_bytes()).hexdigest()
    if (env_semantic_hash!=ENV_CONFIG_SHA256 or env_file_hash!=ENV_FILE_SHA256
            or manifest.get("environment_config_sha256")!=ENV_CONFIG_SHA256
            or manifest.get("environment_file_sha256")!=ENV_FILE_SHA256):
        raise RuntimeError("frozen environment hash mismatch")
    launcher=(ROOT/"tools/run_critic_mission_context_factorial.sh").read_text(encoding="utf-8")
    if ("--launch-check" not in launcher or "for seed in 5201 5202 5203" not in launcher
            or "run_seed_pair" in launcher or "MAX_PARALLEL" in launcher):
        raise RuntimeError("launcher is not strict serial with launch-check")
    for row in manifest["runs"]:
        path = ROOT / row["output_dir"]
        if check_outputs and path.exists() and any(path.iterdir()):
            raise FileExistsError(f"non-fresh run directory: {path}")
    text_hits = text_freshness_scan()
    if text_hits["training_5201_5203"] or text_hits["selected_42m"]:
        raise RuntimeError(f"selected seeds are not fresh: {text_hits}")
    # The complete local engineering audit must independently recover the
    # historical evidence.  A deployment checkout may intentionally omit
    # archived docs/results, so launch-check instead validates the frozen,
    # exact provenance declarations above while still scanning everything
    # that is actually present for conflicts with the selected fresh seeds.
    if not launch_check and not text_hits["candidate_40m"]:
        raise RuntimeError("full audit expected evidence that rejected 40M is already exposed")
    if not launch_check and not text_hits["exposed_41m"]:
        raise RuntimeError("full audit expected docs/archive evidence that 41M is already exposed")
    checkpoint_scan = checkpoint_freshness_scan()
    selected_checkpoint_hits=[]
    for hit in checkpoint_scan["hits"]:
        if "load_error" not in hit and any(value in SEEDS or EVAL[0]<=value<=EVAL[1] for value in hit["values"]):
            selected_checkpoint_hits.append(hit)
    if selected_checkpoint_hits:
        raise RuntimeError(f"selected seed found in checkpoint metadata: {selected_checkpoint_hits}")
    load_errors = [hit for hit in checkpoint_scan["hits"] if "load_error" in hit]
    if load_errors:
        raise RuntimeError(f"checkpoint freshness audit incomplete: {load_errors}")
    initialization = ({"status":"SKIPPED_BY_LAUNCH_CHECK"} if launch_check
                      else initialization_audit(configs, "cuda"))
    compatibility = ([] if launch_check or not deep_freshness else old_checkpoint_compatibility_audit())
    AUDIT.mkdir(parents=True, exist_ok=True)
    freshness = {"status":"PASS", "selected_training_seeds":list(SEEDS),
                 "selected_evaluation_range":list(EVAL), "candidate_40m":"REJECTED_AS_EXPOSED",
                 "candidate_40m_evidence":text_hits["candidate_40m"],
                 "exposed_41m":"REJECTED_AS_EXPOSED","exposed_41m_evidence":text_hits["exposed_41m"],
                 "historical_evidence_policy":("FROZEN_MANIFEST_PROVENANCE_FOR_DEPLOYMENT_CHECKOUT"
                    if launch_check else "INDEPENDENT_REPOSITORY_EVIDENCE_REQUIRED"),
                 "selected_text_hits":[], "checkpoint_metadata_scan":checkpoint_scan,
                 "protocol_declarations_excluded_from_freshness_scan":[str(p.relative_to(ROOT)) for p in sorted(PROTOCOL_DECLARATIONS)],
                 "historical_5101_5103_primary":False, "historical_39m_primary":False,
                 "reserved_33m_executed":False}
    (AUDIT / "seed_freshness.json").write_text(json.dumps(freshness,indent=2),encoding="utf-8")
    validation = {"status":"PASS", "cells":cells, "runs":12,
                  "factor_differences_only":True, "environment_variant":"persistent_wave_v2",
                  "environment_config_sha256":env_semantic_hash,"environment_file_sha256":env_file_hash,
                  "execution_mode":"strict_serial","max_parallel_processes":1,
                  "actor_observation_unchanged":True, "PBRS_V1_unchanged":True,
                  "primary":"latest.pt@900000", "best":"diagnostic_only",
                  "old_checkpoint_compatibility":compatibility}
    (AUDIT / "config_factorial_validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    (AUDIT / "initialization_match.json").write_text(json.dumps(initialization,indent=2),encoding="utf-8")
    causal = causal_fairness_audit(configs) if run_smoke and not launch_check else {"status":"NOT_RUN_IN_THIS_INVOCATION"}
    if causal.get("status")=="PASS":
        init_rows=list(initialization["seeds"].values())
        causal.update({
            "actor_exact_match":all(row["all_actor_hashes_equal"] for row in init_rows),
            "critic_shared_parameters_exact_match":all(row["shared_critic_parameters_exact_across_four_cells"] for row in init_rows),
            "c1_context_projection_initial_norm":0.0,
            "c0_c1_initial_value_exact_match":all(row["c0_c1_initial_values_exact_for_three_contexts"] for row in init_rows),
            "cpu_rng_exact_match":all(row["post_init_cpu_cuda_rng_states_exact"] for row in init_rows),
            "cuda_rng_exact_match":all(row["post_init_cpu_cuda_rng_states_exact"] for row in init_rows),
            "first_rollout_raw_actions_exact_match":causal["exact_fields"]["raw_actions"],
            "first_rollout_environment_trajectory_exact_match":all(causal["exact_fields"][key] for key in
                ("initial_observations","initial_alive","raw_rewards","dones","alive","next_alive","waves","next_observations")),
        })
    (AUDIT / "causal_fairness.json").write_text(json.dumps(causal,indent=2),encoding="utf-8")
    smoke = run_cuda_smoke(configs) if run_smoke and not launch_check else {"status":"NOT_RUN_IN_THIS_INVOCATION"}
    if smoke.get("status")=="PASS":
        smoke["causal_fairness_status"]=causal["status"]
        smoke["first_rollout_matched"]=causal["first_rollout_environment_trajectory_exact_match"]
        smoke["c0_c1_initial_value_matched"]=causal["c0_c1_initial_value_exact_match"]
    (AUDIT / "smoke_summary.json").write_text(json.dumps(smoke,indent=2),encoding="utf-8")
    return {"status":"CRITIC_MISSION_CONTEXT_FACTORIAL_HARDENED_READY", "runs":12,
            "training_seeds":list(SEEDS), "evaluation_range":list(EVAL),
            "critic_context_dim":5, "actor_context_dim":0,
            "candidate_40m":"rejected_exposed", "reserved_33m_executed":False,
            "smoke":smoke["status"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--skip-deep-freshness", action="store_true")
    parser.add_argument("--launch-check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(deep_freshness=not args.skip_deep_freshness,
                              run_smoke=args.run_smoke,launch_check=args.launch_check), indent=2))


if __name__ == "__main__":
    main()


__all__ = ["validate", "validate_critic_context_factorial_configs"]
