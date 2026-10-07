"""Formal CLI for modular MAPPO, including immutable run lineage and resume."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# spawn re-executes this file as __mp_main__ in each environment worker.
# Those workers only need env.process_worker, not Torch/CUDA or training code.
# Keep the real CLI and normal library-import paths unchanged.
if __name__ != "__mp_main__":
    import torch
    from algorithm.train_mappo import (
        TeeOutput, ensure_fresh_output_directory, load_run_config,
        prepare_resume_rollback, reject_stale_resume_checkpoint,
        resolve_runtime_settings, validate_resume_config_snapshots,
    )
    from algorithm.common.protocol import (
        config_sha256,
        runtime_source_branch_provenance,
        runtime_source_manifest,
    )
    from algorithm.mappo.trainer import MAPPO_IMPL_VERSION
    from algorithm.modular_mappo.protocol import (
        checkpoint_architecture, validate_modular_branch, validate_fbmr_stage2_branch,
        validate_fbmr_v2_stage2_branch,
        validate_modular_checkpoint,
        validate_swgp_branch,
        validate_team_credit_branch,
        validate_pwtr_branch,
        validate_w1sg_branch, validate_wsai_branch, validate_wsmh_branch, validate_actor_grad_clip_branch,
        validate_dawe_branch, validate_rv_branch,
    )
    from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
    from algorithm.modular_mappo.trainer import MODULAR_MAPPO_IMPL_VERSION


def _merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result.get(key, {}), value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def resolved(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_config(path: str | Path) -> dict:
    source = resolved(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if "extends" in data:
        parent = data.pop("extends")
        data = _merge(load_config(parent), data)
    return data


def default_output_dir(seed: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "outputs" / f"modular_{stamp}_seed{seed}"


def write_snapshots(output_dir: Path, env_config: dict, algorithm_config: dict) -> None:
    (output_dir / "env_config.yaml").write_text(yaml.safe_dump(env_config, sort_keys=False), encoding="utf-8")
    (output_dir / "algorithm_config.yaml").write_text(yaml.safe_dump(algorithm_config, sort_keys=False), encoding="utf-8")


def write_run_config(path: Path, runner: ModularMAPPOTrainingRunner,
                     env_path: Path, algorithm_path: Path) -> dict:
    protocol = runner.trainer.module_protocol()
    value = {
        "algorithm":"modular_mappo", "seed":runner.seed, "device":runner.device,
        "num_envs":runner.num_envs, "total_sampled_steps":runner.total_sampled_steps,
        "smoke":runner.smoke, "environment_variant":runner.env_config.get("environment_variant","direct_v2_3"),
        "environment_version":runner.env_config.get("environment_version"),
        "environment_config_path":str(env_path), "algorithm_config_path":str(algorithm_path),
        "environment_config_sha256":config_sha256(runner.env_config),
        **runner.environment_provenance(),
        **runner.method_identity(),
        "algorithm_config_sha256":config_sha256(runner.algorithm_config),
        "enabled_modules":protocol["enabled_modules"], "module_config_sha256":protocol["module_config_sha256"],
        "network_architecture":checkpoint_architecture(runner.trainer),
        "warm_start_provenance":runner.trainer.warm_start_provenance,
        "policy_anchor_provenance":runner.trainer.anchor_provenance,
        "curriculum_config":runner.algorithm_config.get("modules",{}).get("curriculum",{}),
        "wave_entry_curriculum_version":runner.trainer.wave_entry_curriculum.version,
        "wave_entry_curriculum_config":deepcopy(
            runner.algorithm_config.get("modules",{}).get("wave_entry_curriculum",{})
        ),
        "output_dir":str(runner.output_dir.resolve()),
        "start_timestamp":datetime.now().astimezone().isoformat(),
        "baseline_mappo_impl_version":MAPPO_IMPL_VERSION,
        "modular_mappo_impl_version":MODULAR_MAPPO_IMPL_VERSION,
        **deepcopy(runner.runtime_source_manifest),
        "branch_provenance":runner.branch_provenance,
        "development_branch":deepcopy(runner.algorithm_config.get("development_branch",{})),
    }
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return value

def future_row_counts(run_dir: Path, checkpoint_steps: int) -> dict[str,int]:
    counts={}
    for name in ("training_metrics.jsonl","optimization_metrics.jsonl"):
        path=run_dir/name
        rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
        counts[name]=sum(int(row["sampled_steps"])>checkpoint_steps for row in rows)
    path=run_dir/"evaluation_history.csv"
    if path.exists():
        import csv
        with path.open(newline="",encoding="utf-8") as stream:counts["evaluation_history.csv"]=sum(int(row["sampled_steps"])>checkpoint_steps for row in csv.DictReader(stream))
    else:counts["evaluation_history.csv"]=0
    return counts


def file_sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def preserve_resume_start_checkpoint(run_dir: Path, checkpoint: Path, state: dict) -> dict:
    """Keep one immutable, content-addressed checkpoint for each resume boundary."""
    digest=file_sha256(checkpoint);steps=int(state["sampled_steps"]);directory=run_dir/"resume_points"
    directory.mkdir(parents=True,exist_ok=True)
    destination=directory/f"resume_start_{steps}_{digest[:12]}.pt"
    created=False
    if destination.exists():
        if file_sha256(destination)!=digest:raise RuntimeError(f"resume-point hash collision: {destination}")
    else:
        shutil.copy2(checkpoint,destination);created=True
    metadata_path=destination.with_suffix(".json")
    metadata={"source_checkpoint":str(checkpoint.resolve()),"source_checkpoint_sha256":digest,"source_sampled_steps":steps,"preserved_checkpoint":str(destination.resolve()),"created":created,"timestamp":datetime.now().astimezone().isoformat()}
    if not metadata_path.exists():metadata_path.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return metadata


def resolve_branch_runtime(algorithm_config: dict, state: dict, *, seed: int | None,
                           num_envs: int | None, total_sampled_steps: int | None,
                           device: str | None, smoke: bool | None) -> dict:
    extra=state.get("extra",{});source_seed=int(extra["training_seed"]);source_envs=int(extra["training_num_envs"]);source_smoke=bool(extra["training_smoke"])
    if seed is not None and int(seed)!=source_seed:raise RuntimeError("branch seed differs from source checkpoint")
    if num_envs is not None and int(num_envs)!=source_envs:raise RuntimeError("branch num_envs differs from source checkpoint")
    if smoke is not None and bool(smoke)!=source_smoke:raise RuntimeError("branch smoke mode differs from source checkpoint")
    if total_sampled_steps is None:raise RuntimeError("branch mode requires explicit --total-sampled-steps")
    target=int(total_sampled_steps);checkpoint_steps=int(state["sampled_steps"])
    if target<=checkpoint_steps:raise RuntimeError("branch target must exceed parent sampled_steps")
    return {"seed":source_seed,"num_envs":source_envs,"total_sampled_steps":target,"device":str(algorithm_config["training"]["device"] if device is None else device),"smoke":source_smoke,"legacy_resume":False,"original":None,"extended_training_target":True}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--env-config",default="configs/persistent_wave_v2_environment.yaml")
    parser.add_argument("--algorithm-config",default="configs/modular_mappo_persistent.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device",choices=("cpu","cuda"));parser.add_argument("--seed",type=int)
    parser.add_argument("--num-envs",type=int);parser.add_argument("--total-sampled-steps",type=int)
    parser.add_argument("--smoke",action="store_true",default=None)
    source_group=parser.add_mutually_exclusive_group();source_group.add_argument("--resume");source_group.add_argument("--branch-from")
    parser.add_argument("--warm-start-checkpoint");parser.add_argument("--reference-checkpoint")
    args=parser.parse_args()
    runtime_manifest=runtime_source_manifest(ROOT)
    env_path, algorithm_path = resolved(args.env_config), resolved(args.algorithm_config)
    env_config=yaml.safe_load(env_path.read_text(encoding="utf-8"));algorithm_config=load_config(algorithm_path)
    resume_path=resolved(args.resume).resolve() if args.resume else None
    branch_path=resolved(args.branch_from).resolve() if args.branch_from else None
    if algorithm_config.get("development_method")=="delayed_swgp_mappo" and branch_path is not None:
        raise RuntimeError("Delayed-SWGP is a fresh 0-to-3M protocol and forbids --branch-from")
    state=None;run_config=None;rollback={};branch_validation=None;branch_provenance={};resume_point={}
    if resume_path is None and branch_path is None:
        if algorithm_config.get("development_branch",{}).get("intervention"):
            raise RuntimeError("Stage-2 development config requires explicit --branch-from")
        seed=int(algorithm_config["training"]["seed"] if args.seed is None else args.seed)
        output_dir=default_output_dir(seed) if args.output_dir is None else resolved(args.output_dir)
        # This is deliberately the first write-capable operation.
        ensure_fresh_output_directory(output_dir)
    elif resume_path is not None:
        if not resume_path.is_file():raise FileNotFoundError(resume_path)
        output_dir=resume_path.parent if args.output_dir is None else resolved(args.output_dir).resolve()
        if output_dir!=resume_path.parent:raise RuntimeError("resume output_dir must equal checkpoint.parent")
        validate_resume_config_snapshots(output_dir,env_config,algorithm_config)
        run_config=load_run_config(output_dir)
        if run_config is None:raise RuntimeError("formal modular resume requires run_config.json")
        state=torch.load(resume_path,map_location="cpu",weights_only=False)
        reject_stale_resume_checkpoint(output_dir,resume_path)
    else:
        if not branch_path.is_file():raise FileNotFoundError(branch_path)
        if args.output_dir is None:raise RuntimeError("branch mode requires explicit --output-dir")
        if args.warm_start_checkpoint or args.reference_checkpoint:raise RuntimeError("branch mode does not accept warm-start/reference checkpoints")
        state=torch.load(branch_path,map_location="cpu",weights_only=False)
        runtime=resolve_branch_runtime(algorithm_config,state,seed=args.seed,num_envs=args.num_envs,total_sampled_steps=args.total_sampled_steps,device=args.device,smoke=args.smoke)
        intervention=algorithm_config.get("development_branch",{}).get("intervention")
        if intervention and (runtime["total_sampled_steps"]!=int(algorithm_config["training"]["total_sampled_steps"]) or runtime["device"]!="cuda"):
            raise RuntimeError("development branch requires its exact configured target and CUDA runtime")
        pwtr_interventions={"pwtr_plain_control","pwtr_stratified","pwtr_current_extra","pwtr_uniform_recent","pwtr_priority_recent","pwtr_full",
                            "pwtr_current_actor_only","pwtr_current_critic_only","pwtr_recent_actor_only","pwtr_recent_critic_only"}
        validator=(validate_fbmr_v2_stage2_branch if intervention=="frozen_base_dual_bounded_mean_residual"
                   else validate_team_credit_branch if intervention in {"team_mean_credit","team_mean_credit_control"}
                   else validate_w1sg_branch if intervention=="w1sg_current_actor"
                   else validate_wsai_branch if intervention=="wave_specific_actor_isolation"
                   else validate_wsmh_branch if intervention=="wave_specific_mean_heads"
                   else validate_actor_grad_clip_branch if intervention in {"actor_grad_clip_05_control","actor_grad_clip_10"}
                   else validate_dawe_branch if intervention in {"dawe_fixed10_control","deployment_aligned_wave_exploration"}
                   else validate_rv_branch if intervention in {"rv_fixed10_control","reference_variance"}
                   else validate_pwtr_branch if intervention in pwtr_interventions
                   else validate_swgp_branch if intervention in {"sequential_wave_gradient_projection","sequential_wave_gradient_projection_control"}
                   else validate_fbmr_stage2_branch if intervention else validate_modular_branch)
        branch_validation=validator(state,env_config,algorithm_config,{"training_seed":runtime["seed"],"training_num_envs":runtime["num_envs"],"training_smoke":runtime["smoke"]})
        parent_digest=file_sha256(branch_path)
        output_dir=resolved(args.output_dir).resolve();ensure_fresh_output_directory(output_dir)
        allowed=(["actor_lr_decay","total_sampled_steps","output_directory","branch_metadata"] if not intervention else
                 ["total_sampled_steps","development_method","development_branch","team_mean_credit","branch_metadata"] if intervention in {"team_mean_credit","team_mean_credit_control"} else
                 ["total_sampled_steps","development_method","development_branch","persistent_wave_trajectory_replay","branch_metadata"] if intervention in pwtr_interventions else
                 ["total_sampled_steps","development_method","development_branch","persistent_wave_trajectory_replay","wave1_sensitivity_gating","branch_metadata"] if intervention=="w1sg_current_actor" else
                 ["total_sampled_steps","development_method","development_branch","wave_specific_actor_isolation","branch_metadata"] if intervention=="wave_specific_actor_isolation" else
                 ["total_sampled_steps","development_method","development_branch","wave_specific_mean_heads","branch_metadata"] if intervention=="wave_specific_mean_heads" else
                 ["total_sampled_steps","development_method","development_branch","actor_gradient_clipping","branch_metadata"] if intervention in {"actor_grad_clip_05_control","actor_grad_clip_10"} else
                 ["total_sampled_steps","development_method","development_branch","actor_gradient_clipping","deployment_aligned_wave_exploration","branch_metadata"] if intervention in {"dawe_fixed10_control","deployment_aligned_wave_exploration"} else
                 ["total_sampled_steps","development_method","development_branch","actor_gradient_clipping","reference_variance","branch_metadata"] if intervention in {"rv_fixed10_control","reference_variance"} else
                 ["total_sampled_steps","development_method","development_branch","sequential_wave_gradient_projection","branch_metadata"] if intervention in {"sequential_wave_gradient_projection","sequential_wave_gradient_projection_control"} else
                 ["total_sampled_steps","development_branch","entity_attention","actor_trainable_parameter_set","actor_optimizer_reset","actor_effective_lr","evaluation_seed_base","development_protocol"])
        branch_provenance={"branch_creation_mode":"explicit_branch_from","parent_checkpoint_path":str(branch_path),"parent_checkpoint_sha256":parent_digest,"parent_sampled_steps":int(state["sampled_steps"]),"source_training_seed":int(state.get("extra",{}).get("training_seed")),"destination_algorithm_config_sha256":config_sha256(algorithm_config),"destination_module_config_sha256":config_sha256(algorithm_config.get("modules",{})),"source_algorithm_config_sha256":state.get("extra",{}).get("algorithm_config_sha256"),"source_module_config_sha256":state.get("module_config_sha256"),"allowed_differences":allowed,**runtime_source_branch_provenance(state,runtime_manifest),**branch_validation}
    if branch_path is None:
        runtime=resolve_runtime_settings(algorithm_config,seed=args.seed,num_envs=args.num_envs,
            total_sampled_steps=args.total_sampled_steps,device=args.device,smoke=args.smoke,
            run_config=run_config,checkpoint_state=state)
    if resume_path is not None:
        validate_modular_checkpoint(state,env_config,algorithm_config,{
            "training_seed":runtime["seed"],"training_num_envs":runtime["num_envs"],"training_smoke":runtime["smoke"]})
        if run_config["module_config_sha256"]!=state["module_config_sha256"]:raise RuntimeError("run_config/checkpoint module protocol mismatch")
        if run_config.get("network_architecture")!=state.get("extra",{}).get("network_architecture"):raise RuntimeError("run_config/checkpoint network architecture mismatch")
        if run_config.get("warm_start_provenance",{})!=state.get("warm_start_provenance",{}):raise RuntimeError("warm-start provenance mismatch")
        if run_config.get("policy_anchor_provenance",{})!=state.get("anchor_provenance",{}):raise RuntimeError("policy-anchor provenance mismatch")
        truncated_counts=future_row_counts(output_dir,int(state["sampled_steps"]))
        resume_point=preserve_resume_start_checkpoint(output_dir,resume_path,state)
        rollback=prepare_resume_rollback(output_dir,resume_path,int(state["sampled_steps"]))
    runner=ModularMAPPOTrainingRunner(env_config,algorithm_config,runtime["num_envs"],runtime["total_sampled_steps"],runtime["device"],runtime["seed"],output_dir,runtime["smoke"],args.warm_start_checkpoint,args.reference_checkpoint,resume_mode=state is not None,branch_provenance=branch_provenance,runtime_source_manifest_data=runtime_manifest)
    if state is None or branch_path is not None:
        write_snapshots(output_dir,env_config,algorithm_config)
        write_run_config(output_dir/"run_config.json",runner,env_path,algorithm_path)
    if resume_path is not None:
        runner.resume(resume_path)
        original=runtime["original"]
        resume_record={"timestamp":datetime.now().astimezone().isoformat(),"resume_checkpoint":str(resume_path),
            "checkpoint_sampled_steps":int(state["sampled_steps"]),"original_target_budget":original["total_sampled_steps"],
            "new_target_budget":runtime["total_sampled_steps"],"extension":runtime["extended_training_target"],
            "seed":runtime["seed"],"num_envs":runtime["num_envs"],"old_device":original["device"],"new_device":runtime["device"],
            "logs_truncated":any(truncated_counts.values()),"truncated_row_counts":truncated_counts,"rollback":rollback,
            "best_restored_sampled_steps":runner.best_sampled_steps,"curriculum_stage":runner.current_stage,
            "enabled_modules":runner.trainer.module_protocol()["enabled_modules"],"resume_point_backup":resume_point}
        with (output_dir/"resume_history.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps(resume_record)+"\n")
    elif branch_path is not None:
        runner.branch_from(branch_path,branch_validation.get("intervention"),branch_provenance["parent_checkpoint_sha256"])
        source_digest_after=file_sha256(branch_path)
        if source_digest_after!=branch_provenance["parent_checkpoint_sha256"]:raise RuntimeError("source checkpoint changed during branch creation")
        branch_record={**branch_provenance,"source_checkpoint_sha256_after_load":source_digest_after,"source_checkpoint_unchanged":True,"rng_resume_metadata":runner.trainer.rng_restore_metadata,"created_at":datetime.now().astimezone().isoformat()}
        (output_dir/"branch_from.json").write_text(json.dumps(branch_record,indent=2),encoding="utf-8")
    with (output_dir/"train.log").open("a",encoding="utf-8") as log_stream:
        with redirect_stdout(TeeOutput(sys.stdout,log_stream)):
            runner.run()


if __name__=="__main__":main()
