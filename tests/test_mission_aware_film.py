"""Focused tests for Mission-Aware Bounded Augmented FiLM-MAPPO."""
from __future__ import annotations

import copy,csv,json
from pathlib import Path
import numpy as np
import pytest
import torch

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modules import MissionFiLMModule,MISSION_FILM_VERSION
from algorithm.modules.wave_survival_pbrs import mission_context_numpy
from algorithm.train_modular_mappo import load_config
from tools.preflight_mission_aware_film import ENV,MANIFEST,REGISTRY,initialization_audit,load_yaml,resolve_protocol,summary_line
from tools.analyze_mission_aware_film import audit_run as film_audit_run

ROOT=Path(__file__).resolve().parents[1]
def configs():return load_config(ROOT/"configs/diag_mappo_learnability_common_3m.yaml"),load_config(ROOT/"configs/dev_actor_mission_context_3m.yaml"),load_config(ROOT/"configs/dev_mission_aware_film_3m.yaml")

def test_module_strict_v1_validation():
    good={"enabled":True,"mode":"bounded_augmented_film","encoder_hidden_dim":32,"alpha":.2,"augmented_residual":True,"identity_init":True}
    assert MissionFiLMModule(good).enabled and MISSION_FILM_VERSION==1
    for key,value in (("mode","other"),("encoder_hidden_dim",0),("alpha",0),("alpha",1.1),("identity_init",False),("augmented_residual",False)):
        bad={**good,key:value}
        with pytest.raises(ValueError):MissionFiLMModule(bad)

def test_disabled_plain_topology_and_raw_concat_are_unchanged():
    plain,raw,_=configs();p=build_modular_mappo_trainer(plain,"cpu",256,3_000_000);r=build_modular_mappo_trainer(raw,"cpu",256,3_000_000)
    assert not any(k.startswith(("mission_encoder.","gamma_head.","beta_head.","residual_head.")) for k in p.actor.state_dict())
    assert checkpoint_architecture(p)["actor_input_dim"]==52 and checkpoint_architecture(p)["actor_context_dim"]==0
    assert checkpoint_architecture(r)["actor_input_dim"]==57 and checkpoint_architecture(r)["actor_context_dim"]==5

def test_film_architecture_parameters_and_identity_fairness():
    plain,_,film=configs();p=build_modular_mappo_trainer(plain,"cpu",256,3_000_000);f=build_modular_mappo_trainer(film,"cpu",256,3_000_000);a=checkpoint_architecture(f)
    assert (a["actor_input_dim"],a["actor_context_dim"],a["critic_context_dim"])==(52,5,0)
    assert (checkpoint_architecture(p)["actor_parameter_count"],a["actor_parameter_count"],a["critic_parameter_count"])==(80902,107494,474113)
    assert a["actor_parameter_count"]-checkpoint_architecture(p)["actor_parameter_count"]==26592
    for prefix in ("backbone.","mean.","log_std."):
        for key,value in p.actor.state_dict().items():
            if key.startswith(prefix):assert torch.equal(value,f.actor.state_dict()[key])
    assert all(torch.equal(v,f.critic.state_dict()[k]) for k,v in p.critic.state_dict().items())

def test_identity_heads_bounds_gradients_and_context_learning():
    plain,_,film=configs();audit=initialization_audit(plain,film)
    for row in audit.values():
        assert row["base_actor_bit_identical"] and row["critic_bit_identical"]
        assert row["step0_mean_bit_identical"] and row["step0_log_std_bit_identical"] and row["heads_strict_zero"]
        assert row["mission_encoder_finite_nonzero"] and all(v>0 for v in row["first_head_gradient_norms"].values())
        assert row["first_mission_encoder_gradient_norm"]==0 and row["second_mission_encoder_gradient_norm"]>0
        assert row["learned_context_sensitive"] and .8<=row["bounds"]["gamma_min"]<=row["bounds"]["gamma_max"]<=1.2
        assert row["bounds"]["beta_abs_max"]<=.2 and row["bounds"]["residual_abs_max"]<=.2

def test_film_requires_context_and_critic_does_not():
    _,_,film=configs();t=build_modular_mappo_trainer(film,"cpu",32,10);obs=np.zeros((1,4,52),np.float32);alive=np.ones((1,4),np.float32)
    with pytest.raises(ValueError,match="mission_film actor context is required"):t.act(obs,alive,True)
    values,_=t.values_step(obs,alive);assert values.shape==(1,4)

def test_canonical_context_is_shared_by_training_eval_holdout_record():
    _,_,film=configs();t=build_modular_mappo_trainer(film,"cpu",32,10);args=(t,np.asarray([2]),np.asarray([3]),np.asarray([[1,0,1,0]],np.float32),np.asarray([900]),3000)
    expected=mission_context_numpy(*args);assert np.all(np.isfinite(expected)) and expected.shape==(1,5)
    for path in ("algorithm/modular_mappo/runner.py","algorithm/modular_mappo/evaluation.py","tools/run_formal_holdout.py","tools/record_combat_episode.py"):
        assert "mission_context_numpy" in (ROOT/path).read_text(encoding="utf-8")

def test_checkpoint_roundtrip_and_versions(tmp_path):
    _,_,film=configs();t=build_modular_mappo_trainer(film,"cpu",32,10);path=tmp_path/"film.pt";t.save(path,{"network_architecture":checkpoint_architecture(t)});before={k:v.clone() for k,v in t.actor.state_dict().items()}
    loaded=build_modular_mappo_trainer(film,"cpu",32,10);loaded.load(path)
    state=torch.load(path,map_location="cpu",weights_only=False);assert state["development_feature_versions"]["mission_film"]==1
    assert all(torch.equal(v,loaded.actor.state_dict()[k]) for k,v in before.items())

def test_old_plain_and_raw_checkpoint_roundtrip(tmp_path):
    plain,raw,_=configs()
    for index,cfg in enumerate((plain,raw)):
        original=build_modular_mappo_trainer(cfg,"cpu",32,10);path=tmp_path/f"old{index}.pt";original.save(path);restored=build_modular_mappo_trainer(cfg,"cpu",32,10);restored.load(path)
        assert set(original.actor.state_dict())==set(restored.actor.state_dict())

def test_identity_and_protocol_metadata_are_consistent():
    _,_,film=configs();t=build_modular_mappo_trainer(film,"cpu",256,3_000_000);r=object.__new__(ModularMAPPOTrainingRunner);r.trainer=t;r.algorithm_config=film
    identity=r.method_identity();arch=checkpoint_architecture(t);state=t.checkpoint_state(identity)
    for key in ("development_method","actor_input_dim","actor_context_dim","critic_context_dim","mission_film_enabled","mission_film_mode","mission_encoder_hidden_dim","mission_film_alpha","mission_film_identity_init","mission_film_augmented_residual"):
        expected=identity[key];assert state["extra"][key]==expected
    assert arch["actor_input_dim"]==identity["actor_input_dim"]

@pytest.mark.parametrize("mutation",[
    ("wave_context","context_target","actor_critic"),("wave_context","context_target","critic_only"),
    ("mission_film","alpha",.3),("mission_film","encoder_hidden_dim",16),("mission_film","identity_init",False),("mission_film","augmented_residual",False),
    ("recurrent_memory","enabled",True),("entity_attention","enabled",True),("multi_wave_reward","enabled",True),("wave_survival_pbrs","enabled",True),
    ("wave_balancing","enabled",True),("curriculum","enabled",True),("policy_anchor","enabled",True),("advantage_priority","enabled",True),
    ("ppo_stabilization","enabled",True),("warm_start","enabled",True),("critic_mission_context","enabled",True)])
def test_preflight_rejects_forbidden_protocol_mutations(mutation):
    _,_,cfg=configs();module,key,value=mutation;cfg["modules"][module][key]=value
    with pytest.raises((RuntimeError,ValueError)):resolve_protocol(check_outputs=False,config=cfg)

@pytest.mark.parametrize("field,value",[("waves",2),("steps",2000)])
def test_preflight_rejects_nonfinal_environment(field,value):
    env=load_yaml(ENV)
    if field=="waves":env["persistent_waves"]["total_waves"]=value
    else:env["simulation"]["max_steps"]=value
    with pytest.raises(RuntimeError):resolve_protocol(check_outputs=False,env=env)

def test_preflight_future_final_and_summary():
    registry=json.loads(REGISTRY.read_text(encoding="utf-8"));registry["evaluation_ranges"]["45000000..45000199"]["executed"]=True
    with pytest.raises(RuntimeError,match="future-final"):resolve_protocol(check_outputs=False,registry=registry)
    result=resolve_protocol(check_outputs=False);line=summary_line(result)
    assert line.startswith("[PREFLIGHT] READY") and "actor=52" in line and "mission_film=bounded_augmented_film/32D/a0.2" in line and "future_final=UNUSED" in line and "{" not in line

def test_compact_train_log_does_not_repeat_film_diagnostics():
    source=(ROOT/"algorithm/modular_mappo/runner.py").read_text(encoding="utf-8")
    body=source[source.index("    def train_log_line"):source.index("    def optimization_warning_line")]
    assert "film_delta_gamma" not in body and "film_saturation" not in body

def test_analyzer_is_cpu_only_and_never_imports_evaluator():
    source=(ROOT/"tools/analyze_mission_aware_film.py").read_text(encoding="utf-8")
    assert 'map_location="cpu"' in source and "evaluate_modular" not in source and "torch.cuda.is_available" not in source

def test_analyzer_checkpoint_audit_executes_without_cuda(tmp_path,monkeypatch):
    _,_,film=configs();env=load_yaml(ENV);trainer=build_modular_mappo_trainer(film,"cpu",256,3_000_000);trainer.sampled_steps=3_000_000
    r=object.__new__(ModularMAPPOTrainingRunner);r.trainer=trainer;r.algorithm_config=film;identity=r.method_identity();extra={**identity,"network_architecture":checkpoint_architecture(trainer),"environment_config":env,"current_total_waves":3,"curriculum_config":{"enabled":False}}
    state=trainer.checkpoint_state(extra)
    for name in ("latest.pt","final.pt","checkpoint_3000000.pt"):torch.save(state,tmp_path/name)
    import yaml
    for name in ("env_config.yaml","runtime_env_config.yaml"):(tmp_path/name).write_text(yaml.safe_dump(env),encoding="utf-8")
    from algorithm.common.protocol import config_sha256
    (tmp_path/"run_config.json").write_text(json.dumps({**identity,"network_architecture":checkpoint_architecture(trainer),"environment_config_sha256":config_sha256(env)}),encoding="utf-8")
    (tmp_path/"run_summary.json").write_text('{"sampled_steps":3000000}',encoding="utf-8")
    fields=["sampled_steps","clear_wave_1_probability","clear_wave_2_probability","clear_wave_3_probability"]
    with (tmp_path/"evaluation_history.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(dict(zip(fields,[3000000,.8,.5,.2])))
    (tmp_path/"training_metrics.jsonl").write_text("",encoding="utf-8");(tmp_path/"optimization_metrics.jsonl").write_text("",encoding="utf-8")
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    _,audit=film_audit_run(tmp_path,"mission_aware_film",5301);assert audit["checkpoint_map_location"]=="cpu" and audit["finite_checkpoint"]

def test_manifest_gate_and_serial_launcher_are_frozen():
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"));gate=manifest["development_gate"]
    assert gate["mean_delta_average_waves_min"]==.15 and gate["paired_average_waves_wins_min"]==2 and gate["mean_delta_w1_min"]==-.05
    launcher=(ROOT/"tools/run_mission_aware_film_3m.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in launcher and "for seed in 5301 5302 5303" in launcher and "--summary" in launcher and "nohup" not in launcher
