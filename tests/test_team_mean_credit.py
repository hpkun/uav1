from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from algorithm.mappo.trainer import compute_gae
from algorithm.common.protocol import (
    aggregate_runtime_source_manifest,
    runtime_source_branch_provenance,
    runtime_source_manifest,
)
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.protocol import validate_team_credit_branch
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer
from algorithm.modules import TeamMeanCreditModule
from algorithm.train_modular_mappo import load_config, write_run_config
from tools.analyze_team_credit_screen import (
    MIN_W2_ENTRY_COUNT_PER_BRANCH,
    assert_finite_numeric_tree,
    check_text_failure_markers,
    optional_delta,
    verify_runtime_intervention,
    verify_six_branch_runtime_source_consistency,
    w2_entry_sample_status,
)

ROOT=Path(__file__).resolve().parents[1]

def rollout(trainer):
 rng=np.random.default_rng(91);t,e,n=4,2,4
 obs=rng.normal(size=(t,e,n,52)).astype("f");alive=np.ones((t,e,n),"f");alive[2:,1,3]=0
 raw=rng.normal(size=(t,e,n,3)).astype("f");actions=np.tanh(raw).astype("f")
 with torch.no_grad():old=trainer.actor._squashed_log_prob(trainer.actor.distribution(torch.from_numpy(obs)),torch.from_numpy(raw),torch.from_numpy(actions)).numpy()*alive
 rewards=np.zeros((t,e,n),"f");rewards[...,0]=10;rewards[...,1]=-2;rewards[2:,1,3]=99
 z=np.zeros((t,e,0),"f")
 return ModularRolloutBatch(obs,actions,raw,old,rewards,rewards.copy(),np.zeros((t,e),"f"),alive,obs.copy(),alive.copy(),np.tile(np.arange(1,5)[:,None],(1,e)).clip(max=3),np.full((t,e),3),z,z,episode_masks=np.ones((t,e),"f"))

def nested_equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(nested_equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 return a==b

@pytest.mark.parametrize("rewards,alive,expected",[
 ([10,0,-10,0],[1,1,1,1],[0,0,0,0]),
 ([10,0,0,0],[1,1,1,1],[2.5,2.5,2.5,2.5]),
 ([10,-10,0,0],[1,1,0,0],[0,0,0,0]),
 ([-10,0,0,0],[1,1,1,0],[-10/3,-10/3,-10/3,0]),
])
def test_transform_examples_and_sum_preservation(rewards,alive,expected):
 module=TeamMeanCreditModule({"enabled":True,"mode":"alive_sum_preserving_mean"})
 r=torch.tensor(rewards,dtype=torch.float32).view(1,1,4);m=torch.tensor(alive,dtype=torch.float32).view(1,1,4);w=torch.ones((1,1),dtype=torch.long)
 out,metrics=module.transform(r,m,w)
 assert torch.allclose(out,torch.tensor(expected,dtype=r.dtype).view(1,1,4),atol=1e-6)
 assert torch.allclose(out.sum(-1),(r*m).sum(-1),atol=1e-6)
 assert torch.equal(out[m==0],torch.zeros_like(out[m==0]));assert metrics["team_credit_sum_abs_error_max"]<=1e-6
 assert out.dtype==r.dtype and out.device==r.device and torch.isfinite(out).all()

def test_zero_alive_death_semantics_disabled_and_validation():
 module=TeamMeanCreditModule({"enabled":True,"mode":"alive_sum_preserving_mean"})
 rewards=torch.tensor([[[5.,-2.,8.,1.]],[[1.,2.,3.,4.]]]);alive=torch.tensor([[[1.,1.,1.,1.]],[[0.,0.,0.,0.]]])
 out,_=module.transform(rewards,alive,torch.ones((2,1),dtype=torch.long));assert torch.equal(out[1],torch.zeros_like(out[1]))
 # alive-before includes an agent dying on this transition; next_alive is not an input.
 assert out[0,0,0]!=0
 disabled=TeamMeanCreditModule({"enabled":False,"mode":"alive_sum_preserving_mean"});same,_=disabled.transform(rewards,alive,torch.ones((2,1),dtype=torch.long));assert same is rewards
 with pytest.raises(ValueError):TeamMeanCreditModule({"enabled":True,"mode":"bad"})
 with pytest.raises(ValueError):TeamMeanCreditModule({"enabled":True,"mode":"alive_sum_preserving_mean","alpha":.5})

def test_control_update_bitwise_plain_and_treatment_diverges():
 modules={"actor_lr_decay":{"enabled":True,"schedule":"delayed_linear","start_step":600000,"end_step":900000,"start_lr":3e-4,"end_lr":1e-4}}
 plain=ModularMAPPOTrainer(hidden_dim=32,ppo_epochs=1,minibatch_size=32,seed=7,modules_config=deepcopy(modules))
 control=ModularMAPPOTrainer(hidden_dim=32,ppo_epochs=1,minibatch_size=32,seed=7,modules_config={**deepcopy(modules),"team_mean_credit":{"enabled":False,"mode":"alive_sum_preserving_mean"}})
 treatment=ModularMAPPOTrainer(hidden_dim=32,ppo_epochs=1,minibatch_size=32,seed=7,modules_config={**deepcopy(modules),"team_mean_credit":{"enabled":True,"mode":"alive_sum_preserving_mean"}})
 batch=rollout(plain);rng=torch.get_rng_state();mp=plain.update(batch);torch.set_rng_state(rng);mc=control.update(batch);torch.set_rng_state(rng);mt=treatment.update(batch)
 assert nested_equal(plain.actor.state_dict(),control.actor.state_dict());assert nested_equal(plain.critic.state_dict(),control.critic.state_dict())
 assert nested_equal(plain.actor_optimizer.state_dict(),control.actor_optimizer.state_dict());assert nested_equal(plain.critic_optimizer.state_dict(),control.critic_optimizer.state_dict())
 for key in ("actor_loss","value_loss","entropy","approx_kl","actor_optimizer_steps_this_update","critic_optimizer_steps_this_update"):assert mp[key]==mc[key]
 assert not nested_equal(control.actor.state_dict(),treatment.actor.state_dict());assert not nested_equal(control.critic.state_dict(),treatment.critic.state_dict())
 assert mt["team_credit_enabled"]==1 and mt["team_credit_sum_abs_error_max"]<=1e-6
 assert mt["actor_optimizer_steps_this_update"]==mc["actor_optimizer_steps_this_update"] and mt["critic_optimizer_steps_this_update"]==mc["critic_optimizer_steps_this_update"]

def test_treatment_changes_gae_and_returns():
 r=torch.tensor([[[10.,0.,0.,0.]],[[0.,0.,0.,0.]]]);alive=torch.ones_like(r);done=torch.zeros((2,1));v=torch.zeros_like(r)
 team,_=TeamMeanCreditModule({"enabled":True,"mode":"alive_sum_preserving_mean"}).transform(r,alive,torch.ones((2,1),dtype=torch.long))
 al,rl=compute_gae(r,v,v,done,alive,alive,.999,.95);at,rt=compute_gae(team,v,v,done,alive,alive,.999,.95)
 assert not torch.equal(al,at) and not torch.equal(rl,rt)

def test_branch_validator_control_treatment_and_rejections():
 source=ROOT/'outputs/diag_mappo_learnability/l3_seed5301/checkpoint_1505280.pt'
 if not source.exists():pytest.skip("formal source checkpoint unavailable")
 state=torch.load(source,map_location="cpu",weights_only=False);env=yaml.safe_load((ROOT/'configs/persistent_wave_v2_environment.yaml').read_text())
 control=load_config(ROOT/'configs/dev_team_credit_control_300k.yaml');treatment=load_config(ROOT/'configs/dev_team_credit_teammean_300k.yaml')
 assert validate_team_credit_branch(state,env,control,{"training_seed":5301,"training_num_envs":24,"training_smoke":False})["intervention"]=="team_mean_credit_control"
 assert validate_team_credit_branch(state,env,treatment,{"training_seed":5301,"training_num_envs":24,"training_smoke":False})["intervention"]=="team_mean_credit"
 mutations=[lambda c:c["development_branch"].update(source_sampled_steps=1),lambda c:c["development_branch"].update(target_sampled_steps=2),lambda c:c["training"].update(gamma=.99),lambda c:c["modules"]["wave_balancing"].update(enabled=True)]
 for mutate in mutations:
  bad=deepcopy(treatment);mutate(bad)
  with pytest.raises(RuntimeError):validate_team_credit_branch(state,env,bad)
 bad=deepcopy(treatment);bad['development_branch']['actor_optimizer_restore']=False
 with pytest.raises(RuntimeError,match='restore'):validate_team_credit_branch(state,env,bad)
 bad_env=deepcopy(env);bad_env['reward']['kill_reward']=999
 with pytest.raises(RuntimeError,match='environment'):validate_team_credit_branch(state,bad_env,treatment)
 with pytest.raises(RuntimeError):validate_team_credit_branch(state,env,treatment,{"training_seed":999,"training_num_envs":24,"training_smoke":False})
 bad=deepcopy(state);bad.pop("rng_state")
 with pytest.raises(RuntimeError,match="required state"):validate_team_credit_branch(bad,env,treatment)

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_cuda_transform():
 r=torch.randn(3,2,4,device="cuda");m=torch.tensor([1,1,1,0],device="cuda").expand_as(r);w=torch.ones((3,2),dtype=torch.long,device="cuda")
 out,metrics=TeamMeanCreditModule({"enabled":True}).transform(r,m,w);assert out.is_cuda and torch.isfinite(out).all() and metrics["team_credit_sum_abs_error_max"]<=1e-6


@pytest.mark.parametrize(
    "control_count,treatment_count,expected",
    [
        (0, 0, "INSUFFICIENT_ENTRY_SAMPLES"),
        (1, 1, "INSUFFICIENT_ENTRY_SAMPLES"),
        (29, 29, "INSUFFICIENT_ENTRY_SAMPLES"),
        (30, 30, "VALID"),
        (30, 29, "INSUFFICIENT_ENTRY_SAMPLES"),
    ],
)
def test_w2_entry_minimum_sample_rule(control_count, treatment_count, expected):
 assert MIN_W2_ENTRY_COUNT_PER_BRANCH == 30
 assert w2_entry_sample_status(control_count, treatment_count) == expected


def test_optional_q2_q3_delta():
 assert optional_delta(.75, .5) == pytest.approx(.25)
 assert optional_delta(None, .5) is None
 assert optional_delta(.5, None) is None


def _analysis_protocol(branch):
 treatment=branch=="teammean"
 run={
  "development_method":"team_credit_matched_teammean" if treatment else "team_credit_matched_control",
  "enabled_modules":["actor_lr_decay","team_mean_credit"] if treatment else ["actor_lr_decay"],
 }
 if treatment:run.update({"team_mean_credit_enabled":True,"team_mean_credit_version":1,"team_mean_credit_mode":"alive_sum_preserving_mean","team_mean_credit_reward_scope":"training_credit_only","team_mean_credit_sum_preserving":True})
 provenance={
  "intervention":"team_mean_credit" if treatment else "team_mean_credit_control",
  "parent_sampled_steps":1505280,"source_sampled_steps":1505280,
  "target_sampled_steps":1805280,"additional_sampled_steps":300000,
 }
 row={
  "sampled_steps":1530000,"team_credit_enabled":1 if treatment else 0,
  "team_credit_live_sample_count":10 if treatment else 0,
  "team_credit_original_live_reward_sum":1.0,
  "team_credit_transformed_live_reward_sum":1.0,
  "team_credit_sum_abs_error_max":0.0,
  "team_credit_mean_abs_redistribution":.25 if treatment else 0.0,
 }
 return run,provenance,[row]


def test_analysis_runtime_identity_and_activation_rejections():
 for branch in ("control","teammean"):
  run,provenance,rows=_analysis_protocol(branch)
  result=verify_runtime_intervention(branch,run,provenance,rows,branch)
  assert result["runtime_intervention_verified"] is True
 run,provenance,rows=_analysis_protocol("teammean");run["enabled_modules"]=["actor_lr_decay"]
 with pytest.raises(RuntimeError,match="enabled_modules"):verify_runtime_intervention("teammean",run,provenance,rows,"treatment")
 run,provenance,rows=_analysis_protocol("teammean");rows[0]["team_credit_enabled"]=0
 with pytest.raises(RuntimeError,match="team_credit_enabled"):verify_runtime_intervention("teammean",run,provenance,rows,"treatment")
 run,provenance,rows=_analysis_protocol("control");rows[0]["team_credit_enabled"]=1
 with pytest.raises(RuntimeError,match="team_credit_enabled"):verify_runtime_intervention("control",run,provenance,rows,"control")


def test_analysis_rejects_zero_treatment_redistribution():
 run,provenance,rows=_analysis_protocol("teammean");rows[0]["team_credit_mean_abs_redistribution"]=0.0
 with pytest.raises(RuntimeError,match="zero throughout"):verify_runtime_intervention("teammean",run,provenance,rows,"treatment")


def test_numeric_nonfinite_scan_and_info_string():
 for value in (float("nan"),float("inf"),float("-inf")):
  with pytest.raises(RuntimeError,match="sampled_steps=17"):
   assert_finite_numeric_tree({"sampled_steps":17,"metric":value},"optimization_metrics.jsonl")
 assert_finite_numeric_tree({"sampled_steps":17,"message":"ordinary info message"},"optimization_metrics.jsonl")
 check_text_failure_markers("ordinary info message", "train.log")


def test_treatment_method_identity_metadata():
 trainer=ModularMAPPOTrainer(hidden_dim=32,ppo_epochs=1,minibatch_size=32,seed=7,modules_config={"actor_lr_decay":{"enabled":True},"team_mean_credit":{"enabled":True,"mode":"alive_sum_preserving_mean"}})
 runner=object.__new__(ModularMAPPOTrainingRunner);runner.trainer=trainer;runner.algorithm_config={"development_method":"team_credit_matched_teammean"};runner.env_config={"observation":{"include_own_fire_ready":False}}
 identity=runner.method_identity()
 assert identity["team_mean_credit_enabled"] is True
 assert identity["team_mean_credit_version"]==1
 assert identity["team_mean_credit_mode"]=="alive_sum_preserving_mean"
 assert identity["team_mean_credit_reward_scope"]=="training_credit_only"
 assert identity["team_mean_credit_sum_preserving"] is True


def test_runtime_source_manifest_determinism_scope_content_and_mtime(tmp_path):
 (tmp_path/"algorithm").mkdir();(tmp_path/"env").mkdir();(tmp_path/"tests").mkdir();(tmp_path/"tools").mkdir();(tmp_path/"outputs").mkdir()
 (tmp_path/"algorithm"/"a.py").write_text("A=1\n",encoding="utf-8")
 (tmp_path/"env"/"b.py").write_text("B=2\n",encoding="utf-8")
 (tmp_path/"algorithm"/"note.txt").write_text("ignored",encoding="utf-8")
 (tmp_path/"tests"/"test_x.py").write_text("ignored",encoding="utf-8")
 (tmp_path/"tools"/"tool.py").write_text("ignored",encoding="utf-8")
 (tmp_path/"outputs"/"artifact.py").write_text("ignored",encoding="utf-8")
 (tmp_path/"algorithm"/"__pycache__").mkdir();(tmp_path/"algorithm"/"__pycache__"/"hidden.py").write_text("ignored",encoding="utf-8")
 first=runtime_source_manifest(tmp_path);second=runtime_source_manifest(tmp_path)
 assert first==second
 assert [row["path"] for row in first["runtime_source_manifest_files"]]==["algorithm/a.py","env/b.py"]
 assert aggregate_runtime_source_manifest(list(reversed(first["runtime_source_manifest_files"])))==first["runtime_source_manifest_sha256"]
 a=tmp_path/"algorithm"/"a.py";stat=a.stat();os.utime(a,(stat.st_atime+10,stat.st_mtime+10))
 assert runtime_source_manifest(tmp_path)["runtime_source_manifest_sha256"]==first["runtime_source_manifest_sha256"]
 a.write_text("A=changed\n",encoding="utf-8")
 assert runtime_source_manifest(tmp_path)["runtime_source_manifest_sha256"]!=first["runtime_source_manifest_sha256"]


def test_runtime_source_manifest_contains_required_core_files():
 manifest=runtime_source_manifest(ROOT);paths={row["path"] for row in manifest["runtime_source_manifest_files"]}
 required={"algorithm/modular_mappo/trainer.py","algorithm/modular_mappo/runner.py","algorithm/modular_mappo/protocol.py","algorithm/train_modular_mappo.py","algorithm/modules/team_mean_credit.py","algorithm/mappo/trainer.py","env/persistent_env.py","env/combat_env.py","env/reward.py","env/observation.py","env/weapon.py"}
 assert required<=paths and manifest["runtime_source_manifest_file_count"]==len(manifest["runtime_source_manifest_files"])


def test_analyzer_six_branch_runtime_source_consistency_and_mismatch():
 manifest=runtime_source_manifest(ROOT);runs=[(f"branch-{index}",deepcopy(manifest)) for index in range(6)]
 result=verify_six_branch_runtime_source_consistency(runs)
 assert result["runtime_source_consistency_verified"] is True
 changed=deepcopy(manifest);changed["runtime_source_manifest_files"][0]["sha256"]="0"*64
 changed["runtime_source_manifest_sha256"]=aggregate_runtime_source_manifest(changed["runtime_source_manifest_files"])
 runs[-1]=(runs[-1][0],changed)
 with pytest.raises(RuntimeError,match="FORMAL_RUNTIME_SOURCE_MISMATCH"):verify_six_branch_runtime_source_consistency(runs)


def test_runtime_source_branch_provenance_historical_missing_and_destination_required():
 manifest=runtime_source_manifest(ROOT);provenance=runtime_source_branch_provenance({"extra":{}},manifest)
 assert provenance["source_runtime_source_manifest_sha256"] is None
 assert provenance["source_runtime_source_provenance_available"] is False
 assert provenance["destination_runtime_source_manifest_sha256"]==manifest["runtime_source_manifest_sha256"]
 with pytest.raises(RuntimeError,match="destination"):runtime_source_branch_provenance({"extra":{}},{})


def test_run_config_full_manifest_and_checkpoint_compact_manifest(tmp_path,monkeypatch):
 import algorithm.train_modular_mappo as train_module
 manifest=runtime_source_manifest(ROOT)
 trainer=SimpleNamespace(
  module_protocol=lambda:{"enabled_modules":["actor_lr_decay"],"module_config_sha256":"module"},
  warm_start_provenance={},anchor_provenance={},wave_entry_curriculum=SimpleNamespace(version=1),
 )
 runner=SimpleNamespace(
  seed=5301,device="cuda",num_envs=24,total_sampled_steps=1805280,smoke=False,
  env_config={"environment_variant":"persistent_wave_v2","environment_version":"2.3"},
  algorithm_config={"modules":{"curriculum":{},"wave_entry_curriculum":{}},"development_branch":{}},
  output_dir=tmp_path,trainer=trainer,branch_provenance={},runtime_source_manifest=manifest,
  environment_provenance=lambda:{},method_identity=lambda:{},
 )
 monkeypatch.setattr(train_module,"checkpoint_architecture",lambda _: {})
 value=write_run_config(tmp_path/"run_config.json",runner,Path("env.yaml"),Path("algorithm.yaml"))
 assert value["runtime_source_manifest_files"]==manifest["runtime_source_manifest_files"]
 assert value["runtime_source_manifest_file_count"]>0
 checkpoint_runner=object.__new__(ModularMAPPOTrainingRunner);checkpoint_runner.runtime_source_manifest=manifest
 compact=checkpoint_runner.runtime_source_checkpoint_provenance()
 assert compact=={"runtime_source_manifest_sha256":manifest["runtime_source_manifest_sha256"],"runtime_source_manifest_file_count":manifest["runtime_source_manifest_file_count"]}
