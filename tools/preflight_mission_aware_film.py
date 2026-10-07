"""Static protocol and initialization audit for Mission-Aware FiLM-MAPPO."""
from __future__ import annotations

import argparse,hashlib,json,sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from algorithm.common.protocol import config_sha256
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture,validate_modular_checkpoint
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner,validate_runtime_environment_contract
from algorithm.modules.wave_survival_pbrs import mission_context_numpy
from algorithm.train_modular_mappo import load_config

ENV=ROOT/"configs/persistent_wave_v2_environment.yaml";BASE=ROOT/"configs/diag_mappo_learnability_common_3m.yaml"
METHOD=ROOT/"configs/dev_mission_aware_film_3m.yaml";MANIFEST=ROOT/"experiments/mission_aware_film_development_manifest.json"
REGISTRY=ROOT/"experiments/current_seed_provenance.json";AUDIT=ROOT/"outputs/mission_aware_film_preflight"
SMOKE=ROOT/"outputs/dev_mission_aware_film_smoke_v1/seed88300003";SEEDS=(5301,5302,5303)

def load_yaml(path):return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
def equal_state(a,b,names=None):
    keys=set(a) if names is None else {k for k in a if k.startswith(names)}
    return keys==({k for k in b} if names is None else {k for k in b if k.startswith(names)}) and all(torch.equal(a[k],b[k]) for k in keys)

def initialization_audit(base,film):
    rows={}
    for seed in SEEDS:
        bc,fc=deepcopy(base),deepcopy(film);bc["training"]["seed"]=fc["training"]["seed"]=seed
        plain=build_modular_mappo_trainer(bc,"cpu",256,3_000_000);candidate=build_modular_mappo_trainer(fc,"cpu",256,3_000_000)
        actor_equal=equal_state(plain.actor.state_dict(),candidate.actor.state_dict(),("backbone.","mean.","log_std."))
        critic_equal=equal_state(plain.critic.state_dict(),candidate.critic.state_dict())
        gen=torch.Generator().manual_seed(seed+901);obs=torch.randn(6,4,52,generator=gen)
        contexts=torch.tensor([[1,0,0,.1,1.0],[0,1,0,.45,.6],[0,0,1,.9,.1],[1,0,0,.2,.8],[0,1,0,.6,.4],[0,0,1,1.,0.]])
        contexts=contexts[:,None,:].expand(-1,4,-1)
        pd=plain.actor.distribution(obs);fd=candidate.actor.distribution(obs,contexts)
        step0_mean_equal=torch.equal(pd.mean,fd.mean);step0_logstd_equal=torch.equal(pd.scale.log(),fd.scale.log())
        heads=(candidate.actor.gamma_head,candidate.actor.beta_head,candidate.actor.residual_head)
        heads_zero=all(torch.count_nonzero(p).item()==0 for h in heads for p in (h.weight,h.bias))
        enc=list(candidate.actor.mission_encoder.parameters());encoder_finite=all(torch.isfinite(p).all() for p in enc)
        encoder_nonzero=any(torch.count_nonzero(p).item()>0 for p in enc)
        candidate.actor.zero_grad();candidate.actor.distribution(obs,contexts).mean.square().mean().backward()
        head_grad={name:float(sum((p.grad.detach().square().sum() for p in head.parameters() if p.grad is not None),torch.tensor(0.)).sqrt())
                   for name,head in (("gamma",heads[0]),("beta",heads[1]),("residual",heads[2]))}
        first_encoder_grad=float(sum((p.grad.detach().square().sum() for p in enc if p.grad is not None),torch.tensor(0.)).sqrt())
        opt=torch.optim.SGD(candidate.actor.parameters(),lr=.01);opt.step();candidate.actor.zero_grad()
        candidate.actor.distribution(obs,contexts).mean.square().mean().backward()
        second_encoder_grad=float(sum((p.grad.detach().square().sum() for p in enc if p.grad is not None),torch.tensor(0.)).sqrt())
        same_obs=obs[:1].expand(3,-1,-1);different=contexts[:3];learned=candidate.actor.distribution(same_obs,different).mean
        context_sensitive=not torch.equal(learned[0],learned[1]) or not torch.equal(learned[1],learned[2])
        _,diag=candidate.actor.distribution(obs,contexts,return_attention=True)
        bounds={"gamma_min":float(diag["film_gamma"].min().detach()),"gamma_max":float(diag["film_gamma"].max().detach()),
                "beta_abs_max":float(diag["film_beta"].abs().max().detach()),"residual_abs_max":float(diag["film_residual"].abs().max().detach())}
        good=actor_equal and critic_equal and step0_mean_equal and step0_logstd_equal and heads_zero and encoder_finite and encoder_nonzero and all(v>0 for v in head_grad.values()) and first_encoder_grad==0 and second_encoder_grad>0 and context_sensitive and bounds["gamma_min"]>=.8 and bounds["gamma_max"]<=1.2 and bounds["beta_abs_max"]<=.2 and bounds["residual_abs_max"]<=.2
        if not good:raise RuntimeError(f"initialization/learning-path audit failed for seed {seed}: {locals()}")
        rows[str(seed)]={"base_actor_bit_identical":actor_equal,"critic_bit_identical":critic_equal,
            "step0_mean_bit_identical":step0_mean_equal,"step0_log_std_bit_identical":step0_logstd_equal,
            "heads_strict_zero":heads_zero,"mission_encoder_finite_nonzero":encoder_finite and encoder_nonzero,
            "first_head_gradient_norms":head_grad,"first_mission_encoder_gradient_norm":first_encoder_grad,
            "second_mission_encoder_gradient_norm":second_encoder_grad,"learned_context_sensitive":context_sensitive,"bounds":bounds}
    return rows

def resolve_protocol(check_outputs=True,env=None,config=None,manifest=None,registry=None):
    env=deepcopy(env if env is not None else load_yaml(ENV));config=deepcopy(config if config is not None else load_config(METHOD));base=load_config(BASE);manifest=deepcopy(manifest if manifest is not None else json.loads(MANIFEST.read_text(encoding="utf-8")))
    if config.get("development_method")!="mission_aware_film":raise RuntimeError("development_method mismatch")
    expected_film={"enabled":True,"mode":"bounded_augmented_film","encoder_hidden_dim":32,"alpha":.2,"augmented_residual":True,"identity_init":True}
    expected_wave={"enabled":True,"context_target":"actor_only","encoding":"mission_markov","max_waves":3}
    if config["modules"].get("mission_film")!=expected_film or config["modules"].get("wave_context")!=expected_wave:raise RuntimeError("mission conditioning config mismatch")
    enabled=sorted(k for k,v in config["modules"].items() if v.get("enabled",False))
    if enabled!=["actor_lr_decay","mission_film","wave_context"]:raise RuntimeError(f"enabled modules mismatch: {enabled}")
    expected_training={"actor_learning_rate":3e-4,"critic_learning_rate":3e-4,"gamma":.999,"gae_lambda":.95,"clip_ratio":.2,
      "entropy_coefficient":.01,"value_loss_coefficient":.5,"max_grad_norm":.5,"rollout_steps":256,"ppo_epochs":10,
      "minibatch_size":512,"num_train_envs":24,"total_sampled_steps":3_000_000,"evaluation_episodes":50,
      "evaluation_interval_sampled_steps":100_000,"device":"cuda"}
    bad={k:(config["training"].get(k),v) for k,v in expected_training.items() if config["training"].get(k)!=v}
    if bad:raise RuntimeError(f"training protocol mismatch: {bad}")
    decay=config["modules"]["actor_lr_decay"]
    if decay!={"enabled":True,"schedule":"delayed_linear","start_step":600000,"end_step":900000,"start_lr":.0003,"end_lr":.0001}:raise RuntimeError("actor LR decay mismatch")
    if (env["environment_variant"],env["persistent_waves"]["total_waves"],env["simulation"]["max_steps"],env["scenario"]["team_size"])!=("persistent_wave_v2",3,3000,4):raise RuntimeError("environment contract mismatch")
    trainer=build_modular_mappo_trainer(config,"cpu",256,3_000_000);plain=build_modular_mappo_trainer(base,"cpu",256,3_000_000)
    effective=trainer.curriculum.runtime_config(env,0);contract=validate_runtime_environment_contract(env,effective,env,curriculum_enabled=False)
    if len({contract[k]["config_sha256"] for k in ("declared","effective_training","evaluation")})!=1:raise RuntimeError("environment hashes differ")
    frozen=manifest["frozen_environment_contract"];actual={"environment_config_sha256":config_sha256(env),"action_config_sha256":config_sha256(env["action"]),"weapon_config_sha256":config_sha256(env["weapon"]),"reward_config_sha256":config_sha256(env["reward"]),"blue_policy_config_sha256":config_sha256(env["blue_policy"]),"blue_policy_source_sha256":hashlib.sha256((ROOT/"env/fixed_policy.py").read_bytes()).hexdigest()}
    if any(frozen.get(k)!=v for k,v in actual.items()):raise RuntimeError("frozen environment fingerprint mismatch")
    arch=checkpoint_architecture(trainer);base_arch=checkpoint_architecture(plain)
    required={"actor_input_dim":52,"actor_context_dim":5,"critic_context_dim":0,"mission_film_enabled":True,"mission_film_mode":"bounded_augmented_film","mission_encoder_hidden_dim":32,"mission_film_alpha":.2,"mission_film_identity_init":True,"mission_film_augmented_residual":True,"actor_parameter_count":107494,"critic_parameter_count":474113}
    if any(arch.get(k)!=v for k,v in required.items()):raise RuntimeError(f"architecture mismatch: {arch}")
    if arch["actor_parameter_count"]-base_arch["actor_parameter_count"]!=26592 or base_arch["actor_parameter_count"]!=80902:raise RuntimeError("parameter-count mismatch")
    if manifest["training_seeds"]!=list(SEEDS) or manifest["development_evaluation"]["seed_start"]!=44_000_000 or manifest["development_evaluation"]["seed_end"]!=44_000_049:raise RuntimeError("seed protocol mismatch")
    registry=deepcopy(registry if registry is not None else json.loads(REGISTRY.read_text(encoding="utf-8")));future=registry["evaluation_ranges"]["45000000..45000199"]
    if future.get("executed") is not False or manifest["future_final"].get("executed") is not False:raise RuntimeError("future-final is not untouched")
    existing=[r["output_dir"] for r in manifest["runs"] if (ROOT/r["output_dir"]).exists()] if check_outputs else []
    if existing:raise RuntimeError(f"formal outputs are not fresh: {existing}")
    return {"status":"READY_FOR_MISSION_AWARE_FILM_DEVELOPMENT","development_method":"mission_aware_film","enabled_modules":enabled,
      "environment_contract":contract,"architecture":arch,"baseline_architecture":base_arch,"mission_only_parameter_count":26592,
      "training":expected_training,"training_seeds":list(SEEDS),"evaluation_seed_range":[44_000_000,44_000_049],
      "future_final":{"range":[45_000_000,45_000_199],"executed":False},"formal_outputs_fresh":True,
      "initialization_audit":initialization_audit(base,config)}

def cuda_smoke(protocol):
    if not torch.cuda.is_available():raise RuntimeError("CUDA required for smoke; CPU fallback forbidden")
    if not SMOKE.exists():
        cfg=load_config(METHOD);cfg["training"].update({"seed":88_300_003,"num_train_envs":1,"rollout_steps":2,"ppo_epochs":1,"minibatch_size":8,"total_sampled_steps":8,"evaluation_episodes":1,"evaluation_interval_sampled_steps":10_000_000})
        cfg["implementation"]["evaluation_seed_base"]=88_301_002
        ModularMAPPOTrainingRunner(load_yaml(ENV),cfg,1,8,"cuda",88_300_003,SMOKE,True).run()
    state=torch.load(SMOKE/"latest.pt",map_location="cuda",weights_only=False);extra=state["extra"]
    validate_modular_checkpoint(state,load_yaml(ENV),extra["algorithm_config"])
    required=("mission_feature_norm","film_delta_gamma_abs_mean","film_beta_abs_mean","film_residual_abs_mean","film_saturation_fraction")
    opt=[json.loads(x) for x in (SMOKE/"optimization_metrics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    finite=all(torch.isfinite(v).all() for group in (state["actor"],state["critic"]) for v in group.values())
    expected={"development_method":"mission_aware_film","actor_input_dim":52,"actor_context_dim":5,"critic_context_dim":0,"mission_film_enabled":True,"mission_film_mode":"bounded_augmented_film","mission_encoder_hidden_dim":32,"mission_film_alpha":.2,"runtime_total_waves":3,"runtime_max_steps":3000}
    bad={k:(extra.get(k),v) for k,v in expected.items() if extra.get(k)!=v}
    trainer=build_modular_mappo_trainer(extra["algorithm_config"],"cuda",64,8);trainer.load(SMOKE/"latest.pt")
    context=mission_context_numpy(trainer,np.asarray([1]),np.asarray([3]),np.ones((1,4),np.float32),np.asarray([0]),3000)
    checkpoint_metrics=extra.get("last_optimization_metrics",{})
    if bad or not np.all(np.isfinite(context)) or not opt or any(k not in opt[-1] or not np.isfinite(opt[-1][k]) for k in required) or any(k not in checkpoint_metrics for k in required) or not finite:raise RuntimeError(f"CUDA smoke finite/provenance/diagnostic failure: {bad}")
    return {"sampled_steps":state["sampled_steps"],"training_seed":88_300_003,"evaluation_seed":88_301_002,"checkpoint_finite":True,"film_diagnostics_finite":True,"checkpoint_film_diagnostics_present":True,"evaluation_rows":sum(1 for _ in open(SMOKE/"evaluation_history.csv",encoding="utf-8"))-1,"checkpoint":str(SMOKE/"latest.pt")}

def summary_line(result):
    a=result["architecture"]
    return f"[PREFLIGHT] READY | method=mission_aware_film | waves=3 | max_steps=3000 | actor=52 | actor_ctx=5 | critic_ctx=0 | mission_film=bounded_augmented_film/32D/a0.2 | params={a['actor_parameter_count']}/{a['critic_parameter_count']} | seeds=5301,5302,5303 | eval=44000000..44000049 | future_final=UNUSED"
def write_audit(result):
    AUDIT.mkdir(parents=True,exist_ok=True);(AUDIT/"preflight.json").write_text(json.dumps(result,indent=2),encoding="utf-8");(AUDIT/"preflight.md").write_text("# Mission-Aware FiLM preflight\n\n```json\n"+json.dumps(result,indent=2)+"\n```\n",encoding="utf-8")
def main():
    p=argparse.ArgumentParser();p.add_argument("--cuda-smoke",action="store_true");g=p.add_mutually_exclusive_group();g.add_argument("--summary",action="store_true");g.add_argument("--verbose",action="store_true");args=p.parse_args()
    result=resolve_protocol();result["cuda_smoke"]=cuda_smoke(result) if args.cuda_smoke else "NOT_REQUESTED";write_audit(result);print(json.dumps(result,indent=2) if args.verbose else summary_line(result))
if __name__=="__main__":main()
