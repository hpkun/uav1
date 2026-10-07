from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer
from algorithm.modules import (SequentialWaveGradientProjectionModule,gradient_dot,
    natural_wave_fractions,ordered_upstream_pairwise,project_nonconflicting)
from algorithm.train_modular_mappo import load_config
from tools.audit_wave_gradient_conflict import state_sha256
from tools.preflight_delayed_swgp_mappo import validate_delayed_config

DECAY={"enabled":True,"schedule":"delayed_linear","start_step":600000,"end_step":900000,
       "start_lr":3e-4,"end_lr":1e-4}
DELAYED={"enabled":True,"mode":"ordered_upstream_pairwise","epsilon":1e-12,
         "activation_start_step":1_500_000}


def make_trainers(seed=5302):
    common=dict(hidden_dim=16,ppo_epochs=1,minibatch_size=32,seed=seed,total_sampled_steps=3_000_000)
    plain=ModularMAPPOTrainer(**common,modules_config={"actor_lr_decay":deepcopy(DECAY)})
    delayed=ModularMAPPOTrainer(**common,modules_config={"actor_lr_decay":deepcopy(DECAY),
        "sequential_wave_gradient_projection":deepcopy(DELAYED)})
    return plain,delayed


def rollout(trainer,waves=None):
    rng=np.random.default_rng(904);t,e,a=4,2,4
    obs=rng.normal(size=(t,e,a,52)).astype("f");alive=np.ones((t,e,a),"f")
    raw=rng.normal(size=(t,e,a,3)).astype("f");actions=np.tanh(raw).astype("f")
    with torch.no_grad():
        old=trainer.actor._squashed_log_prob(trainer.actor.distribution(torch.tensor(obs)),
            torch.tensor(raw),torch.tensor(actions)).numpy()
    rewards=rng.normal(size=(t,e,a)).astype("f");ctx=np.zeros((t,e,0),"f")
    wave=np.ones((t,e),dtype=np.int64) if waves is None else np.asarray(waves,dtype=np.int64)
    return ModularRolloutBatch(obs,actions,raw,old,rewards,rewards.copy(),np.zeros((t,e),"f"),alive,
        obs+.01,alive.copy(),wave,np.full((t,e),3),ctx,ctx,episode_masks=np.ones((t,e),"f"))


def fingerprint(trainer):
    return (state_sha256(trainer.actor.state_dict()),state_sha256(trainer.critic.state_dict()),
        state_sha256(trainer.actor_optimizer.state_dict()),state_sha256(trainer.critic_optimizer.state_dict()),
        trainer.actor_update_count,trainer.critic_update_count,trainer.ppo_update_count,
        deepcopy(trainer.rng.bit_generator.state))


def run_paired_before_activation(step):
    plain,delayed=make_trainers();plain.sampled_steps=delayed.sampled_steps=step
    batch=rollout(plain,[[1,2],[3,1],[2,3],[1,3]])
    initial=torch.get_rng_state();plain_metrics=plain.update(deepcopy(batch));plain_rng=torch.get_rng_state()
    torch.set_rng_state(initial);delayed_metrics=delayed.update(deepcopy(batch));delayed_rng=torch.get_rng_state()
    return plain,delayed,plain_metrics,delayed_metrics,plain_rng,delayed_rng


def test_01_activation_config_parse():
    cfg=load_config("configs/dev_delayed_swgp_3m.yaml")
    assert cfg["modules"]["sequential_wave_gradient_projection"]["activation_start_step"]==1_500_000


def test_02_activation_start_step_validation():
    for invalid in (-1,0):
        with pytest.raises(ValueError,match="positive"):
            SequentialWaveGradientProjectionModule({**DELAYED,"activation_start_step":invalid})
    # Omission retains the original full-horizon SWGP V1 behaviour.
    assert SequentialWaveGradientProjectionModule({"enabled":True}).activation_start_step==0


def test_03_is_active_boundary():
    module=SequentialWaveGradientProjectionModule(DELAYED)
    assert not module.is_active(1_499_999)
    assert module.is_active(1_500_000) and module.is_active(1_505_280)


@pytest.mark.parametrize("step",[0,900_000,1_499_136])
def test_04_preactivation_plain_bitwise_parity(step):
    plain,delayed,pm,dm,plain_rng,delayed_rng=run_paired_before_activation(step)
    assert fingerprint(plain)==fingerprint(delayed)
    assert torch.equal(plain_rng,delayed_rng)
    for key in ("actor_loss","value_loss","entropy","approx_kl"):
        assert pm[key]==dm[key]
    assert dm["swgp_currently_active"]==0 and dm["swgp_total_minibatches"]==0


def test_05_activation_boundary_and_first_step_once():
    _,trainer=make_trainers();batch=rollout(trainer,[[1,2],[3,1],[2,3],[1,3]])
    trainer.sampled_steps=1_499_136;before=trainer.update(deepcopy(batch))
    assert before["swgp_currently_active"]==0
    trainer.sampled_steps=1_505_280;first=trainer.update(deepcopy(batch))
    assert first["swgp_currently_active"]==first["swgp_activated_this_update"]==1
    assert trainer.sequential_wave_gradient_projection.first_activation_sampled_steps==1_505_280
    trainer.sampled_steps=1_511_424;second=trainer.update(deepcopy(batch))
    assert second["swgp_activated_this_update"]==0
    assert trainer.sequential_wave_gradient_projection.first_activation_sampled_steps==1_505_280
    assert trainer.sequential_wave_gradient_projection.pre_activation_plain_update_count==1
    assert trainer.sequential_wave_gradient_projection.activation_update_count==2


def test_06_postactivation_mixed_wave_path_and_natural_alpha():
    _,trainer=make_trainers();trainer.sampled_steps=1_505_280
    metrics=trainer.update(rollout(trainer,[[1,2],[3,1],[2,3],[1,3]]))
    assert metrics["swgp_currently_active"]==metrics["swgp_active"]==1
    assert metrics["swgp_total_minibatches"]==1
    assert (metrics["swgp_wave1_alive_fraction"],metrics["swgp_wave2_alive_fraction"],
            metrics["swgp_wave3_alive_fraction"])==pytest.approx((3/8,2/8,3/8))
    assert natural_wave_fractions({1:12,2:8,3:12})=={1:3/8,2:2/8,3:3/8}


def test_07_existing_ordered_projection_unchanged():
    g=lambda x,y:[torch.tensor([float(x),float(y)])]
    projected,diag=ordered_upstream_pairwise({1:g(1,0),2:g(-1,1),3:g(-1,-1)})
    assert diag["w2_applied"] and diag["w3_w1_applied"] and diag["w3_w2_applied"]
    assert gradient_dot(projected[2],projected[1])>=-1e-7
    assert gradient_dot(projected[3],projected[1])>=-1e-7
    assert gradient_dot(projected[3],projected[2])>=-1e-7


def test_08_postactivation_critic_and_entropy_remain_plain():
    plain,delayed=make_trainers();plain.sampled_steps=delayed.sampled_steps=1_505_280
    batch=rollout(plain,[[1,2],[3,1],[2,3],[1,3]])
    initial=torch.get_rng_state();pm=plain.update(deepcopy(batch));torch.set_rng_state(initial);dm=delayed.update(deepcopy(batch))
    assert state_sha256(plain.critic.state_dict())==state_sha256(delayed.critic.state_dict())
    assert state_sha256(plain.critic_optimizer.state_dict())==state_sha256(delayed.critic_optimizer.state_dict())
    assert pm["entropy"]==dm["entropy"]


def test_09_float32_residual_halfspace_fix_retained():
    generator=torch.Generator().manual_seed(5302);reference=[torch.randn(80902,generator=generator)]
    orthogonal=torch.randn(80902,generator=generator)
    orthogonal-=gradient_dot([orthogonal],reference)/gradient_dot(reference,reference)*reference[0]
    projected,_,applied=project_nonconflicting([-2.75*reference[0]+1e-5*orthogonal],reference)
    tolerance=-1e-7*float(torch.linalg.vector_norm(projected[0]))*float(torch.linalg.vector_norm(reference[0]))
    assert applied and float(gradient_dot(projected,reference))>=tolerance


def test_10_checkpoint_before_activation_resume(tmp_path):
    _,trainer=make_trainers();trainer.sampled_steps=1_499_136;trainer.update(rollout(trainer))
    path=tmp_path/"before.pt";trainer.save(path);restored=make_trainers()[1];restored.load(path)
    assert not restored.sequential_wave_gradient_projection.is_active(restored.sampled_steps)
    assert restored.sequential_wave_gradient_projection.first_activation_sampled_steps is None
    assert restored.sequential_wave_gradient_projection.pre_activation_plain_update_count==1


def test_11_checkpoint_after_activation_resume_and_counters(tmp_path):
    _,trainer=make_trainers();trainer.sampled_steps=1_505_280
    trainer.update(rollout(trainer,[[1,2],[3,1],[2,3],[1,3]]))
    path=tmp_path/"after.pt";trainer.save(path);restored=make_trainers()[1];restored.load(path)
    assert restored.sequential_wave_gradient_projection.state_dict()==trainer.sequential_wave_gradient_projection.state_dict()
    restored.sampled_steps=1_511_424;restored.update(rollout(restored))
    assert restored.sequential_wave_gradient_projection.first_activation_sampled_steps==1_505_280
    assert restored.sequential_wave_gradient_projection.activation_update_count==2


def test_12_strict_activation_config_mismatch_rejected():
    module=SequentialWaveGradientProjectionModule(DELAYED);state=module.state_dict()
    changed=SequentialWaveGradientProjectionModule({**DELAYED,"activation_start_step":1_500_001})
    with pytest.raises(RuntimeError,match="config mismatch"):changed.load_state_dict(state)


def test_13_fresh_config_identity_and_exact_modules():
    cfg=load_config("configs/dev_delayed_swgp_3m.yaml")
    enabled=sorted(k for k,v in cfg["modules"].items() if isinstance(v,dict) and v.get("enabled",False))
    assert cfg["development_method"]=="delayed_swgp_mappo"
    assert "development_branch" not in cfg
    assert enabled==["actor_lr_decay","sequential_wave_gradient_projection"]


def test_14_preflight_rejects_development_branch():
    cfg=load_config("configs/dev_delayed_swgp_3m.yaml");cfg["development_branch"]={"intervention":"bad"}
    env=load_config("configs/persistent_wave_v2_environment.yaml")
    registry=json.loads(Path("experiments/current_seed_provenance.json").read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError,match="forbids development_branch"):
        validate_delayed_config(cfg,env,registry)


def test_15_future_final_45m_remains_unused():
    registry=json.loads(Path("experiments/current_seed_provenance.json").read_text(encoding="utf-8"))
    assert registry["evaluation_ranges"]["45000000..45000199"]["executed"] is False


def test_16_preactivation_diagnostics_are_numeric_without_projection():
    _,trainer=make_trainers();trainer.sampled_steps=900_000
    metrics=trainer.update(rollout(trainer,[[1,2],[3,1],[2,3],[1,3]]))
    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert metrics["swgp_first_activation_sampled_steps"]==-1
    assert metrics["swgp_pre_activation_plain_update_count"]==1
    assert metrics["swgp_activation_update_count"]==0
