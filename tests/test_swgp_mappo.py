from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithm.common.protocol import config_sha256
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer
from algorithm.modules import (SequentialWaveGradientProjectionModule, gradient_dot,
    natural_wave_fractions, ordered_upstream_pairwise, project_nonconflicting,
    weighted_gradient_sum)
from algorithm.train_modular_mappo import load_config
from tools.audit_wave_gradient_conflict import state_sha256


DECAY={"enabled":True,"schedule":"delayed_linear","start_step":600000,"end_step":900000,"start_lr":3e-4,"end_lr":1e-4}
SWGP={"enabled":True,"mode":"ordered_upstream_pairwise","epsilon":1e-12}


def gradient(x, y): return [torch.tensor([float(x), float(y)])]


def synthetic_rollout(trainer, waves=None):
    rng=np.random.default_rng(41);t,e,a=4,2,4
    obs=rng.normal(size=(t,e,a,52)).astype("f");alive=np.ones((t,e,a),"f")
    raw=rng.normal(size=(t,e,a,3)).astype("f");actions=np.tanh(raw).astype("f")
    with torch.no_grad(): old=trainer.actor._squashed_log_prob(trainer.actor.distribution(torch.tensor(obs)),torch.tensor(raw),torch.tensor(actions)).numpy()
    rewards=rng.normal(size=(t,e,a)).astype("f");ctx=np.zeros((t,e,0),"f")
    wave=np.ones((t,e),dtype=np.int64) if waves is None else np.asarray(waves,dtype=np.int64)
    return ModularRolloutBatch(obs,actions,raw,old,rewards,rewards.copy(),np.zeros((t,e),"f"),alive,obs+.01,alive.copy(),wave,np.full((t,e),3),ctx,ctx,episode_masks=np.ones((t,e),"f"))


def trainers(seed=7):
    plain=ModularMAPPOTrainer(hidden_dim=16,ppo_epochs=1,minibatch_size=8,seed=seed,modules_config={"actor_lr_decay":deepcopy(DECAY)})
    swgp=ModularMAPPOTrainer(hidden_dim=16,ppo_epochs=1,minibatch_size=8,seed=seed,modules_config={"actor_lr_decay":deepcopy(DECAY),"sequential_wave_gradient_projection":deepcopy(SWGP)})
    return plain,swgp


def test_01_single_reference_conflict_projection():
    result,dot,applied=project_nonconflicting(gradient(-1,1),gradient(1,0));assert applied and dot==-1 and gradient_dot(result,gradient(1,0)).item()==pytest.approx(0,abs=1e-7)


def test_02_no_conflict_unchanged_exact():
    source=gradient(1,2);result,_,applied=project_nonconflicting(source,gradient(1,0));assert not applied and torch.equal(result[0],source[0])


def test_03_zero_reference_unchanged():
    source=gradient(-1,2);result,_,applied=project_nonconflicting(source,gradient(0,0));assert not applied and torch.equal(result[0],source[0])


def test_04_w2_protects_w1():
    projected,diag=ordered_upstream_pairwise({1:gradient(1,0),2:gradient(-1,1)});assert diag["w2_applied"] and gradient_dot(projected[2],projected[1])>=0


def test_05_w3_protects_w1():
    projected,diag=ordered_upstream_pairwise({1:gradient(1,0),3:gradient(-1,-1)});assert diag["w3_w1_applied"] and gradient_dot(projected[3],projected[1])>=-1e-7


def test_06_w3_protects_protected_w2():
    projected,diag=ordered_upstream_pairwise({1:gradient(1,0),2:gradient(-1,1),3:gradient(-1,-1)});assert diag["w3_w2_applied"] and gradient_dot(projected[3],projected[2])>=-1e-7


def test_07_second_w3_projection_does_not_recreate_w1_conflict():
    projected,_=ordered_upstream_pairwise({1:gradient(1,0),2:gradient(-1,1),3:gradient(-1,-1)});assert gradient_dot(projected[3],projected[1])>=-1e-7


def test_08_missing_w1_behavior():
    g2=gradient(1,0);projected,_=ordered_upstream_pairwise({2:g2,3:gradient(-1,1)});assert torch.equal(projected[2][0],g2[0]) and gradient_dot(projected[3],projected[2])>=-1e-7


def test_09_single_wave_fallback():
    g3=gradient(-3,4);projected,diag=ordered_upstream_pairwise({3:g3});assert torch.equal(projected[3][0],g3[0]) and not any(diag[key] for key in ("w2_applied","w3_w1_applied","w3_w2_applied"))


def test_10_natural_alpha_calculation():
    assert natural_wave_fractions({1:8,2:4,3:4})=={1:.5,2:.25,3:.25}


def test_11_no_equal_wave_weighting():
    alpha=natural_wave_fractions({1:8,2:4,3:4});assert alpha[1]!=pytest.approx(1/3) and alpha[2]!=pytest.approx(1/3)


def test_12_no_projection_plain_surrogate_equivalence():
    gradients={1:gradient(1,0),2:gradient(0,2),3:gradient(1,1)};weights=natural_wave_fractions({1:8,2:4,3:4})
    projected,_=ordered_upstream_pairwise(gradients);combined=weighted_gradient_sum(projected,weights,gradients[1])[0]
    expected=.5*gradients[1][0]+.25*gradients[2][0]+.25*gradients[3][0];assert torch.equal(combined,expected)


def test_13_single_wave_plain_update_equivalence():
    plain,swgp=trainers();batch=synthetic_rollout(plain);torch_state=torch.get_rng_state();plain_metrics=plain.update(deepcopy(batch));torch.set_rng_state(torch_state);swgp_metrics=swgp.update(deepcopy(batch))
    max_error=max(float((a-b).detach().abs().max()) for a,b in zip(plain.actor.parameters(),swgp.actor.parameters()))
    assert max_error<=1e-8 and swgp_metrics["swgp_single_wave_minibatch_fraction"]==1 and swgp_metrics["swgp_no_projection_fraction"]==1
    assert plain_metrics["actor_loss"]==pytest.approx(swgp_metrics["actor_loss"],abs=1e-8)


def test_14_critic_parity():
    plain,swgp=trainers();batch=synthetic_rollout(plain);state=torch.get_rng_state();plain.update(deepcopy(batch));torch.set_rng_state(state);swgp.update(deepcopy(batch))
    assert all(torch.equal(a,b) for a,b in zip(plain.critic.parameters(),swgp.critic.parameters()))
    assert state_sha256(plain.critic_optimizer.state_dict())==state_sha256(swgp.critic_optimizer.state_dict())


def test_15_entropy_is_global_and_unprojected():
    plain,swgp=trainers();batch=synthetic_rollout(plain);state=torch.get_rng_state();p=plain.update(deepcopy(batch));torch.set_rng_state(state);s=swgp.update(deepcopy(batch))
    assert p["entropy"]==pytest.approx(s["entropy"],abs=0,rel=0) and s["swgp_entropy_grad_norm"]>0


def test_16_actor_lr_decay_unchanged():
    _,swgp=trainers();swgp.sampled_steps=750000;metrics=swgp.update(synthetic_rollout(swgp));assert metrics["actor_learning_rate"]==pytest.approx(2e-4) and metrics["critic_learning_rate"]==pytest.approx(3e-4)


def test_17_actor_update_count_one_per_minibatch():
    _,swgp=trainers();before=swgp.actor_update_count;metrics=swgp.update(synthetic_rollout(swgp));assert swgp.actor_update_count-before==1 and metrics["actor_optimizer_steps_this_update"]==1


def test_18_checkpoint_roundtrip(tmp_path):
    _,swgp=trainers();swgp.update(synthetic_rollout(swgp));path=tmp_path/"swgp.pt";swgp.save(path);restored=trainers()[1];restored.load(path)
    assert all(torch.equal(a,b) for a,b in zip(swgp.actor.parameters(),restored.actor.parameters())) and all(torch.equal(a,b) for a,b in zip(swgp.critic.parameters(),restored.critic.parameters()))


def test_19_swgp_counter_roundtrip(tmp_path):
    _,swgp=trainers();swgp.update(synthetic_rollout(swgp));path=tmp_path/"swgp.pt";swgp.save(path);restored=trainers()[1];restored.load(path)
    assert restored.sequential_wave_gradient_projection.state_dict()==swgp.sequential_wave_gradient_projection.state_dict()


def test_20_branch_from_plain_counters_zero(tmp_path):
    plain,swgp=trainers();path=tmp_path/"plain.pt";plain.save(path);swgp.load(path,strict_protocol=False,restore_rng=True)
    state=swgp.sequential_wave_gradient_projection.state_dict();assert state["total_swgp_minibatches"]==state["w2_projection_count"]==state["w3_w1_projection_count"]==0


def test_21_source_actor_exact_restore(tmp_path):
    plain,swgp=trainers();path=tmp_path/"plain.pt";plain.save(path);swgp.load(path,strict_protocol=False);assert state_sha256(plain.actor.state_dict())==state_sha256(swgp.actor.state_dict())


def test_22_source_critic_exact_restore(tmp_path):
    plain,swgp=trainers();path=tmp_path/"plain.pt";plain.save(path);swgp.load(path,strict_protocol=False);assert state_sha256(plain.critic.state_dict())==state_sha256(swgp.critic.state_dict())


def test_23_source_optimizers_exact_restore(tmp_path):
    plain,swgp=trainers();plain.sampled_steps=1_505_280;plain.actor_lr_decay.apply(plain.actor_optimizer,plain.sampled_steps,plain.base_actor_learning_rate);path=tmp_path/"plain.pt";plain.save(path);swgp.load(path,strict_protocol=False)
    assert state_sha256(plain.actor_optimizer.state_dict())==state_sha256(swgp.actor_optimizer.state_dict()) and state_sha256(plain.critic_optimizer.state_dict())==state_sha256(swgp.critic_optimizer.state_dict())


def test_24_source_rng_restore(tmp_path):
    plain,swgp=trainers();path=tmp_path/"plain.pt";plain.save(path);expected=plain.checkpoint_state()["rng_state"]["trainer_permutation_rng_state"];swgp.load(path,strict_protocol=False,restore_rng=True);assert swgp.rng.bit_generator.state==expected


def test_25_branch_source_step_preserved(tmp_path):
    plain,swgp=trainers();plain.sampled_steps=1_505_280;plain.actor_update_count=12;plain.critic_update_count=13;plain.ppo_update_count=2;path=tmp_path/"plain.pt";plain.save(path);swgp.load(path,strict_protocol=False)
    assert (swgp.sampled_steps,swgp.actor_update_count,swgp.critic_update_count,swgp.ppo_update_count)==(1_505_280,12,13,2)


def test_26_incompatible_modules_rejected():
    with pytest.raises(ValueError,match="exact enabled modules"):
        ModularMAPPOTrainer(hidden_dim=16,modules_config={"actor_lr_decay":deepcopy(DECAY),"sequential_wave_gradient_projection":deepcopy(SWGP),"wave_balancing":{"enabled":True}})


def test_27_no_environment_reward_observation_changes():
    base=load_config("configs/diag_mappo_learnability_common_3m.yaml");control=load_config("configs/dev_swgp_branch_control_3m.yaml");swgp=load_config("configs/dev_swgp_branch_3m.yaml")
    env=load_config("configs/persistent_wave_v2_environment.yaml")
    for cfg in (control,swgp):
        assert cfg["network"]==base["network"] and cfg["training"]==base["training"]
    assert env["environment_variant"]=="persistent_wave_v2" and config_sha256(env["reward"])==config_sha256(load_config("configs/persistent_wave_v2_environment.yaml")["reward"])


def test_28_mixed_wave_metrics_are_finite_and_projection_counters_match():
    _,swgp=trainers();waves=np.asarray([[1,2],[3,1],[2,3],[1,3]]);metrics=swgp.update(synthetic_rollout(swgp,waves))
    assert np.isfinite(list(metrics.values())).all() and metrics["swgp_active"]==1
    assert swgp.sequential_wave_gradient_projection.total_swgp_minibatches==1


def test_29_module_state_requires_version_and_config():
    module=SequentialWaveGradientProjectionModule(SWGP);state=module.state_dict();state["version"]=99
    with pytest.raises(RuntimeError,match="version"):module.load_state_dict(state)


def test_30_projection_removes_float32_residual_conflict():
    # A near-collinear, high-dimensional case exercises the same cancellation
    # regime as the first real branch update that originally tripped the guard.
    generator=torch.Generator().manual_seed(5302)
    reference=[torch.randn(80902,generator=generator,dtype=torch.float32)]
    orthogonal=torch.randn(80902,generator=generator,dtype=torch.float32)
    orthogonal-=gradient_dot([orthogonal],reference)/gradient_dot(reference,reference)*reference[0]
    source=[-2.75*reference[0]+1e-5*orthogonal]
    projected,_,applied=project_nonconflicting(source,reference)
    assert applied
    dot=float(gradient_dot(projected,reference))
    tolerance=-1e-7*float(torch.linalg.vector_norm(projected[0]))*float(torch.linalg.vector_norm(reference[0]))
    assert dot>=tolerance
    assert all(torch.isfinite(value).all() for value in projected)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_31_projection_residual_guard_on_cuda():
    generator=torch.Generator().manual_seed(5302)
    reference=[torch.randn(80902,generator=generator,dtype=torch.float32).cuda()]
    orthogonal=torch.randn(80902,generator=generator,dtype=torch.float32).cuda()
    orthogonal-=gradient_dot([orthogonal],reference)/gradient_dot(reference,reference)*reference[0]
    source=[-2.75*reference[0]+1e-5*orthogonal]
    projected,_,applied=project_nonconflicting(source,reference)
    dot=float(gradient_dot(projected,reference))
    tolerance=-1e-7*float(torch.linalg.vector_norm(projected[0]))*float(torch.linalg.vector_norm(reference[0]))
    assert applied and dot>=tolerance
