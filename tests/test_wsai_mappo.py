from copy import deepcopy
from pathlib import Path
import numpy as np
import torch

from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer
from algorithm.modular_mappo.protocol import validate_wsai_branch
from algorithm.modules.wave_specific_actor_isolation import EXPECTED_WSAI_CONFIG


ROOT=Path(__file__).resolve().parents[1]


def make_trainer(device="cpu",epochs=1,minibatch=8):
    config=load_config(ROOT/"configs/dev_wsai_300k.yaml")
    return ModularMAPPOTrainer(device=device,seed=7,modules_config=config["modules"],
        total_sampled_steps=1_805_280,ppo_epochs=epochs,minibatch_size=minibatch)


def tensor_hash(state):
    return [(name,value.detach().cpu().clone()) for name,value in state.items()]


def assert_same(left,right):
    assert [name for name,_ in left]==[name for name,_ in right]
    assert all(torch.equal(a,b) for (_,a),(_,b) in zip(left,right))


def test_wsai_config_and_explicit_routing():
    trainer=make_trainer();assert trainer.wave_specific_actor_isolation.config==EXPECTED_WSAI_CONFIG
    obs=torch.randn(3,4,52);alive=torch.ones(3,4);waves=torch.tensor([1,2,3])
    with torch.no_grad():
        trainer.wave2_actor.mean.bias.add_(1.0);trainer.wave3_actor.mean.bias.sub_(1.0)
        routed=trainer._routed_actor_distribution(obs,alive,waves)
        individual=[actor.distribution_step(obs,None,None,None,alive)[0] for actor in trainer._wsai_actors()]
    for index in range(3):assert torch.equal(routed.loc[index],individual[index].loc[index])


def test_wsai_global_objective_routes_gradients_and_skips_absent_actor():
    trainer=make_trainer();shape=(2,4);obs=torch.randn(*shape,4,52);alive=torch.ones(*shape,4)
    raw=torch.randn(*shape,4,3);act=torch.tanh(raw);waves=torch.tensor([[1,1,2,2],[1,2,2,2]])
    with torch.no_grad():
        flat_obs=obs.reshape(-1,4,52);flat_alive=alive.reshape(-1,4);flat_waves=waves.reshape(-1)
        dist=trainer._routed_actor_distribution(flat_obs,flat_alive,flat_waves)
        oldlog=trainer.actor._squashed_log_prob(dist,raw.reshape(-1,4,3),act.reshape(-1,4,3)).reshape(*shape,4)
        oldvalue=trainer.critic.forward_step(flat_obs,flat_alive,None,None,None)[0].reshape(*shape,4)
    before=[tensor_hash(actor.state_dict()) for actor in trainer._wsai_actors()]
    metrics=trainer._update_flat_wsai(obs,act,raw,oldlog,alive,torch.randn(*shape,4),oldvalue,
        oldvalue+torch.randn(*shape,4),torch.zeros(*shape,0),waves)
    after=[tensor_hash(actor.state_dict()) for actor in trainer._wsai_actors()]
    assert any(not torch.equal(a,b) for (_,a),(_,b) in zip(before[0],after[0]))
    assert any(not torch.equal(a,b) for (_,a),(_,b) in zip(before[1],after[1]))
    assert_same(before[2],after[2])
    assert metrics["wsai_wave3_optimizer_steps"]==0
    assert np.isfinite(list(metrics.values())).all()


def test_wsai_plain_branch_clones_actor_and_optimizer_then_strict_resume(tmp_path):
    source=ROOT/"outputs/diag_mappo_learnability/l3_seed5301/checkpoint_1505280.pt"
    if not source.is_file():return
    trainer=make_trainer();trainer.load(source,strict_protocol=False,restore_rng=False)
    hashes=[tensor_hash(actor.state_dict()) for actor in trainer._wsai_actors()]
    assert_same(hashes[0],hashes[1]);assert_same(hashes[0],hashes[2])
    assert trainer.actor_optimizer.state_dict()["state"].keys()==trainer.wave2_actor_optimizer.state_dict()["state"].keys()
    with torch.no_grad():trainer.wave2_actor.mean.bias.add_(.25)
    path=tmp_path/"wsai.pt";trainer.save(path)
    restored=make_trainer();restored.load(path,strict_protocol=True,restore_rng=False)
    assert_same(tensor_hash(trainer.wave2_actor.state_dict()),tensor_hash(restored.wave2_actor.state_dict()))
    assert not all(torch.equal(a,b) for (_,a),(_,b) in zip(tensor_hash(restored.actor.state_dict()),tensor_hash(restored.wave2_actor.state_dict())))


def test_wsai_protocol_validation():
    source=ROOT/"outputs/diag_mappo_learnability/l3_seed5301/checkpoint_1505280.pt"
    if not source.is_file():return
    config=load_config(ROOT/"configs/dev_wsai_300k.yaml")
    env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml")
    state=torch.load(source,map_location="cpu",weights_only=False)
    result=validate_wsai_branch(state,env,config,{"training_seed":5301,"training_num_envs":24,"training_smoke":False})
    assert result["intervention"]=="wave_specific_actor_isolation"
