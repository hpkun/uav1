"""Independent navigation ETM: identity, TBPTT, probabilities and real CUDA loops."""
import copy
from pathlib import Path
import json
import numpy as np
import pytest
import torch
import yaml
from algorithm.etm_mappo.networks import EntityTemporalActor,CentralizedNavigationCritic,squashed_log_prob
from algorithm.etm_mappo.memory import MemoryState,identity_matches
from algorithm.etm_mappo.config import runtime_config,validate_config

ROOT=Path(__file__).resolve().parents[1]
DEVICE='cuda'


@pytest.fixture(scope='module',autouse=True)
def numerical_context():
    assert torch.cuda.is_available(),'CUDA is mandatory'
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**overrides):
    value=yaml.safe_load((ROOT/'configs/etm_mappo.yaml').read_text(encoding='utf-8'))
    return runtime_config(value,**overrides)


def observation(envs=2,agents=4,peers=3,obstacles=3,dims=(10,4,5)):
    own,peer,obstacle=dims
    return dict(self_features=torch.randn(envs,agents,own,device=DEVICE),
        peer_features=torch.randn(envs,agents,peers,peer,device=DEVICE),
        obstacle_features=torch.randn(envs,agents,obstacles,obstacle,device=DEVICE),
        peer_ids=torch.arange(peers,device=DEVICE,dtype=torch.long).expand(envs,agents,-1).clone(),
        obstacle_ids=(1000+torch.arange(obstacles,device=DEVICE,dtype=torch.long)).expand(envs,agents,-1).clone(),
        peer_valid=torch.ones(envs,agents,peers,device=DEVICE,dtype=torch.bool),
        obstacle_valid=torch.ones(envs,agents,obstacles,device=DEVICE,dtype=torch.bool),
        agent_alive=torch.ones(envs,agents,device=DEVICE,dtype=torch.bool))


def actor(variant='B3',dims=(10,4,5)):
    torch.manual_seed(17)
    n=copy.deepcopy(config()['network']);n.update(variant=variant,self_dim=dims[0],peer_dim=dims[1],obstacle_dim=dims[2])
    return EntityTemporalActor(n).to(DEVICE)


@pytest.mark.parametrize('agents,peers,obstacles',[(1,0,0),(3,2,1),(4,3,4),(7,6,2)])
def test_network_shapes_and_empty_sets(agents,peers,obstacles):
    a=actor();o=observation(2,agents,peers,obstacles)
    state=a.initial_state(o)
    d,next_state,diag=a.step(o,state,torch.ones(2,device=DEVICE,dtype=torch.bool))
    assert d.mean.shape==(2,agents,2)
    assert next_state.self_hidden.shape==(2,agents,64)
    assert next_state.peer_hidden.shape==(2,agents,peers,64)
    assert next_state.obstacle_hidden.shape==(2,agents,obstacles,64)
    assert diag['gate'].shape==(2,agents,128)
    assert torch.isfinite(d.mean).all() and torch.isfinite(d.scale).all()
    critic=CentralizedNavigationCritic(10).to(DEVICE)
    assert critic(o['self_features'],o['agent_alive']).shape==(2,agents)
    if peers==obstacles==0:assert torch.count_nonzero(diag['external_context'])==0


def test_network_configurable_features_and_type_parameter_sharing():
    a=actor(dims=(6,8,3));o=observation(dims=(6,8,3))
    d,s,_=a.step(o,a.initial_state(o),torch.zeros(2,device=DEVICE,dtype=torch.bool))
    assert d.mean.shape==(2,4,2)
    assert isinstance(a.peer_gru,torch.nn.GRUCell) and isinstance(a.obstacle_gru,torch.nn.GRUCell)
    assert a.peer_gru.weight_ih.data_ptr()!=a.obstacle_gru.weight_ih.data_ptr()
    assert sum(isinstance(m,torch.nn.GRUCell) for m in a.modules())==3


@pytest.mark.parametrize('variant',['B0','B1','B2','B3'])
def test_network_ablations_forward_backward(variant):
    a=actor(variant);o=observation();s=a.initial_state(o);start=torch.zeros(2,device=DEVICE,dtype=torch.bool)
    for _ in range(2):d,s,diag=a.step(o,s,start)
    raw=d.rsample();loss=-(squashed_log_prob(d,raw).mean())+d.mean.square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in a.parameters() if p.grad is not None)
    modules=['self_encoder','peer_encoder','obstacle_encoder','policy_hidden','mean','log_std']
    if variant in ('B2','B3'):modules+=['self_gru','peer_gru','obstacle_gru']
    if variant!='B0':modules+=['peer_attention','obstacle_attention']
    if variant=='B1':modules+=['post_gru']
    if variant=='B3':modules+=['gate','self_projection','context_projection']
    for name in modules:
        assert sum(float(p.grad.abs().sum()) for p in getattr(a,name).parameters() if p.grad is not None)>0,name
    assert (raw.tanh().abs()<=1).all()


def permute_entities(o,side,order):
    result={key:value.clone() for key,value in o.items()}
    for kind in ('features','ids','valid'):result[f'{side}_{kind}']=result[f'{side}_{kind}'][...,order,:] if kind=='features' else result[f'{side}_{kind}'][...,order]
    return result


def test_memory_follows_identity_permutation_and_id_values_are_not_features():
    a=actor();o=observation();start=torch.zeros(2,device=DEVICE,dtype=torch.bool)
    _,s,_=a.step(o,a.initial_state(o),start)
    d,next_state,_=a.step(o,s,start)
    reordered=permute_entities(o,'peer',[2,0,1]);reordered=permute_entities(reordered,'obstacle',[1,2,0])
    q,permuted,_=a.step(reordered,s,start)
    torch.testing.assert_close(d.mean,q.mean,atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(next_state.peer_hidden[..., [2,0,1],:],permuted.peer_hidden)
    renamed={key:value.clone() for key,value in o.items()};renamed['peer_ids']+=321
    altered=s.clone();altered.peer_ids+=321
    renamed_distribution,_,_=a.step(renamed,altered,start)
    torch.testing.assert_close(d.mean,renamed_distribution.mean,rtol=0,atol=0)


def test_memory_disappearance_reappearance_empty_mask_and_bias():
    a=actor();o=observation();start=torch.zeros(2,device=DEVICE,dtype=torch.bool)
    _,s,_=a.step(o,a.initial_state(o),start)
    removed={key:value.clone() for key,value in o.items()}
    for side in ('peer','obstacle'):
        removed[side+'_valid'].zero_();removed[side+'_ids'].fill_(-1)
    d,cleared,diag=a.step(removed,s,start)
    assert torch.count_nonzero(cleared.peer_hidden)==torch.count_nonzero(cleared.obstacle_hidden)==0
    assert torch.count_nonzero(diag['peer_attention_weights'])==torch.count_nonzero(diag['obstacle_attention_weights'])==0
    assert torch.count_nonzero(diag['external_context'])==0
    changed={key:value.clone() for key,value in removed.items()}
    changed['peer_features'].fill_(1e4);changed['obstacle_features'].fill_(-1e4)
    other,_,_=a.step(changed,s,start)
    torch.testing.assert_close(d.mean,other.mean,rtol=0,atol=0)
    _,reappeared,_=a.step(o,cleared,start)
    _,fresh,_=a.step(o,a.initial_state(o),start)
    torch.testing.assert_close(reappeared.peer_hidden,fresh.peer_hidden,rtol=0,atol=0)


def test_memory_environment_observer_isolation_reset_and_counterfactual():
    a=actor();o=observation();start=torch.zeros(2,device=DEVICE,dtype=torch.bool)
    _,s,_=a.step(o,a.initial_state(o),start)
    changed={key:value.clone() for key,value in o.items()};changed['peer_features'][0,0,0]+=3
    _,counter,_=a.step(changed,a.initial_state(o),start)
    assert not torch.equal(s.peer_hidden[0,0,0],counter.peer_hidden[0,0,0])
    torch.testing.assert_close(s.self_hidden,counter.self_hidden,rtol=0,atol=0)
    torch.testing.assert_close(s.peer_hidden[1],counter.peer_hidden[1],rtol=0,atol=0)
    torch.testing.assert_close(s.peer_hidden[0,1:],counter.peer_hidden[0,1:],rtol=0,atol=0)
    d,reset,_=a.step(o,s,torch.tensor([True,False],device=DEVICE))
    fresh,zero,_=a.step(o,a.initial_state(o),start)
    torch.testing.assert_close(d.mean[0],fresh.mean[0],rtol=0,atol=0)
    torch.testing.assert_close(reset.peer_hidden[0],zero.peer_hidden[0],rtol=0,atol=0)
    o['agent_alive'][0,1]=False
    _,dead,_=a.step(o,s,start)
    assert torch.count_nonzero(dead.self_hidden[0,1])==torch.count_nonzero(dead.peer_hidden[0,1])==0


@pytest.mark.parametrize('case',['duplicate','float_ids','invalid_ids','shape','previous_duplicate'])
def test_memory_rejects_inconsistent_ids(case):
    a=actor();o=observation();s=a.initial_state(o)
    if case=='duplicate':o['peer_ids'][...,1]=o['peer_ids'][...,0]
    if case=='float_ids':o['peer_ids']=o['peer_ids'].float()
    if case=='invalid_ids':o['peer_valid'][...,0]=False
    if case=='shape':o['peer_features']=o['peer_features'][...,:2,:]
    if case=='previous_duplicate':s.peer_ids.fill_(1)
    with pytest.raises(ValueError):a.step(o,s,torch.zeros(2,device=DEVICE,dtype=torch.bool))


def test_network_log_probability_jacobian_and_saturation():
    a=actor();o=observation();d,_,_=a.step(o,a.initial_state(o),torch.ones(2,device=DEVICE,dtype=torch.bool))
    raw=torch.randn_like(d.mean,dtype=torch.float64)
    torch.testing.assert_close(squashed_log_prob(d,raw),(d.log_prob(raw)-torch.log1p(-raw.tanh().square())).sum(-1))
    assert torch.isfinite(squashed_log_prob(d,raw*100)).all()


def test_network_critic_joint_mask_and_actor_locality():
    o=observation();critic=CentralizedNavigationCritic(10).to(DEVICE)
    alive=o['agent_alive'].clone();alive[:,1]=False
    before=critic(o['self_features'],alive)
    changed=o['self_features'].clone();changed[:,1]+=100
    torch.testing.assert_close(before,critic(changed,alive),rtol=0,atol=0)
    assert torch.count_nonzero(before[:,1])==0
    assert torch.count_nonzero(critic(changed,torch.zeros_like(alive)))==0


def synthetic_rollout(trainer,time=7,envs=2):
    from algorithm.etm_mappo.trainer import EntityRollout
    observations=[];states=[];actions=[];raws=[];logs=[];values=[]
    starts=np.zeros((time,envs),dtype=bool);starts[0]=True;starts[3,0]=True;starts[5,1]=True
    state=None
    for step in range(time):
        o=observation(envs)
        if step==2:
            o['peer_valid'][...,0]=False;o['peer_ids'][...,0]=-1
        if step%2:o=permute_entities(o,'peer',[2,0,1])
        array={key:value.cpu().numpy() for key,value in o.items()}
        if state is None:state=trainer.initial_state(array)
        states.append(state.numpy());observations.append(array)
        a,r,l,state=trainer.act(array,state,starts[step])
        actions.append(a);raws.append(r);logs.append(l);values.append(trainer.values(array))
    obs={key:np.stack([o[key] for o in observations]) for key in observations[0]}
    memory={key:np.stack([s[key] for s in states]) for key in states[0]}
    terminated=np.zeros_like(starts);terminated[2,0]=True
    truncated=np.zeros_like(starts);truncated[4,1]=True
    rewards=np.random.default_rng(12).normal(size=(time,envs,4)).astype('f')
    return EntityRollout(obs,copy.deepcopy(obs),np.stack(actions),np.stack(raws),np.stack(logs),rewards,
        np.stack(values),np.stack(values),terminated,truncated,starts,memory,np.ones((time,envs),dtype=bool))


def test_training_terminal_truncation_bootstrap_and_gae_formula():
    from algorithm.etm_mappo.trainer import compute_gae
    rewards=torch.tensor([[[2.]],[[100.]]],device=DEVICE)
    values=torch.ones_like(rewards);following=values*4
    alive=torch.ones_like(rewards,dtype=torch.bool)
    term=torch.tensor([[False],[False]],device=DEVICE);trunc=torch.tensor([[True],[False]],device=DEVICE)
    a,target=compute_gae(rewards,values,following,term,trunc,alive,alive,.9,.8)
    assert a[0].item()==pytest.approx(2+.9*4-1)
    assert target[0].item()==pytest.approx(2+.9*4)
    term[0]=True;trunc[0]=False
    _,target=compute_gae(rewards,values,following,term,trunc,alive,alive,.9,.8)
    assert target[0].item()==2
    term[:]=False;next_alive=alive.clone();next_alive[0]=False
    _,target=compute_gae(rewards,values,following,term,trunc,alive,next_alive,.9,.8)
    assert target[0].item()==2


def test_training_chunk_coverage_padding_and_masked_loss_invariance():
    from algorithm.etm_mappo.trainer import sequence_chunks,pack_tensor,clipped_losses
    chunks=sequence_chunks(7,2,4)
    packed=pack_tensor(torch.arange(14,device=DEVICE).reshape(7,2),chunks,4,-1)
    assert packed.tolist()==[[0,2,4,6],[8,10,12,-1],[1,3,5,7],[9,11,13,-1]]
    assert sorted(packed[packed>=0].tolist())==list(range(14))
    mask=torch.tensor([True,True,False],device=DEVICE)
    log=torch.tensor([np.log(1.5),np.log(.5),0.],device=DEVICE)
    old=torch.zeros_like(log);adv=torch.tensor([2.,-2.,0.],device=DEVICE)
    value=torch.tensor([1.,1.,0.],device=DEVICE);previous=torch.tensor([0.,2.,0.],device=DEVICE)
    target=torch.tensor([1.,0.,0.],device=DEVICE);ent=torch.ones_like(log)
    first=clipped_losses(log,old,adv,value,previous,target,ent,mask,.2)
    assert first[0].item()==pytest.approx(-.4)
    assert first[1].item()==pytest.approx(.5*(.64+3.24)/2)
    for tensor in (log,old,adv,value,previous,target,ent):tensor[-1]=float('nan')
    second=clipped_losses(log,old,adv,value,previous,target,ent,mask,.2)
    for a,b in zip(first[:3],second[:3]):torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('variant',['B0','B1','B2','B3'])
def test_training_synthetic_ratio_and_real_module_updates(variant):
    from algorithm.etm_mappo.trainer import ETMMAPPOTrainer
    cfg=config(variant=variant)
    cfg['training'].update(sequence_length=4,minibatch_joint_transitions=8,ppo_epochs=2)
    trainer=ETMMAPPOTrainer(cfg)
    rollout=synthetic_rollout(trainer)
    audit=trainer.audit_rollout_ratio(rollout)
    assert audit['pre_update_ratio_max_abs_error']<=1e-4
    before={name:copy.deepcopy(getattr(trainer,name).state_dict()) for name in ('actor','critic')}
    metrics=trainer.update(rollout)
    assert metrics['valid_joint_transitions']==14 and metrics['padded_joint_transitions']==2
    assert metrics['actual_actor_optimizer_steps']==metrics['actual_critic_optimizer_steps']==4
    assert np.isfinite(list(metrics.values())).all()
    for name in ('actor','critic'):
        assert any(not torch.equal(value,before[name][key]) for key,value in getattr(trainer,name).state_dict().items())
    corrupt=copy.deepcopy(rollout);corrupt.old_log_probs+=.01
    with pytest.raises(RuntimeError,match='ratio audit'):trainer.audit_rollout_ratio(corrupt)


def test_training_bptt_identity_path_and_midchunk_reset():
    a=actor();o=observation(1);o['peer_features'].requires_grad_()
    start=torch.zeros(1,device=DEVICE,dtype=torch.bool)
    _,state,_=a.step(o,a.initial_state(o),start)
    next_o=permute_entities(o,'peer',[2,0,1])
    next_o['peer_features']=next_o['peer_features'].detach()
    distribution,_,_=a.step(next_o,state,start)
    gradient=torch.autograd.grad(distribution.mean.sum(),o['peer_features'],retain_graph=True)[0]
    assert torch.count_nonzero(gradient)>0
    distribution,_,_=a.step(next_o,state,torch.ones_like(start))
    gradient=torch.autograd.grad(distribution.mean.sum(),o['peer_features'],allow_unused=True)[0]
    assert gradient is None or torch.count_nonzero(gradient)==0


def test_training_history_reset_keeps_network_parameters():
    a=actor();o=observation();start=torch.zeros(2,device=DEVICE,dtype=torch.bool)
    _,state,_=a.step(o,a.initial_state(o),start)
    reset,_,_=a.step(o,state,start,reset_memory=True)
    fresh,_,_=a.step(o,a.initial_state(o),start)
    torch.testing.assert_close(reset.mean,fresh.mean,rtol=0,atol=0)
    assert 'peer_gru.weight_hh' in a.state_dict() and 'gate.weight' in a.state_dict()


def test_navigation_physics_masks_and_shuffle_do_not_change_dynamics():
    from algorithm.etm_mappo.navigation import CooperativeNavigationEnv
    e=config()['environment'];second=copy.deepcopy(e);second['shuffle_slots']=False
    first=CooperativeNavigationEnv(e);other=CooperativeNavigationEnv(second)
    a=first.reset(4);b=other.reset(4)
    np.testing.assert_array_equal(first.positions,other.positions)
    for _ in range(5):
        x=first.step(np.full((4,2),.1,np.float32));y=other.step(np.full((4,2),.1,np.float32))
        np.testing.assert_array_equal(first.positions,other.positions)
        np.testing.assert_array_equal(x[1],y[1])
    assert set(first.component_totals)=={'progress','formation','collision','smoothness'}
    first.positions[1]=[50,50]
    observed=first.observe()
    assert 1 not in observed['peer_ids'][0]
    assert np.all(observed['peer_ids'][~observed['peer_valid']]==-1)
    assert observed['self_features'].shape==(4,10)
    first.positions[1]=[70,70]
    following=first.observe()
    # Moving an invisible peer cannot change the observer's own features.
    np.testing.assert_array_equal(observed['self_features'][0],following['self_features'][0])


def test_navigation_vector_terminal_frame_and_reset_memory(tmp_path):
    from algorithm.etm_mappo.runner import ETMTrainingRunner
    cfg=config(num_envs=2,total_sampled_steps=34)
    cfg['training'].update(ppo_epochs=1,rollout_steps=17,evaluation_episodes=1)
    cfg['environment']['max_steps']=5
    runner=ETMTrainingRunner(cfg,tmp_path/'collector')
    rollout=runner.collect_rollout(17)
    assert rollout.actions.shape==(17,2,4,2)
    assert rollout.truncated[4].all() and not rollout.terminated[4].any()
    assert rollout.episode_starts[5].all()
    assert not np.array_equal(rollout.next_observations['self_features'][4],rollout.observations['self_features'][5])
    for key,value in rollout.memory.items():
        if key.endswith('_ids'):assert (value[5]==-1).all()
        else:assert np.count_nonzero(value[5])==0
    assert runner.trainer.sampled_steps==34 and runner.trainer.vector_steps==17
    assert runner.trainer.audit_rollout_ratio(rollout)['pre_update_ratio_max_abs_error']<=1e-4
    metric=runner.trainer.update(rollout)
    assert metric['valid_joint_transitions']==34 and metric['padded_joint_transitions']==30


@pytest.fixture(scope='module')
def navigation_run(tmp_path_factory):
    from algorithm.etm_mappo.runner import ETMTrainingRunner
    cfg=config(seed=12,num_envs=2,total_sampled_steps=66)
    cfg['training'].update(ppo_epochs=1,rollout_steps=33,evaluation_episodes=1,evaluation_interval=66,checkpoint_interval=66)
    cfg['environment']['max_steps']=5
    output=tmp_path_factory.mktemp('etm_navigation')/'run'
    runner=ETMTrainingRunner(cfg,output)
    before={key:copy.deepcopy(getattr(runner.trainer,key).state_dict()) for key in ('actor','critic')}
    summary=runner.run()
    delta={name:float(torch.sqrt(sum((value-before[name][key]).square().sum() for key,value in getattr(runner.trainer,name).state_dict().items())))
        for name in ('actor','critic')}
    assert all(value>0 for value in delta.values())
    evidence=dict(gpu=torch.cuda.get_device_name(),sampled_steps=summary['sampled_steps'],vector_steps=summary['vector_steps'],
        parameter_counts=summary['parameter_counts'],parameter_delta_l2=delta,
        actor_optimizer_steps=summary['actor_optimizer_steps'],critic_optimizer_steps=summary['critic_optimizer_steps'],
        optimization=summary['last_optimization'])
    path=ROOT/'outputs/etm_validation/cuda_integration_evidence.json'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    return runner,output,summary


def test_navigation_real_cuda_training_artifacts(navigation_run):
    runner,output,summary=navigation_run
    assert summary['sampled_steps']==66 and summary['vector_steps']==33
    assert summary['actor_optimizer_steps']==summary['critic_optimizer_steps']==1
    for name in ('latest.pt','final.pt','best_eval.pt','checkpoint_66.pt','run_summary.json','run_config.json',
                 'evaluation_history.csv','algorithm_config.yaml','optimization_metrics.jsonl','episode_metrics.jsonl'):
        assert (output/name).is_file(),name
    metrics=summary['last_optimization']
    assert metrics['actor_gradient_norm']>0 and metrics['critic_gradient_norm']>0
    assert metrics['peer_hidden_norm']>0 and metrics['obstacle_hidden_norm']>0


def test_navigation_checkpoint_reload_deterministic_evaluation_and_reject(navigation_run,tmp_path):
    from algorithm.etm_mappo.checkpoint import load_checkpoint,validate_checkpoint
    from algorithm.etm_mappo.evaluation import evaluate
    runner,output,summary=navigation_run
    loaded,state=load_checkpoint(output/'final.pt')
    current=runner.observations
    zero=runner.trainer.initial_state(current)
    rng=torch.cuda.get_rng_state().clone()
    a=runner.trainer.act(current,zero,np.ones(2,dtype=bool),deterministic=True)[0]
    np.testing.assert_array_equal(a,loaded.act(current,zero,np.ones(2,dtype=bool),deterministic=True)[0])
    assert torch.equal(rng,torch.cuda.get_rng_state())
    inactive={key:value.copy() for key,value in current.items()};inactive['agent_alive'][0,1]=False
    action,raw,log,hidden=loaded.act(inactive,zero,np.zeros(2,dtype=bool),deterministic=True)
    assert np.count_nonzero(action[0,1])==np.count_nonzero(raw[0,1])==0 and log[0,1]==0
    assert torch.count_nonzero(hidden.self_hidden[0,1])==torch.count_nonzero(hidden.peer_hidden[0,1])==0
    assert evaluate(runner.trainer,[10000000])==evaluate(loaded,[10000000])
    for key,value in [('algorithm','MAPPO'),('algorithm','STEA-MAPPO'),('algorithm','ippo'),
                      ('implementation_version',999),('config_sha256','broken')]:
        altered=copy.deepcopy(state);altered[key]=value
        with pytest.raises(RuntimeError):validate_checkpoint(altered,runner.config)
    altered=copy.deepcopy(state);altered['network_architecture']=dict(altered['network_architecture'],memory_dim=32)
    with pytest.raises(RuntimeError):validate_checkpoint(altered,runner.config)
    with pytest.raises(FileExistsError):
        from algorithm.etm_mappo.runner import ETMTrainingRunner
        ETMTrainingRunner(runner.config,output)


@pytest.mark.parametrize('seed',[21,22])
def test_navigation_distinct_seed_runs(seed,tmp_path):
    from algorithm.etm_mappo.runner import ETMTrainingRunner
    cfg=config(seed=seed,num_envs=2,total_sampled_steps=8)
    cfg['training'].update(ppo_epochs=1,rollout_steps=4,evaluation_episodes=1)
    cfg['environment']['max_steps']=3
    summary=ETMTrainingRunner(cfg,tmp_path/f'seed{seed}').run()
    assert summary['seed']==seed and summary['sampled_steps']==8


@pytest.mark.parametrize('key,value',[('num_envs',0),('total_sampled_steps',0),('sequence_length',3),('gamma',1.1)])
def test_configuration_rejects_bad_runtime(key,value):
    with pytest.raises(ValueError):config(**{key:value})


def test_ablation_critics_start_identically_and_protocols_match():
    from algorithm.etm_mappo.trainer import ETMMAPPOTrainer
    trainers=[ETMMAPPOTrainer(config(variant=variant)) for variant in ('B0','B1','B2','B3')]
    first=trainers[0]
    for trainer in trainers[1:]:
        assert trainer.config['training']==first.config['training']
        assert trainer.config['environment']==first.config['environment']
        for key,value in first.critic.state_dict().items():
            torch.testing.assert_close(value,trainer.critic.state_dict()[key],rtol=0,atol=0)


def test_b2_is_exactly_b3_with_constant_gate_and_no_gate_parameters():
    full=actor('B3');ungated=actor('B2')
    ungated.load_state_dict({key:value for key,value in full.state_dict().items() if not key.startswith('gate.')})
    torch.nn.init.zeros_(full.gate.weight);torch.nn.init.zeros_(full.gate.bias)
    difference=sum(p.numel() for p in full.parameters())-sum(p.numel() for p in ungated.parameters())
    assert difference==sum(p.numel() for p in full.gate.parameters())==32896
    o=observation();start=torch.zeros(2,device=DEVICE,dtype=torch.bool)
    left=full.initial_state(o);right=ungated.initial_state(o)
    for _ in range(3):
        d,left,_=full.step(o,left,start);q,right,_=ungated.step(o,right,start)
        torch.testing.assert_close(d.mean,q.mean,rtol=0,atol=0)


def test_simple_navigation_has_reachable_collision_free_goal_and_no_free_return():
    from algorithm.etm_mappo.navigation import CooperativeNavigationEnv
    e=config()['environment'];env=CooperativeNavigationEnv(e);env.reset(3)
    for _ in range(e['max_steps']):
        _,_,terminated,truncated,stationary=env.step(np.zeros((4,2),np.float32))
        if terminated or truncated:break
    assert not stationary['success'] and not stationary['collision_episode']
    assert abs(stationary['episode_return'])<1e-5
    # A physical reachability check only: this controller never supplies policy
    # targets, rollout data or demonstrations to any learned method.
    env.reset(3)
    for _ in range(e['max_steps']):
        action=np.clip(1.5*(env.goal+env.offsets-env.positions)-2*env.velocities,-1,1).astype('f')
        _,_,terminated,truncated,controlled=env.step(action)
        if terminated or truncated:break
    assert controlled['success'] and not controlled['collision_episode']
    assert controlled['episode_return']>20
