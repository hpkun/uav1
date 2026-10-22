"""49D MAAC integration, strict identity and continuous-dwell log semantics."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from algorithm.common.evaluator import aggregate_combat_records,episode_return_metrics
from algorithm.maac.metrics import evaluation_metrics
from algorithm.maac.protocol import validate_config_pair,validate_checkpoint,validate_checkpoint_for_resume
from algorithm.maac.replay import JointReplayBuffer
from algorithm.maac.runner import MAACTrainingRunner
from algorithm.maac.trainer import MAACTrainer
from env.config import load_config,environment_dimensions
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState


def configs(version):
    return load_config(f'configs/combat_environment_v{version}.yaml'),yaml.safe_load(
        Path(f'configs/maac_4v4_v{version}.yaml').read_text())


@pytest.fixture(scope='module',autouse=True)
def require_cuda():
    assert torch.cuda.is_available(),'CUDA mandatory for MAAC integration audit'
    torch.set_num_threads(1)


@pytest.mark.parametrize('version',['344','346'])
def test_formal_configuration_only_identity_and_observation_changed(version):
    ec,ac=configs(version);old=configs('341')[1]
    validate_config_pair(ec,ac)
    assert environment_dimensions(ec)==(49,3,4)
    normalized=deepcopy(ac);normalized['environment']['environment_version']='3.41'
    normalized['network']['observation_dim']=45
    assert normalized==old
    env=MultiUAVCombatEnv(ec);obs,_=env.reset(1)
    assert obs.shape==(4,49) and np.isfinite(obs).all()


@pytest.mark.parametrize('version',['344','346'])
@pytest.mark.parametrize('field,value',[('observation_dim',45),('action_dim',4),('num_agents',5),
                                       ('hidden_dim',32),('attention_heads',2)])
def test_strict_dimensions(version,field,value):
    ec,ac=configs(version);ac['network'][field]=value
    with pytest.raises(RuntimeError,match='exact dimensions'):validate_config_pair(ec,ac)


@pytest.mark.parametrize('version',['344','346'])
def test_gamma_and_identity_fail_closed(version):
    ec,ac=configs(version);bad=deepcopy(ac);bad['training']['gamma']=.98
    with pytest.raises(ValueError,match='gamma'):validate_config_pair(ec,bad)
    bad=deepcopy(ac);bad['environment']['environment_version']='3.46' if version=='344' else '3.44'
    with pytest.raises(RuntimeError,match='identity'):validate_config_pair(ec,bad)


def complete_episode(version):
    ec,_=configs(version);e=MultiUAVCombatEnv(ec);obs,_=e.reset(7)
    replay=JointReplayBuffer(512,seed=1,chunk_size=47,observation_dim=49,num_agents=4,action_dim=3)
    returns=np.zeros(4);terminals=0
    # Cross two full episodes without clearing joint replay.
    for episode in range(2):
        if episode:obs,_=e.reset(8)
        for _ in range(e.max_steps):
            alive=e.red_alive_mask.copy();actions=np.zeros((4,3),dtype=np.float32)
            nxt,r,term,trunc,info=e.step(actions)
            replay.add_batch(obs[None],actions[None],r[None],nxt[None],np.array([term or trunc]),
                             alive[None],e.red_alive_mask[None])
            returns+=r;obs=nxt
            if term or trunc:terminals+=1;break
    assert terminals==2
    return replay,obs,info


@pytest.mark.parametrize('version',['344','346'])
def test_joint_replay_full_episodes_dead_mask_baseline_and_update(version):
    replay,obs,_=complete_episode(version)
    batch=replay.sample(len(replay))
    assert batch.observations.shape==(len(replay),4,49) and batch.dones.sum()==2
    assert np.any(batch.next_alive_masks==0)
    t=MAACTrainer(*configs(version),'cuda',seed=3,smoke=True)
    alive=np.array([1,0,1,1]);a=t.act(obs,alive)
    assert a.shape==(4,3) and np.isfinite(a).all() and (a[1]==0).all()
    d=t.batch_tensors(batch)
    q,w=t.critic(d['observations'],d['actions'],d['alive_masks'],True)
    assert q.shape==(len(replay),4,1) and w.shape==(len(replay),4,4,4)
    assert torch.isfinite(q).all() and torch.isfinite(w).all()
    dead=~d['alive_masks'].bool()
    assert (q.squeeze(-1)[dead]==0).all()
    assert (w.masked_select(dead[:,None,:,None].expand_as(w))==0).all()
    assert (w.masked_select(dead[:,None,None,:].expand_as(w))==0).all()
    terminal=t.td_targets(d['rewards'],d['next_observations'],d['dones'],d['next_alive_masks'])
    stopped=d['dones'][:,None].bool()|~d['next_alive_masks'].bool()
    torch.testing.assert_close(terminal[stopped],d['rewards'][stopped],rtol=0,atol=0)
    base=t.counterfactual_baseline(d['observations'],d['actions'],d['alive_masks'],0)
    assert base.shape==(len(replay),) and torch.isfinite(base).all() and not base.requires_grad
    originals={n:[p.clone() for p in getattr(t,n).parameters()] for n in ('actor','critic','target_actor','target_critic')}
    gradients=[]
    handles=[p.register_hook(lambda g:gradients.append(g.detach().clone())) for p in t.critic.shared_parameters()]
    try:metrics=t.update(batch)
    finally:
        for h in handles:h.remove()
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics['actor_grad_norm']>0 and metrics['critic_grad_norm']>0
    assert gradients and all(torch.isfinite(g).all() for g in gradients) and any(g.abs().max()>0 for g in gradients)
    for n,before in originals.items():
        assert any(not torch.equal(p,b) for p,b in zip(getattr(t,n).parameters(),before))


def frozen_record(version,sharing=False):
    ec,_=configs(version);e=MultiUAVCombatEnv(ec);e.reset(1)
    def state(x=0,y=0,psi=0,alive=True):return AircraftState(x,y,-3000,225,0,psi,alive)
    if sharing:
        e.red=[state() for _ in range(4)]
        e.blue=[state(x=1500)]+[state(alive=False) for _ in range(3)]
    else:
        e.red=[state(x=x,psi=np.pi/2) for x in (-2500,-800,800,2500)]
        e.blue=[state(x=x,y=1500,psi=-np.pi/2) for x in (-2500,-800,800,2500)]
    e._advance=lambda states,actions:np.zeros(4)
    returns=np.zeros(4);infos=[]
    for k in range(5):
        _,r,term,trunc,info=e.step(np.zeros((4,3)))
        infos.append(info);returns+=r
        assert (term or trunc)==(k==4)
    team,agent=episode_return_metrics(returns)
    return dict(episode_return=team,mean_agent_episode_return=agent,**info),infos


@pytest.mark.parametrize('version',['344','346'])
def test_dwell_log_rates_unique_victims_and_evaluator_consistency(version,tmp_path):
    record,infos=frozen_record(version);shared,_=frozen_record(version,True)
    assert record['draw'] and record['red_survivors']==record['blue_survivors']==0
    assert shared['red_dwell_completed']==4 and shared['red_dwell_neutralizations']==1
    assert shared['red_attack_kills']==1 and shared['red_fire_attempts']==shared['red_weapon_hits']==0
    # Include an early stopped diagnostic record to test episode-rate denominators.
    empty=deepcopy(record)
    for side in ('red','blue'):
        for event in ('start','complete','geometry'):empty[f'{side}_first_dwell_{event}_step']=None
        for event in ('dwell_started','dwell_completed','dwell_neutralizations'):empty[f'{side}_{event}']=0
    empty.update(draw=False,red_survivors=4,blue_survivors=4)
    runner=MAACTrainingRunner.__new__(MAACTrainingRunner)
    runner.env_config,runner.config=configs(version);runner.records=[record,empty]
    runner.output_dir=tmp_path;runner.update_cycles=1;runner.replay=[None]*10
    runner.last_metrics={};runner.trainer=type('Counter',(),dict(sampled_steps=10))()
    recent=runner.recent();expected=evaluation_metrics(aggregate_combat_records(runner.records),runner.env_config)
    for key,value in expected.items():assert recent[key]==value
    assert recent['dwell_start_episode_rate']==recent['dwell_complete_episode_rate']==.5
    assert recent['mutual_annihilation_draw_rate']==.5 and recent['fire_rate'] is None
    assert recent['red_dwell_completed']==2 and recent['red_dwell_neutralizations']==2
    runner.write_metrics(infos)
    row=json.loads((tmp_path/'training_metrics.jsonl').read_text())
    assert row['red_step_dwell_completed']==4 and row['red_step_fire_attempts']==row['red_step_weapon_hits']==0
    assert row['event_semantics']=='continuous_dwell_neutralization' and row['fire_rate'] is None
    runner.records=[];assert runner.recent()['dwell_complete_episode_rate'] is None
    assert runner.recent()['legacy_fire_metrics_applicable'] is False


@pytest.mark.parametrize('source',['341','344','346'])
def test_checkpoint_all_networks_optimizers_and_version_isolation(source,tmp_path):
    ec,ac=configs(source);t=MAACTrainer(ec,ac,'cuda',seed=4,smoke=True)
    obs,_=MultiUAVCombatEnv(ec).reset(1);path=tmp_path/'c.pt';t.sampled_steps=24;t.save(path)
    state=torch.load(path,map_location='cuda',weights_only=False)
    loaded=MAACTrainer(ec,ac,'cuda',seed=5,smoke=True);loaded.load(path)
    np.testing.assert_array_equal(t.act(obs),loaded.act(obs))
    for name in ('actor','critic','target_actor','target_critic'):
        for p,q in zip(getattr(t,name).parameters(),getattr(loaded,name).parameters()):assert torch.equal(p,q)
    for name in ('actor_optimizer','critic_optimizer','actor_updates','policy_updates','critic_updates','target_updates'):
        assert name in state
    for other in set(('341','344','346'))-{source}:
        with pytest.raises(RuntimeError):validate_checkpoint(state,*configs(other))
    for key in ('environment_config_sha256','algorithm_config_sha256'):
        changed=deepcopy(state);changed['extra'].pop(key)
        with pytest.raises(RuntimeError,match=key):validate_checkpoint(changed,ec,ac)
    with pytest.raises(RuntimeError,match='exact resume unsupported'):validate_checkpoint_for_resume(state,ec,ac)


def test_historical_maac_log_meaning_preserved():
    r=MAACTrainingRunner.__new__(MAACTrainingRunner);r.env_config,r.config=configs('341')
    r.records=[dict(episode_return=5,red_success=True,red_first_attempt_step=2,red_first_kill_step=None)]
    assert r.recent()==dict(mean_return=5.,win_rate=1.,fire_rate=1.,kill_rate=0.)
    assert evaluation_metrics(dict(red_attempt_episode_rate=.5,red_kill_episode_rate=.25),r.env_config)==dict(
        red_attempt_episode_rate=.5,red_kill_episode_rate=.25,fire_rate=.5,kill_rate=.25)


def test_missing_legacy_funnel_skipped_and_existing_values_preserved():
    from algorithm.common.controller_diagnostics import controller_metric_values
    assert controller_metric_values([{'red_firing_geometry_agent_steps':{}}])=={}
    values=dict(live=4,range=3,range_az=2,geometry=1,eligible=1)
    actual=controller_metric_values([{'red_firing_geometry_agent_steps':values},
                                     {'red_firing_geometry_agent_steps':{}}])
    assert set(actual)=={f'red_funnel_{key}_agent_steps_per_env_step' for key in values}
    for key,value in values.items():np.testing.assert_array_equal(actual[f'red_funnel_{key}_agent_steps_per_env_step'],[value])
