"""Auditable ACIRL: manual targets, local information, recurrent PPO and CUDA."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import torch
import yaml
from algorithm.acirl.protocol import validate_config,validate_checkpoint,require_cuda,MODES
from algorithm.acirl.trainer import ACIRLTrainer
from algorithm.acirl.state import initial_state,decide,gate_decision,finish_transition,PolicySession
from algorithm.acirl.advantages import team_gae,macro_advantages,ppo_component,event_mean
from algorithm.acirl.rollout import RolloutCollector
from algorithm.acirl.runner import ACIRLTrainingRunner
from algorithm.acirl.evaluation import evaluate,load_checkpoint,validate_seeds
from algorithm.acirl.networks import squashed_log_prob
from algorithm.common.vector_env import VectorStep,ParallelVectorEnv
from env.factory import make_combat_environment

ROOT=Path(__file__).resolve().parents[1]


def configs(mode='full'):
    suffix='' if mode=='full' else '_'+mode
    return (yaml.safe_load((ROOT/'configs/combat_environment_v339.yaml').read_text()),
            yaml.safe_load((ROOT/f'configs/acirl_4v4_v339{suffix}.yaml').read_text()))


@pytest.fixture
def trainer():
    _,c=configs();c['training'].update(ppo_epochs=1,minibatch_size=128,critic_minibatch_size=64)
    return ACIRLTrainer(c)


def vector(module):return torch.nn.utils.parameters_to_vector(module.parameters()).detach().clone()


def observation(trainer,environments=2):
    e,_=configs();env=make_combat_environment(e);obs,_=env.reset(21)
    return trainer.tensor(np.repeat(obs[None],environments,axis=0)),torch.ones((environments,4),device='cuda')


class FakeVector:
    """Deterministic aligned transitions; environment tests separately use real workers."""
    num_envs=1
    def __init__(self,death=None,end=None):
        e,_=configs();env=make_combat_environment(e);self.obs=env.reset(21)[0][None]
        self.current_alive_masks=np.ones((1,4),'f');self.step=0;self.death=death;self.end=end
    def reset(self):return self.obs.copy()
    def step_batch(self,actions):
        before=self.current_alive_masks.copy();next_alive=before.copy();self.step+=1
        if self.death==self.step:next_alive[:,0]=0
        done=np.array([self.end==self.step]);transition=self.obs.copy()+self.step*.001
        transition[next_alive==0]=0
        current=transition.copy()
        self.current_alive_masks=next_alive.copy()
        if done.any():
            self.current_alive_masks[:]=1;current=self.obs.copy()+10
        info=dict(executed_red_actions=actions.copy()[0])
        if done.any():
            info.update(episode_length=self.step,red_losses=4,blue_losses=0)
        return VectorStep(current,transition,np.ones((1,4),'f'),done,np.zeros(1,bool),[info],before,next_alive)


def test_network_shapes_independent_critics_and_optimizer_ownership(trainer):
    t=trainer;obs,alive=observation(t);state=initial_state(t.policy,2,t.device)
    data,new=decide(t.policy,obs,alive,state,t.behavior_rng)
    assert data['actions'].shape==(2,4,3) and new.hidden.shape==(2,4,128)
    assert data['gate'].shape==data['intent'].shape==(2,4)
    value,q=t.critics(obs,alive,data['hidden'],data['old_intent'],data['age_before'])
    assert value.shape==(2,) and q.shape==(2,4,2)
    assert t.critics.joint_dim==724 and t.critics.gate[0].in_features==909
    assert torch.all(data['actions'].abs()<=1)
    h,allies,valid,aw,ew=t.policy.memory(obs,state.hidden,state.previous_action,alive,state.reset)
    context,logits=t.policy.context(h,allies,valid)
    assert logits.shape==(2,4,3,6) and context.shape==(2,4,32)
    torch.testing.assert_close(logits.softmax(-1).sum(-1),torch.ones((2,4,3),device='cuda'))
    sets=[{id(p) for g in opt.param_groups for p in g['params']} for opt in (t.policy_optimizer,t.team_optimizer,t.gate_optimizer)]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    assert sets[0]=={id(p) for p in t.policy.parameters()}
    assert t.parameter_counts()['policy_total']==642835 and t.parameter_counts()['critics_total']==550915


def test_entity_masks_empty_attention_and_dead_nan_slots(trainer):
    t=trainer;obs,alive=observation(t);obs[...,7:]=0
    obs[:,3]=float('nan');alive[:,3]=0
    state=initial_state(t.policy,2,t.device)
    h,allies,valid,aw,ew=t.policy.memory(obs,state.hidden,state.previous_action,alive,state.reset)
    assert torch.isfinite(h).all() and not valid.any() and aw.count_nonzero()==ew.count_nonzero()==0
    assert h[:,3].count_nonzero()==0
    context,logits=t.policy.context(h,allies,valid)
    assert torch.isfinite(logits).all() and context.count_nonzero()==0


def test_no_global_observation_or_true_teammate_intent_leak(trainer):
    t=trainer;obs,alive=observation(t,1);state=initial_state(t.policy,1,t.device)
    state.intent[:]=1;state.age[:]=5;state.reset[:]=0
    a,_=decide(t.policy,obs,alive,state,t.behavior_rng,True)
    changed=state.clone();changed.intent[:,1:]=4;changed.hidden[:,1:]=100
    modified=obs.clone();modified[:,1:]*=100
    b,_=decide(t.policy,modified,alive,changed,t.behavior_rng,True)
    assert torch.equal(a['actions'][:,0],b['actions'][:,0])
    assert torch.equal(a['gate_probability'][:,0],b['gate_probability'][:,0])


def test_tanh_logprob_saturation_and_initial_logstd(trainer):
    t=trainer;obs,alive=observation(t);state=initial_state(t.policy,2,t.device)
    data,_=decide(t.policy,obs,alive,state,t.behavior_rng)
    h,allies,valid,*_=t.policy.memory(obs,state.hidden,state.previous_action,alive,state.reset)
    ctx,_=t.policy.context(h,allies,valid)
    _,_,mean,logstd=t.policy.heads(h,ctx,state.intent,state.age,data['intent'])
    torch.testing.assert_close(logstd,torch.full_like(logstd,-1.5))
    assert torch.equal(squashed_log_prob(mean,logstd,data['raw_actions']),data['action_old_logp'])
    assert torch.isfinite(squashed_log_prob(mean,logstd,torch.full_like(mean,100))).all()


@pytest.mark.parametrize('age,expected_valid,forced',[(0,False,False),(2,False,False),(3,True,False),(29,True,False),(30,False,True)])
def test_gate_duration_boundaries(age,expected_valid,forced):
    _,c=configs();logits=torch.ones(1,4,device='cuda')*100
    renew,valid,f,logp,reason=gate_decision(logits,torch.zeros_like(logits,dtype=torch.long),torch.full_like(logits,age),torch.ones_like(logits),c,None,True)
    assert valid.bool().all()==expected_valid and f.bool().all()==forced
    assert renew.bool().all()==(expected_valid or forced)
    if not expected_valid:assert logp.count_nonzero()==0


def test_initial_async_decisions_keep_renew_and_forced_gradient_masks(trainer):
    _,c=configs();old=torch.tensor([[-1,0,1,2]],device='cuda');age=torch.tensor([[0.,1,8,30]],device='cuda')
    logits=torch.tensor([[-5.,100.,5.,-5.]],device='cuda',requires_grad=True)
    renew,valid,forced,lp,reason=gate_decision(logits,old,age,torch.ones_like(age),c,None,True)
    assert renew.tolist()==[[True,False,True,True]]
    assert valid.tolist()==[[False,False,True,False]] and forced.tolist()==[[True,False,False,True]]
    (-event_mean(lp,valid)).backward()
    assert logits.grad[0,2]!=0 and logits.grad[0,[0,1,3]].count_nonzero()==0


def test_memory_death_reset_and_actions_recomputed_under_kept_intention(trainer):
    t=trainer;obs,alive=observation(t,1);state=initial_state(t.policy,1,t.device)
    first,state=decide(t.policy,obs,alive,state,t.behavior_rng,True)
    changed=obs.clone();changed[...,:7]+=.5
    second,next_state=decide(t.policy,changed,alive,state,t.behavior_rng,True)
    assert torch.equal(first['intent'],second['intent']) and not torch.equal(first['actions'],second['actions'])
    mask=alive.clone();mask[:,1]=0
    next_state=finish_transition(next_state,mask,torch.zeros(1,device='cuda'))
    assert next_state.hidden[:,1].count_nonzero()==0 and next_state.intent[0,1]==-1
    next_state=finish_transition(next_state,alive,torch.ones(1,device='cuda'))
    assert next_state.hidden.count_nonzero()==0 and (next_state.intent==-1).all() and next_state.reset.bool().all()


def test_manual_step_gae_and_episode_timeout_masks():
    reward=torch.tensor([[1.],[2.],[3.]])
    value=torch.tensor([[.5],[.6],[.7]]);next_value=torch.tensor([[.6],[.7],[8.]])
    done=torch.tensor([[0.],[0.],[1.]])
    adv,ret=team_gae(reward,value,next_value,done,.9,.8)
    a2=3-.7;a1=2+.9*.7-.6+.9*.8*a2;a0=1+.9*.6-.5+.9*.8*a1
    torch.testing.assert_close(adv,torch.tensor([[a0],[a1],[a2]]));torch.testing.assert_close(ret,adv+value)
    bootstrap,_=team_gae(reward,value,next_value,torch.zeros_like(done),.9,.8)
    assert bootstrap[-1]>adv[-1]  # Rollout cutoff bootstraps; environment timeout does not.


@pytest.mark.parametrize('duration',[3,5,10])
def test_hand_computed_smdp_duration_discount(duration):
    rewards=torch.ones(duration,1);values=torch.zeros_like(rewards);next_values=torch.full_like(rewards,2.)
    alive=torch.ones(duration,1,1);decisions=torch.zeros_like(alive);decisions[0]=1
    high,events=macro_advantages(rewards,values,next_values,torch.zeros_like(rewards),alive,alive,decisions,.9,.8)
    expected=sum(.9**k for k in range(duration))+.9**duration*2
    assert float(high[0,0,0])==pytest.approx(expected)
    assert events[0]['duration']==duration and events[0]['bootstrap_discount']==pytest.approx(.9**duration)
    assert high[1:].count_nonzero()==0


def test_smdp_asynchronous_recursion_death_and_episode_boundaries():
    rewards=torch.ones(6,1);values=torch.zeros_like(rewards);alive=torch.ones(6,1,2);next_alive=alive.clone()
    decisions=torch.zeros_like(alive);decisions[0,0,:]=1;decisions[3,0,0]=1;decisions[5,0,1]=1
    done=torch.zeros_like(rewards);done[-1]=1
    high,events=macro_advantages(rewards,values,values,done,alive,next_alive,decisions,.9,.8)
    macro3=1+.9+.9**2
    assert high[0,0,0]==pytest.approx(macro3+.9**3*.8*macro3)
    assert sorted((e['agent'],e['start'],e['duration']) for e in events)==[(0,0,3),(0,3,3),(1,0,5),(1,5,1)]
    next_alive[1:,0,1]=0;alive[2:,0,1]=0;decisions[5,0,1]=0
    high,events=macro_advantages(rewards,values,torch.full_like(values,99),done,alive,next_alive,decisions,.9,.8)
    assert high[0,0,1]==pytest.approx(1+.9)
    assert [e for e in events if e['agent']==1][0]['reason']=='agent_death'


def test_cross_rollout_intent_preserved_no_old_policy_reward_stitching(trainer):
    t=trainer
    with torch.no_grad():t.policy.gate_actor[-1].weight.zero_();t.policy.gate_actor[-1].bias.fill_(-100)
    collector=RolloutCollector(t,FakeVector())
    a=collector.collect(2);intent=collector.state.intent.clone();age=collector.state.age.clone()
    assert a.data['intent_valid'][0].bool().all() and a.data['intent_valid'][1].count_nonzero()==0
    assert all(e['reason']=='rollout_bootstrap' and e['duration']==2 for e in a.events)
    b=collector.collect(2)
    assert torch.equal(intent,collector.state.intent) and torch.equal(collector.state.age,age+2)
    assert not b.events and b.data['intent_valid'].count_nonzero()==0


def test_collector_alignment_death_reward_and_auto_reset(trainer):
    c=RolloutCollector(trainer,FakeVector(death=2,end=3));r=c.collect(4);d=r.data
    assert d['rewards'][1,0,0]==1 and d['alive'][1,0,0]==1 and d['next_alive'][1,0,0]==0
    assert d['intent'][2,0,0]==-1
    assert not torch.equal(d['next_observations'][2],d['observations'][3])
    assert d['reset'][3].bool().all() and d['episode_id'][3,0]==1
    for event in r.events:
        assert event['episode_id']==int(d['episode_id'][event['start'],event['environment']])
    assert all(e['end']<=3 for e in r.events if e['episode_id']==0)
    record=r.episode_records[0]
    assert record['internal_training_return']==pytest.approx(record['original_team_training_scale_return']-record['switching_cost_total'],abs=1e-5)


def test_counterfactual_selected_branch_supervision_and_detached_gate_advantage(trainer):
    t=trainer;obs,alive=observation(t);hidden=torch.zeros(2,4,128,device='cuda');z=torch.zeros(2,4,dtype=torch.long,device='cuda');age=torch.ones(2,4,device='cuda')
    v,q=t.critics(obs,alive,hidden,z,age)
    loss=(q[...,0]-7).square().mean();t.gate_optimizer.zero_grad();loss.backward()
    assert t.critics.gate[-1].weight.grad[0].abs().sum()>0
    assert t.critics.gate[-1].weight.grad[1].count_nonzero()==0
    assert all(p.grad is None for p in t.critics.team.parameters())
    for p in t.critics.parameters():p.grad=None
    logits=torch.zeros(2,4,device='cuda',requires_grad=True);prob=logits.sigmoid()
    chosen=q.detach()[...,1];adv=chosen-((1-prob.detach())*q.detach()[...,0]+prob.detach()*q.detach()[...,1])
    gate_loss=-(torch.nn.functional.logsigmoid(logits)*adv).mean();gate_loss.backward()
    assert logits.grad.abs().sum()>0 and all(p.grad is None for p in t.critics.parameters())


def test_clipped_ppo_and_invalid_samples():
    logp=torch.log(torch.tensor([1.5,.5,100.],requires_grad=True));old=torch.zeros(3);adv=torch.tensor([2.,-2.,float('nan')]);mask=torch.tensor([1.,1.,0.])
    loss,kl,clip=ppo_component(logp,old,adv,mask,.2)
    assert float(loss.detach())==pytest.approx((-(1.2*2)+1.6)/2) and clip==1
    empty,_,_=ppo_component(logp,old,adv,mask*0,.2);assert empty==0


def test_recurrent_ratio_burn_in_padding_and_real_gradients(trainer):
    t=trainer;rollout=RolloutCollector(t,FakeVector()).collect(35)
    errors=t.audit_rollout_ratio(rollout);assert max(errors.values())<2e-4
    packed,out=t.evaluate_sequences(rollout.data,[(0,32,35)],False)
    assert packed['alive'][3:].count_nonzero()==0
    assert packed['gate_valid'][3:].count_nonzero()==packed['intent_valid'][3:].count_nonzero()==0
    assert out['action_logp'].shape==(32,1,4)
    before={key:vector(module) for key,module in t.modules_for_audit().items()}
    metrics=t.update(rollout)
    assert metrics['valid_high_decisions']>0 and metrics['valid_gate_decisions']>0
    assert metrics['policy_grad_post']<=.500001 and metrics['team_grad_post']<=.500001 and metrics['gate_critic_grad_post']<=.500001
    assert all(not torch.equal(before[key],vector(module)) for key,module in t.modules_for_audit().items())
    assert all(v>0 for v in metrics['gradient_norms'].values())
    with pytest.raises(RuntimeError,match='stale'):t.update(rollout)


def test_empty_gate_events_skip_adam_momentum_and_critics_do_not_update_policy(trainer):
    t=trainer;first=RolloutCollector(t,FakeVector()).collect(12);t.update(first)
    short=RolloutCollector(t,FakeVector()).collect(2)
    assert short.data['gate_valid'].count_nonzero()==0
    gate_before=vector(t.policy.gate_actor);policy_before=vector(t.policy)
    t._fit_critics(short.data)
    assert torch.equal(policy_before,vector(t.policy))
    t.update(short)
    assert torch.equal(gate_before,vector(t.policy.gate_actor))


def test_warm_restart_new_episodes_explicit_provenance(tmp_path):
    e,c=configs();c['training'].update(num_train_envs=2,total_sampled_steps=128,rollout_steps=32,ppo_epochs=1,
        minibatch_size=128,critic_minibatch_size=64,evaluation_episodes=1,evaluation_interval=128,checkpoint_interval=64)
    first=ACIRLTrainingRunner(e,c,tmp_path/'original');original=first.run()
    resumed=ACIRLTrainingRunner(e,c,tmp_path/'warm',warm_start=tmp_path/'original/checkpoint_64.pt')
    assert resumed.collector.state.hidden.count_nonzero()==0 and (resumed.collector.state.intent==-1).all()
    summary=resumed.run()
    assert summary['sampled_steps']==128 and summary['update_count']==2
    assert summary['warm_start_source'] and not summary['formal_exact_resume_supported']
    metadata=json.loads((tmp_path/'warm/run_config.json').read_text())
    assert metadata['environment_seed_base']==67


@pytest.mark.parametrize('mode',MODES)
def test_all_ablation_objectives_real_cuda_update_and_architecture(mode):
    e,c=configs(mode);validate_config(e,c);c['training'].update(ppo_epochs=1,minibatch_size=128,critic_minibatch_size=64)
    t=ACIRLTrainer(c);r=RolloutCollector(t,FakeVector()).collect(16);m=t.update(r)
    assert m['parameter_drift_l2']['action_actor']>0 and m['parameter_drift_l2']['team_critic']>0
    if mode=='no_intent':
        assert not t.policy.has_intent and not t.policy.has_gate and not t.policy.has_inference
        assert m['valid_high_decisions']==m['valid_gate_decisions']==0 and m['intention_loss']==m['gate_loss']==m['inference_loss']==0
    if mode=='fixed_interval':assert not t.policy.has_gate and m['valid_gate_decisions']==0 and m['gate_q_loss'] is None
    if mode=='no_gate_q':assert not t.critics.has_gate_q and m['gate_q_loss'] is None and m['valid_gate_decisions']>0
    if mode=='no_teammate_inference':assert not t.policy.has_inference and m['inference_samples']==0
    if mode=='no_temporal_critic':
        high=r.data['intent_valid'].bool()
        assert torch.equal(r.data['high_advantage'][high],r.data['low_advantage'][high])
        assert all(event['advantage_method']=='step_gae_ablation' for event in r.events)
    if mode in ('no_switching_cost','fixed_interval','no_intent'):assert m['switching_cost_sum']==0


@pytest.mark.parametrize('case',['identity','dimensions','heads','duration','std','budget','mode','weights','cost','seed'])
def test_invalid_protocol_config_rejected(case):
    e,c=configs()
    if case=='identity':c['environment']['environment_version']='3.41'
    elif case=='dimensions':c['network']['observation_dim']=44
    elif case=='heads':c['network']['attention_heads']=3
    elif case=='duration':c['intention']['min_duration']=31
    elif case=='std':c['network']['log_std_init']=10
    elif case=='budget':c['training']['total_sampled_steps']=17
    elif case=='mode':c['ablation']['mode']='unknown'
    elif case=='weights':c['loss_weights']['gate']=0
    elif case=='cost':c['training']['switching_cost']=-1
    else:c['training']['evaluation_seed_base']=1
    with pytest.raises((ValueError,RuntimeError)):validate_config(e,c)


def test_seed_holdout_and_cuda_requirements(trainer):
    _,c=configs()
    assert validate_seeds([70000000],c,True)==(70000000,)
    for seeds,holdout in (([70000000],False),([10000000],True),([1],False),([10000000,10000000],False)):
        with pytest.raises(ValueError):validate_seeds(seeds,c,holdout)
    with pytest.raises(RuntimeError,match='CUDA'):require_cuda('cpu')
    with pytest.raises(RuntimeError,match='unsupported'):trainer.resume()


@pytest.fixture(scope='module')
def smoke(tmp_path_factory):
    e,c=configs();out=tmp_path_factory.mktemp('acirl_cuda')/'run'
    runner=ACIRLTrainingRunner(e,c,out,smoke=True);summary=runner.run()
    return e,runner.config,out,summary


def test_real_cuda_parallel_train_all_heads_updated_artifacts(smoke):
    e,c,out,s=smoke
    assert s['sampled_steps']==128 and s['vector_steps']==64 and s['update_count']==2
    assert all(v>0 for v in s['parameter_drift_l2'].values())
    config=json.loads((out/'run_config.json').read_text())
    assert len(set(config['worker_pids']))==2 and config['worker_metadata'][0]['environment_class']=='CombatEnvironmentV339'
    assert c['network']['actor_hidden']==[256,256] and c['network']['gru_hidden']==128
    for name in ('env_config.yaml','algorithm_config.yaml','requested_algorithm_config.yaml','run_config.json','optimization_metrics.jsonl',
                 'training_metrics.jsonl','macro_decisions.jsonl','intention_events.jsonl','evaluation_history.csv','best_eval.pt','latest.pt','final.pt','run_summary.json'):
        assert (out/name).is_file()


def test_checkpoint_reload_modes_reproducible_evaluation_and_rng_isolated(smoke):
    e,c,out,s=smoke;a=load_checkpoint(out/'final.pt',e,c);b=load_checkpoint(out/'latest.pt',e,c)
    behavior=a.behavior_rng.get_state().clone();global_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    permutation=deepcopy(a.permutation_rng.bit_generator.state)
    for mode in ('deterministic','stochastic'):
        first=evaluate(a,e,[10000000],mode);second=evaluate(b,e,[10000000],mode)
        assert first==second and first['episode_length']>0
        if mode=='deterministic':assert first['average_return']==s['evaluation_history'][-1]['average_return']
    assert torch.equal(behavior,a.behavior_rng.get_state()) and permutation==a.permutation_rng.bit_generator.state
    assert torch.equal(global_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
    held=evaluate(a,e,[70000000],'deterministic',True);assert held['holdout']


@pytest.mark.parametrize('case',['algorithm','version','env_hash','config_hash','mode','missing_rng','optimizer_steps','resume'])
def test_checkpoint_strict_identity_and_metadata(smoke,case):
    e,c,out,_=smoke;state=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
    if case=='algorithm':state['algorithm']='MAPPO'
    elif case=='version':state['implementation_version']=99
    elif case=='env_hash':state['environment_config_sha256']='0'*64
    elif case=='config_hash':state['algorithm_config_sha256']='0'*64
    elif case=='mode':state['ablation_mode']='no_gate_q'
    elif case=='missing_rng':state['rng_states'].pop('behavior')
    elif case=='optimizer_steps':state['policy_optimizer_steps']=0
    else:state['formal_exact_resume_supported']=True
    with pytest.raises(RuntimeError):validate_checkpoint(state,e,c)


def test_cross_ablation_loading_rejected(smoke):
    e,c,out,_=smoke;other=deepcopy(c);other['ablation']['mode']='no_gate_q'
    with pytest.raises(RuntimeError,match='mismatch'):ACIRLTrainer(other).load(out/'final.pt',e)


def test_reloaded_canonical_recording_trace_sidecars_rng_and_render(smoke,tmp_path):
    from tools.record_combat_episode import load_recording_policy,record
    from tools.combat_trace import read_trace
    from tools.render_combat_episode import render
    from tools.combat_visualization import RecordingCombatEnv
    e,c,out,_=smoke;trainer,state,metadata=load_recording_policy(out/'final.pt',e,c)
    result=record(trainer,e,10000000,tmp_path/'recorded',metadata)
    trace=read_trace(tmp_path/'recorded/episode_trace.npz',result)
    sidecar=np.load(tmp_path/'recorded/acirl_decisions.npz')
    assert sidecar['intent'].shape==trace['red_actions'].shape[:2]
    assert result['algorithm_decision_rows']==result['episode_steps']
    assert (tmp_path/'recorded/acirl_intention_events.json').is_file()
    env=make_combat_environment(e);rec=RecordingCombatEnv(e)
    obs,_=env.reset(10000000);rec.reset(10000000);session=PolicySession(trainer.policy,trainer.device,10000000,True)
    for index in range(result['episode_steps']):
        action=session.act(obs,env.red_alive_mask)
        np.testing.assert_array_equal(action,trace['red_actions'][index])
        obs,reward,terminated,truncated,info=env.step(action);robs,rreward,rt,ru,ri=rec.step(action)
        assert env.rng.bit_generator.state==rec.rng.bit_generator.state
        np.testing.assert_array_equal(obs,robs);np.testing.assert_array_equal(reward,rreward)
    assert terminated or truncated
    render(tmp_path/'recorded/episode_trace.npz',tmp_path/'recorded/metadata.json',tmp_path/'recorded/trajectory.png')
    assert (tmp_path/'recorded/trajectory.png').stat().st_size>0
