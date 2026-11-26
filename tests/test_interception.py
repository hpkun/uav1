import numpy as np
import pytest

from env.interception import InterceptionConfig, InterceptionEnv
from env.parallel import ParallelInterceptionEnvs


def test_seed_observation_and_kinematics():
    config=InterceptionConfig()
    left=InterceptionEnv(config)
    right=InterceptionEnv(config)
    obs,info=left.reset(seed=42)
    other,_=right.reset(seed=42)
    np.testing.assert_array_equal(obs,other)
    assert obs.shape==(5,49) and info['central_state'].shape==(245,)
    assert np.all(left.positions[0,:,1]>=-95) and np.all(left.positions[0,:,1]<=-75)
    assert np.all(left.positions[1,:,1]>=75) and np.all(left.positions[1,:,1]<=90)
    before=left.positions.copy()
    left.step(np.full((5,2),[2.,0.]))
    np.testing.assert_allclose(left.positions,before)
    assert np.all(np.linalg.norm(left.velocities[0],axis=1)<=config.defender_speed_max)


def test_all_in_range_agents_have_fixed_relative_slots():
    env=InterceptionEnv()
    env.reset(seed=11)
    env.positions[0]=[[0,0],[5,0],[10,0],[15,0],[20,0]]
    env.positions[1]=[[0,25],[0,30],[0,35],[0,40],[0,45]]
    env.velocities.fill(0)
    observation=env.observations()[0]
    assert observation.shape==(49,)
    assert observation[4:24:5].tolist()==pytest.approx([.05,.1,.15,.2])
    assert observation[24:49:5].tolist()==pytest.approx([0]*5)
    assert observation[25:49:5].tolist()==pytest.approx([.25,.3,.35,.4,.45])
    assert observation[8:24:5].tolist()==[1]*4
    assert observation[28:49:5].tolist()==[1]*5
    env.positions[1,4]=[100,100]
    assert not env.observations()[0,44:49].any()
    env.positions[0,0]=[10,0]
    env.velocities[0,0]=[1,2]
    env.velocities[0,2]=[3,4]
    absolute=env.observations()[0]
    np.testing.assert_allclose(absolute[:4],[.1,0,1/env.config.attacker_speed_max,2/env.config.attacker_speed_max])
    np.testing.assert_allclose(absolute[4:9],[.1,0,3/env.config.attacker_speed_max,4/env.config.attacker_speed_max,1])
    legacy=InterceptionEnv(InterceptionConfig(visible_neighbors=3,visible_attackers=3,
                                               observation_layout='relative_with_base',communication_radius=60))
    assert legacy.reset(seed=11)[0].shape==(5,36)


def test_one_sensing_radius_applies_to_both_sides():
    env=InterceptionEnv(InterceptionConfig(sensing_radius=80))
    env.reset(seed=3)
    env.positions[0]=[[0,0],[70,0],[90,0],[0,90],[-90,0]]
    env.positions[1]=[[0,70],[90,10],[90,20],[90,30],[90,40]]
    env.velocities.fill(0)
    observation=env.observations()[0]
    np.testing.assert_allclose(observation[4:9],[.7,0,0,0,1])
    np.testing.assert_allclose(observation[24:29],[0,.7,0,0,1])
    assert not observation[9:24].any()
    assert not observation[29:49].any()
    np.testing.assert_array_equal(env.central_state(),env.observations().reshape(245))


def test_capture_precedes_breach_and_reward_balances():
    env=InterceptionEnv()
    env.reset(seed=7)
    env.positions[0,0]=[0,-65]
    env.positions[1,0]=[0,-65]
    env.velocities.fill(0)
    _,_,terminated,_,info=env.step(np.zeros((5,2)))
    assert not terminated
    assert not any(event['type']=='penetration' and event['attacker']==0 for event in info['events'])
    assert any(event['type']=='capture' and event['attacker']==0 for event in info['events'])
    assert not env.defender_alive[0]
    assert info['active_masks'][0]==0
    assert np.isclose(info['team_reward'],np.mean(info['individual_rewards']))
    assert np.isclose(info['team_reward'],sum(info['reward_components'].values()))


def test_all_capture_ends_episode_once():
    env=InterceptionEnv()
    env.reset(seed=5)
    env.positions[0]=[[x,-65] for x in (-40,-20,0,20,40)]
    env.positions[1]=env.positions[0].copy()
    env.velocities.fill(0)
    _,_,terminated,truncated,info=env.step(np.zeros((5,2)))
    assert terminated and not truncated
    assert info['episode_summary']['outcome']=='all_captured'
    assert info['episode_summary']['captured_count']==5
    assert len([event for event in info['events'] if event['type']=='capture'])==5
    assert not env.defender_alive.any()
    assert not info['active_masks'].any()
    assert not any(event['type']=='penetration' for event in info['events'])


def test_parallel_sampler_preserves_terminal_state():
    config=InterceptionConfig(episode_steps=1)
    with ParallelInterceptionEnvs(2,config,seed=3) as sampler:
        obs,states,masks,_=sampler.reset()
        result=sampler.step(np.zeros((2,5,2)))
    assert result.dones.tolist()==[1,1]
    assert result.terminated.tolist()==[True,True]
    assert result.truncated.tolist()==[False,False]
    assert result.rewards.shape==(2,)
    assert result.individual_rewards.shape==(2,5)
    np.testing.assert_allclose(result.rewards,result.individual_rewards.mean(axis=1),atol=1e-5)
    assert all(info['episode_summary']['outcome']=='penetrated' for info in result.infos)
    assert all(info['episode_summary']['penetrated_count']==5 for info in result.infos)
    assert np.any(result.observations!=result.transition_observations)


def test_unopposed_attackers_reach_the_lower_target():
    for mode in ('direct_rush','split_attack'):
        env=InterceptionEnv(InterceptionConfig(attack_mode=mode))
        env.reset(seed=7)
        for _ in range(500):
            _,_,terminated,truncated,info=env.step(np.zeros((5,2)))
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert info['episode_summary']['outcome']=='penetrated'
        assert 0<info['episode_summary']['steps']<500


def test_mappo_team_and_maddpg_individual_interfaces():
    from algorithm.mappo_config import MAPPOConfig
    from algorithm.maddpg_config import MADDPGConfig
    from experiment.interception import make_policy

    config=InterceptionConfig()
    env=InterceptionEnv(config)
    observations,info=env.reset(seed=9)
    batch_obs=observations[None]
    masks=info['active_masks'][None]
    state=info['central_state'][None]
    np.testing.assert_array_equal(state[0],observations.reshape(245))
    mappo=make_policy('mappo',config,MAPPOConfig(hidden_size=16),'cpu',9)
    assert mappo.critic.network[0].in_features==245
    assert mappo.policy_actions(batch_obs,masks).shape==(1,5,2)
    assert mappo.values(state).shape==(1,)
    maddpg=make_policy('maddpg',config,MADDPGConfig(hidden_dim=16,batch_size=2,learning_starts=2),'cpu',9)
    assert maddpg.critics[0].net[0].in_features==255
    assert maddpg.policy_actions(batch_obs,masks).shape==(1,5,2)
    transition=dict(observations=np.repeat(batch_obs,2,axis=0),next_observations=np.repeat(batch_obs,2,axis=0),
                    states=np.repeat(state,2,axis=0),next_states=np.repeat(state,2,axis=0),
                    actions=np.zeros((2,5,2),np.float32),rewards=np.zeros((2,5),np.float32),
                    dones=np.zeros(2,np.float32),active_masks=np.ones((2,5),np.float32),
                    next_active_masks=np.ones((2,5),np.float32))
    maddpg._batch(transition)
    with pytest.raises(ValueError,match='Invalid MADDPG joint replay batch'):
        maddpg._batch(dict(transition,rewards=np.zeros(2,np.float32)))


def test_evaluation_rows_use_environment_seed_once():
    from experiment.interception import evaluate_policy

    class StationaryPolicy:
        def policy_actions(self,observations,active_masks,*,deterministic):
            return np.zeros((len(observations),5,2),np.float32)

    rows=evaluate_policy(InterceptionConfig(episode_steps=2),StationaryPolicy(),
                         'direct_rush',episodes=2,seed_start=600000)
    assert [row['seed'] for row in rows]==[600000,600001]
    assert all(row['outcome']=='penetrated' for row in rows)


def test_approach_rewards_use_actual_distance_change_with_visible_target():
    config=InterceptionConfig()
    env=InterceptionEnv(config)
    env.reset(seed=7)
    env.attacker_alive[:]=False
    env.attacker_alive[0]=True
    env.positions[1,0]=[0,0]
    env.positions[0]=[[0,-50],[20,-50],[100,-90],[-100,-90],[100,-80]]
    env.velocities.fill(0)
    env.velocities[1,0]=[0,-config.attacker_speed_max]
    env.velocities[0,1]=[0,5]
    observations=env.observations()
    start=4+5*config.visible_neighbors
    np.testing.assert_allclose(observations[0,start:start+5],[0,0,0,-1,1])
    assert observations[2,start+4]==0
    _,_,_,_,info=env.step(np.zeros((5,2)))
    assert info['global_reward']==0
    assert 'danger' not in info['reward_components']
    assert info['local_rewards'][0]==pytest.approx(config.approach_weight*0.5)
    before=np.hypot(20,50)
    after=np.hypot(20,49)
    assert info['local_rewards'][1]==pytest.approx(config.approach_weight*(before-after))
    assert info['local_rewards'][2]==pytest.approx(0,abs=1e-12)


def test_breach_removes_only_that_attacker_and_shared_events_sum_to_minus_twenty():
    env=InterceptionEnv()
    env.reset(seed=7)
    env.positions[0]=[[0,-65],[40,-80],[50,-80],[60,-80],[70,-80]]
    env.positions[1]=[[0,-80],[-30,80],[0,80],[20,80],[30,80]]
    env.velocities.fill(0)
    _,rewards,terminated,truncated,info=env.step(np.zeros((5,2)))
    assert not terminated and not truncated
    assert info['reward_components']['penetration']==-100
    assert not env.attacker_alive[0] and env.attacker_alive[1:].all()
    assert info['team_reward']==pytest.approx(-100)
    np.testing.assert_allclose(rewards,-100)
    env.positions[1,1:]=env.positions[0,1:].copy()
    _,rewards,terminated,truncated,info=env.step(np.zeros((5,2)))
    assert terminated and not truncated
    assert info['episode_summary']['outcome']=='penetrated'
    assert info['episode_summary']['captured_count']==4
    assert info['episode_summary']['penetrated_count']==1
    assert info['episode_summary']['global_return']==-20
    assert info['reward_components']['capture']==80
    assert info['episode_summary']['defender_retired_count']==4
    assert info['active_masks'].tolist()==[1,0,0,0,0]
    np.testing.assert_allclose(rewards,80,atol=1)


def test_capture_retires_one_defender_and_cannot_capture_again():
    env=InterceptionEnv()
    env.reset(seed=7)
    env.positions[0]=[[0,0],[30,0],[60,0],[-30,0],[-60,0]]
    env.positions[1]=[[0,0],[0,0],[0,80],[20,80],[-20,80]]
    env.velocities.fill(0)
    _,_,terminated,_,info=env.step(np.zeros((5,2)))
    assert not terminated
    assert len([event for event in info['events'] if event['type']=='capture'])==1
    assert info['events'][0]['defender']==0
    assert not env.defender_alive[0] and not env.attacker_alive[0] and env.attacker_alive[1]
    assert not env.observations()[0].any()
    assert env.observations()[1,[8,13,18,23]].tolist()==[1,1,1,0]
    before=env.positions[0,0].copy()
    _,_,_,_,info=env.step(np.full((5,2),[2,0]))
    np.testing.assert_array_equal(env.positions[0,0],before)
    np.testing.assert_array_equal(env.velocities[0,0],[0,0])
    np.testing.assert_array_equal(info['executed_actions'][0,0],[0,0])
    assert env.attacker_alive[1]
    assert not any(event['type']=='capture' and event['defender']==0 for event in info['events'])


def test_legacy_capture_keeps_defender_active():
    env=InterceptionEnv(InterceptionConfig(episode_protocol='resolve_all_one_penalty'))
    env.reset(seed=7)
    env.positions[0,0]=[0,0]
    env.positions[1]=[0,0]
    env.velocities.fill(0)
    _,_,_,_,info=env.step(np.zeros((5,2)))
    assert env.defender_alive.all()
    assert info['active_masks'].tolist()==[1]*5
    assert info['episode_summary']['captured_count']==5


def test_replay_records_retirement_and_checks_one_to_one_capture():
    from tools.replay import html,validate_trace

    env=InterceptionEnv()
    env.reset(seed=7)
    env.positions[0]=[[x,-65] for x in (-40,-20,0,20,40)]
    env.positions[1]=env.positions[0].copy()
    env.velocities.fill(0)
    first=dict(step=0,defender_alive=env.defender_alive.tolist())
    _,_,_,_,info=env.step(np.zeros((5,2)))
    last=dict(step=1,defender_alive=env.defender_alive.tolist())
    trace=dict(frames=[first,last],events=info['events'],summary=info['episode_summary'],
               config=dict(episode_protocol=env.config.episode_protocol),mode='direct_rush',seed=7)
    assert validate_trace(trace) is trace
    page=html(trace,1)
    assert 'show-links' in page and 'defender_alive' in page
    assert first['defender_alive']==[True]*5 and last['defender_alive']==[False]*5
    trace['events'][1]['defender']=trace['events'][0]['defender']
    with pytest.raises(ValueError,match='more than one attacker'):
        validate_trace(trace)


def test_new_breach_penalty_is_once_per_episode_including_time_limit():
    env=InterceptionEnv(InterceptionConfig(episode_steps=3))
    env.reset(seed=7)
    env.positions[0]=[[50,-90],[60,-90],[70,-90],[80,-90],[90,-90]]
    env.positions[1,0]=[0,-80]
    env.velocities.fill(0)
    _,_,terminated,_,first=env.step(np.zeros((5,2)))
    assert not terminated
    assert first['reward_components']['penetration']==-100
    env.positions[1,1]=[0,-80]
    _,_,terminated,_,second=env.step(np.zeros((5,2)))
    assert not terminated
    assert second['reward_components']['penetration']==0
    _,_,terminated,_,last=env.step(np.zeros((5,2)))
    assert terminated
    assert last['reward_components']['penetration']==0
    assert last['episode_summary']['penetrated_count']==5
    assert last['episode_summary']['global_return']==-100

    expired=InterceptionEnv(InterceptionConfig(episode_steps=1))
    expired.reset(seed=7)
    _,_,terminated,_,info=expired.step(np.zeros((5,2)))
    assert terminated
    assert info['episode_summary']['penetrated_count']==5
    assert info['reward_components']['penetration']==-100


def test_prior_resolve_all_protocol_retains_per_attacker_penalty():
    env=InterceptionEnv(InterceptionConfig(episode_steps=1,episode_protocol='resolve_all'))
    env.reset(seed=7)
    _,_,terminated,_,info=env.step(np.zeros((5,2)))
    assert terminated
    assert info['episode_summary']['penetrated_count']==5
    assert info['reward_components']['penetration']==-500


def test_historical_first_breach_protocol_retains_original_reward_and_termination():
    env=InterceptionEnv(InterceptionConfig(episode_protocol='first_breach'))
    env.reset(seed=7)
    env.positions[1,0]=[0,-80]
    env.velocities.fill(0)
    _,rewards,terminated,truncated,info=env.step(np.zeros((5,2)))
    assert terminated and not truncated
    assert info['episode_summary']['outcome']=='penetrated'
    assert info['team_reward']==pytest.approx(-100)
    np.testing.assert_allclose(rewards,-100)


@pytest.mark.parametrize('algorithm',('mappo','maddpg'))
def test_visible_attacker_reaches_actor_input(algorithm):
    from algorithm.mappo_config import MAPPOConfig
    from algorithm.maddpg_config import MADDPGConfig
    from experiment.interception import make_policy

    env=InterceptionEnv()
    env.reset(seed=7)
    env.attacker_alive[:]=False
    env.attacker_alive[0]=True
    env.positions[0,0]=[0,-50]
    env.positions[1,0]=[10,0]
    observations=env.observations()
    start=4+5*env.config.visible_neighbors
    assert observations[0,start+4]==1
    parameters=MAPPOConfig(hidden_size=16) if algorithm=='mappo' else MADDPGConfig(hidden_dim=16)
    policy=make_policy(algorithm,env.config,parameters,'cpu',7)
    layer=policy.actor.network[0] if algorithm=='mappo' else policy.actors[0].net[0]
    seen=[]
    hook=layer.register_forward_pre_hook(lambda _module,args:seen.append(args[0].detach().cpu().numpy().copy()))
    try:
        policy.policy_actions(observations[None],np.ones((1,5),np.float32),deterministic=True)
    finally:
        hook.remove()
    actual=seen[0][0,0] if algorithm=='mappo' else seen[0][0]
    np.testing.assert_array_equal(actual[start:start+5],observations[0,start:start+5])


@pytest.mark.parametrize('algorithm',('mappo','maddpg'))
@pytest.mark.parametrize('eval_interval',(0,2,4))
@pytest.mark.parametrize('attack_mode',('direct_rush','split_attack'))
def test_training_writes_compact_episode_rewards(tmp_path,algorithm,eval_interval,attack_mode):
    import json

    from algorithm.mappo_config import MAPPOConfig
    from algorithm.maddpg_config import MADDPGConfig
    from experiment.interception import train

    config=InterceptionConfig(episode_steps=2)
    parameters=(MAPPOConfig(hidden_size=16,update_epochs=1,minibatch_size=4) if algorithm=='mappo'
                else MADDPGConfig(hidden_dim=16,batch_size=2,buffer_capacity=4,
                                  learning_starts=2,update_interval_steps=2,updates_per_interval=1))
    output=tmp_path/algorithm
    train(config,parameters,2,algorithm,steps=4,num_envs=2,seed=7,device='cpu',
          attack_mode=attack_mode,output=output,log_every_steps=4,
          eval_interval=eval_interval,eval_episodes=1)
    assert not (output/'reward_steps.jsonl.gz').exists()
    rows=[json.loads(line) for line in (output/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==2
    assert all(row['attack_mode']==attack_mode for row in rows)
    for row in rows:
        assert set(row['reward_components'])=={'capture','boundary','penetration','approach'}
        assert np.isclose(sum(row['reward_components'].values()),row['team_return'],rtol=0,atol=1e-6)
    expected_selection='periodic_evaluation' if eval_interval else 'final_unvalidated'
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    settings=json.loads((output/'settings.json').read_text(encoding='utf-8'))
    index=json.loads((output/'checkpoints'/'index.json').read_text(encoding='utf-8'))
    assert settings['attack_mode']==attack_mode
    assert settings['environment']['attack_mode']==attack_mode
    assert settings['environment']['episode_protocol']=='resolve_all_one_penalty_trade'
    assert summary['best_selection']==expected_selection
    assert summary['best_selection_mode']==attack_mode
    assert index['checkpoints']['Best']['selection']==expected_selection
    assert index['checkpoints']['Best']['selection_mode']==attack_mode
    assert (summary['best_score'] is not None)==bool(eval_interval)
    evaluations=[json.loads(line) for line in (output/'evaluations.jsonl').read_text(encoding='utf-8').splitlines()]
    if eval_interval:
        assert [row['step'] for row in evaluations]==list(range(eval_interval,5,eval_interval))
        assert all(set(row['modes'])=={attack_mode} for row in evaluations)
        for evaluation in evaluations:
            result=evaluation['modes'][attack_mode]
            components=result['mean_reward_components']
            assert set(components)=={'capture','boundary','penetration','approach'}
            assert np.isclose(sum(components.values()),result['mean_team_return'],atol=1e-6)
            assert components['penetration']==-100
    else:
        assert evaluations==[]
    if algorithm=='mappo' and eval_interval==0 and attack_mode=='direct_rush':
        from experiment.cli import main as cli_main
        from tools.replay import main as replay_main

        with pytest.raises(ValueError,match='Checkpoint trained for direct_rush'):
            replay_main(['--checkpoint',str(output),'--attack-mode','split_attack',
                         '--output',str(tmp_path/'wrong_replay')])
        with pytest.raises(ValueError,match='Checkpoint trained for direct_rush'):
            cli_main(['evaluate','--checkpoint',str(output),'--attack-mode','split_attack',
                      '--episodes-per-mode','1','--output-dir',str(tmp_path/'wrong_eval')])
        assert not (tmp_path/'wrong_replay').exists()
        assert not (tmp_path/'wrong_eval').exists()
        replay_main(['--checkpoint',str(output),'--attack-mode','direct_rush',
                     '--output',str(tmp_path/'matching_replay')])
        assert (tmp_path/'matching_replay'/'replay.html').exists()
        assert cli_main(['report','--run',str(output)])['best_selection_mode']=='direct_rush'


def test_unsupported_training_mode_is_rejected(tmp_path):
    from algorithm.mappo_config import MAPPOConfig
    from experiment.cli import parser
    from experiment.interception import train

    with pytest.raises(SystemExit):
        parser().parse_args(['train','--env-config','configs/interception.yaml',
            '--algorithm-config','configs/mappo_interception.yaml','--algorithm','mappo',
            '--attack-mode','unsupported_mode','--total-sampled-steps','4',
            '--output-dir',str(tmp_path/'rejected')])
    with pytest.raises(SystemExit):
        parser().parse_args(['evaluate','--checkpoint',str(tmp_path/'checkpoint'),
            '--attack-mode','unsupported_mode','--output-dir',str(tmp_path/'evaluation')])
    with pytest.raises(ValueError,match='Invalid training budget or attack mode'):
        train(InterceptionConfig(),MAPPOConfig(hidden_size=16),2,'mappo',steps=4,
              num_envs=2,seed=7,device='cpu',attack_mode='unsupported_mode',output=tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()


def test_unsupported_checkpoint_mode_is_rejected(tmp_path):
    import json

    from experiment.cli import main as cli_main
    from experiment.interception import load_run,require_matching_attack_mode

    (tmp_path/'settings.json').write_text(json.dumps({'attack_mode':'unsupported_mode'}),encoding='utf-8')
    with pytest.raises(ValueError,match='training attack mode is unsupported'):
        load_run(tmp_path)
    with pytest.raises(ValueError,match='Checkpoint trained for unsupported_mode'):
        require_matching_attack_mode({'attack_mode':'unsupported_mode'},'direct_rush')
    with pytest.raises(ValueError,match='Run training attack mode is unsupported'):
        cli_main(['report','--run',str(tmp_path)])


@pytest.mark.parametrize('algorithm',('mappo','maddpg'))
def test_eval_checkpoints_saved_by_default_and_loadable_across_tools(tmp_path,algorithm):
    import json
    import shutil

    from algorithm.mappo_config import MAPPOConfig
    from algorithm.maddpg_config import MADDPGConfig
    from experiment.cli import main as cli_main
    from experiment.interception import digest,load_run,train
    from tools.diagnose import main as diagnose_main
    from tools.replay import main as replay_main

    config=InterceptionConfig(episode_steps=2,observation_layout='relative_assignment',
                              assignment_method='distance_matching')
    parameters=(MAPPOConfig(hidden_size=16,update_epochs=1,minibatch_size=4) if algorithm=='mappo'
                else MADDPGConfig(hidden_dim=16,batch_size=2,buffer_capacity=4,
                                  learning_starts=2,update_interval_steps=2,updates_per_interval=1))
    output=tmp_path/algorithm
    train(config,parameters,2,algorithm,steps=4,num_envs=2,seed=7,device='cpu',
          attack_mode='direct_rush',output=output,log_every_steps=4,
          eval_interval=2,eval_episodes=1)
    settings=json.loads((output/'settings.json').read_text(encoding='utf-8'))
    assert settings['save_eval_checkpoints'] is True
    checkpoint_dir=output/'checkpoints'
    assert sorted(p.name for p in checkpoint_dir.iterdir())==[
        'best.pt','index.json','last.pt','step_2.pt','step_4.pt']
    index=json.loads((checkpoint_dir/'index.json').read_text(encoding='utf-8'))
    assert index['schema']=='InterceptionCheckpoints/1'
    catalog=index['checkpoints']
    assert set(catalog)=={'Best','Last','step_2','step_4'}
    assert catalog['step_2']['path']=='checkpoints/step_2.pt'
    assert catalog['step_2']['step']==2
    assert catalog['step_2']['selection']=='periodic_evaluation_snapshot'
    assert catalog['step_2']['selection_mode']=='direct_rush'
    assert catalog['step_2']['sha256']==digest(checkpoint_dir/'step_2.pt')
    assert catalog['Last']['sha256']==catalog['step_4']['sha256']
    assert catalog['Best']['sha256']==catalog[f"step_{catalog['Best']['step']}"]['sha256']
    evaluations=[json.loads(line) for line in (output/'evaluations.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [set(row) for row in evaluations]==[{'step','modes'},{'step','modes'}]

    _,by_label,_,model_by_label,_=load_run(output,'step_2','cpu')
    _,by_file,_,model_by_file,_=load_run(checkpoint_dir/'step_2.pt',device='cpu')
    assert model_by_label==model_by_file==(checkpoint_dir/'step_2.pt').resolve()
    assert type(by_label) is type(by_file)

    eval_by_label=cli_main(['evaluate','--checkpoint',str(output),'--label','step_2',
                            '--episodes-per-mode','1','--output-dir',str(tmp_path/'eval_label')])
    eval_by_file=cli_main(['evaluate','--checkpoint',str(checkpoint_dir/'step_2.pt'),
                           '--episodes-per-mode','1','--output-dir',str(tmp_path/'eval_file')])
    assert eval_by_label['checkpoint']==eval_by_file['checkpoint']==str((checkpoint_dir/'step_2.pt').resolve())

    replay_main(['--checkpoint',str(output),'--label','step_2','--attack-mode','direct_rush',
                 '--output',str(tmp_path/'replay_label')])
    replay_main(['--checkpoint',str(checkpoint_dir/'step_2.pt'),'--attack-mode','direct_rush',
                 '--output',str(tmp_path/'replay_file')])
    assert (tmp_path/'replay_label'/'replay.html').exists()
    assert (tmp_path/'replay_file'/'replay.html').exists()

    diag_label=diagnose_main(['--checkpoint',str(output),'--label','step_2','--attack-mode','direct_rush',
                              '--episodes','1','--output',str(tmp_path/'diag_label.json')])
    diag_file=diagnose_main(['--checkpoint',str(checkpoint_dir/'step_2.pt'),'--attack-mode','direct_rush',
                             '--episodes','1','--output',str(tmp_path/'diag_file.json')])
    assert diag_label['label']=='step_2' and diag_file['label']=='step_2'
    assert diag_label['checkpoint_sha256']==diag_file['checkpoint_sha256']==catalog['step_2']['sha256']

    with pytest.raises(ValueError,match='Unknown checkpoint label'):
        load_run(output,'step_999','cpu')
    with pytest.raises(ValueError,match='does not match requested label'):
        load_run(checkpoint_dir/'step_2.pt','Last','cpu')
    extra=checkpoint_dir/'unregistered.pt'
    shutil.copyfile(checkpoint_dir/'step_2.pt',extra)
    with pytest.raises(ValueError,match='not registered in catalog'):
        load_run(extra,device='cpu')
    (checkpoint_dir/'step_2.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='Checkpoint catalog digest mismatch'):
        load_run(output,'step_2','cpu')


def test_save_eval_checkpoints_flag_and_mid_training_catalog(tmp_path,monkeypatch):
    import json

    from algorithm.mappo_config import MAPPOConfig
    from experiment import interception as exp
    from experiment.cli import main as cli_main

    default_plan=cli_main(['preflight','--env-config','configs/interception_smoke.yaml',
                           '--algorithm-config','configs/mappo_interception.yaml',
                           '--algorithm','mappo','--attack-mode','direct_rush',
                           '--total-sampled-steps','4','--num-envs','2',
                           '--output-dir',str(tmp_path/'plan_default')])
    disabled_plan=cli_main(['preflight','--env-config','configs/interception_smoke.yaml',
                            '--algorithm-config','configs/mappo_interception.yaml',
                            '--algorithm','mappo','--attack-mode','direct_rush',
                            '--total-sampled-steps','4','--num-envs','2',
                            '--no-save-eval-checkpoints',
                            '--output-dir',str(tmp_path/'plan_disabled')])
    assert default_plan['save_eval_checkpoints'] is True
    assert disabled_plan['save_eval_checkpoints'] is False

    config=InterceptionConfig(episode_steps=2)
    parameters=MAPPOConfig(hidden_size=16,update_epochs=1,minibatch_size=4)
    disabled_out=tmp_path/'disabled'
    exp.train(config,parameters,2,'mappo',steps=4,num_envs=2,seed=7,device='cpu',
              attack_mode='direct_rush',output=disabled_out,log_every_steps=4,
              eval_interval=2,eval_episodes=1,save_eval_checkpoints=False)
    assert sorted(p.name for p in (disabled_out/'checkpoints').iterdir())==['best.pt','index.json','last.pt']
    disabled_index=json.loads((disabled_out/'checkpoints'/'index.json').read_text(encoding='utf-8'))
    assert set(disabled_index['checkpoints'])=={'Best','Last'}
    exp.load_run(disabled_out,'Best','cpu')
    exp.load_run(disabled_out,'Last','cpu')

    real_evaluate=exp.evaluate_policy
    calls=0
    def fail_on_second_eval(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:
            raise RuntimeError('simulated interruption')
        return real_evaluate(*args,**kwargs)

    monkeypatch.setattr(exp,'evaluate_policy',fail_on_second_eval)
    interrupted=tmp_path/'interrupted'
    with pytest.raises(RuntimeError,match='simulated interruption'):
        exp.train(config,parameters,2,'mappo',steps=4,num_envs=2,seed=7,device='cpu',
                  attack_mode='direct_rush',output=interrupted,log_every_steps=4,
                  eval_interval=2,eval_episodes=1)
    partial_index=json.loads((interrupted/'checkpoints'/'index.json').read_text(encoding='utf-8'))
    assert set(partial_index['checkpoints'])=={'Best','step_2'}
    assert not any(p.name.endswith('.tmp') for p in (interrupted/'checkpoints').iterdir())
    exp.load_run(interrupted,'Best','cpu')
    exp.load_run(interrupted,'step_2','cpu')
    exp.load_run(interrupted/'checkpoints'/'step_2.pt',device='cpu')
