"""Audit correctness; production environment/algorithm protocols are unchanged."""
import numpy as np
import torch
import yaml
from tools.audit_v30_initial_policy_baselines import (
    ROOT, EpisodeProbe, build_actor, first_order, wilson, Moments)
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config


def test_probe_does_not_change_environment_rng_or_transitions():
    config = load_config(ROOT/'configs/combat_environment_v30.yaml')
    a, b = MultiUAVCombatEnv(config), MultiUAVCombatEnv(config)
    a.reset(31000000); b.reset(31000000)
    probe = EpisodeProbe(b)
    for _ in range(180):
        actions = a.fixed_policy.team_actions(a.red,a.blue)
        probe.before_step(actions)
        x, y = a.step(actions), b.step(actions)
        for i in range(4):
            np.testing.assert_array_equal(x[i],y[i])
        for key in ('red_fire_attempts','blue_fire_attempts','red_weapon_hits','red_attack_kills','blue_attack_kills'):
            assert x[4][key]==y[4][key]
        assert a.rng.bit_generator.state==b.rng.bit_generator.state
        if x[2] or x[3]:
            break
    assert probe.window is not None


def test_actor_is_same_formal_initial_actor_and_modes_reproducible():
    from algorithm.common.control_factory import trainer_kwargs
    from algorithm.mappo.trainer import MAPPOTrainer
    config = yaml.safe_load((ROOT/'configs/mappo_5v5_v30.yaml').read_text())
    actor = build_actor(config,1,'cpu')
    trainer = MAPPOTrainer(**trainer_kwargs(config,'cpu',1))
    assert all(torch.equal(v,trainer.actor.state_dict()[k]) for k,v in actor.state_dict().items())
    obs = torch.ones(5,66)
    dist = actor.distribution(obs)
    torch.testing.assert_close(dist.scale,torch.full((5,3),np.exp(-.5)),rtol=1e-6,atol=0)
    torch.testing.assert_close(actor(obs),torch.tanh(dist.mean),rtol=0,atol=0)
    samples = []
    for seed in (10,10,11):
        rng = torch.Generator().manual_seed(seed)
        samples.append(torch.tanh(dist.mean+dist.scale*torch.randn((5,3),generator=rng)))
    assert torch.equal(samples[0],samples[1]) and not torch.equal(samples[0],samples[2])


def test_first_event_ties_and_missing_events_are_not_assigned_to_a_side():
    for r,b,expected in [(2,3,'red_first'),(3,2,'blue_first'),(2,2,'tie'),
                         (None,2,'blue_first'),(2,None,'red_first'),(None,None,'neither')]:
        assert first_order({'red_first_attempt_step':r,'blue_first_attempt_step':b},'attempt')==expected
    assert wilson(0,0) is None
    assert wilson(0,20)[1]>.1


def test_moments_per_dimension():
    m = Moments(); m.add(action=np.array([[1.,2.,3.],[-1.,0.,1.]]))
    result = m.result()['action']
    np.testing.assert_allclose(result['mean'],[0,1,2])
    np.testing.assert_allclose(result['std'],[1,1,1])
    np.testing.assert_allclose(result['mean_absolute'],[1,1,2])


def test_v30_parallel_autoreset_and_completed_episode_accounting():
    from algorithm.common.vector_env import ParallelVectorEnv
    from algorithm.mappo.runner import MAPPOTrainingRunner
    config = load_config(ROOT/'configs/combat_environment_v30.yaml')
    vector = ParallelVectorEnv(2,config,base_seed=31000100)
    references = [MultiUAVCombatEnv(config) for _ in range(2)]
    runner = object.__new__(MAPPOTrainingRunner)
    runner.agent_episode_returns=np.zeros((2,5));runner.completed_records=[]
    reference_returns=np.zeros((2,5))
    done_count=0
    try:
        initial=vector.reset()
        for i,e in enumerate(references):
            np.testing.assert_array_equal(initial[i],e.reset(31000100+i)[0])
        for _ in range(2100):
            actions=np.zeros((2,5,3),dtype=np.float32)
            expected=[e.step(actions[i]) for i,e in enumerate(references)]
            actual=vector.step_batch(actions)
            reference_returns+=actual.rewards
            completed=runner._completed(actual)
            assert len(completed)==int((actual.terminated|actual.truncated).sum())
            for i,(obs,reward,t,tr,info) in enumerate(expected):
                np.testing.assert_array_equal(actual.transition_next_observations[i],obs)
                np.testing.assert_array_equal(actual.rewards[i],reward)
                assert (actual.terminated[i],actual.truncated[i])==(t,tr)
                assert actual.infos[i]['termination_reason']==info['termination_reason']
                if t or tr:
                    done_count+=1
                    row=next(r for r in completed if r['episode_return']==float(reference_returns[i].sum()))
                    np.testing.assert_array_equal(row['per_agent_episode_returns'],reference_returns[i])
                    reference_returns[i]=0
                    fresh=references[i].reset(int(vector.last_reset_seeds[i]))[0]
                    np.testing.assert_array_equal(actual.observations[i],fresh)
                else:
                    np.testing.assert_array_equal(actual.observations[i],obs)
            if done_count>=4:
                break
        assert done_count>=4 and len(runner.completed_records)==done_count
        np.testing.assert_array_equal(runner.agent_episode_returns,reference_returns)
        runner.recent_episode_window=100
        runner.completed_records=[dict(team_episode_return=float(i),red_success=i>=5,red_losses=0,
            red_first_fire_window_step=1,red_first_kill_step=None) for i in range(105)]
        assert runner.recent_episode_metrics()['win']==1
        assert runner.recent_episode_metrics()['return']==54.5
    finally:
        vector.close()


def test_cuda_whole_stochastic_episodes_repeat_with_same_policy_seed(tmp_path):
    import multiprocessing as mp
    from tools.audit_v30_initial_policy_baselines import worker, run_group
    from algorithm.rmappo.protocol import require_cuda
    require_cuda('cuda')
    config=load_config(ROOT/'configs/combat_environment_v30.yaml')
    algorithm=yaml.safe_load((ROOT/'configs/mappo_5v5_v30.yaml').read_text())
    actor=build_actor(algorithm,3,'cuda')
    context=mp.get_context('spawn');connections=[];processes=[]
    try:
        for _ in range(2):
            parent,child=context.Pipe()
            process=context.Process(target=worker,args=(child,config),daemon=True)
            process.start();child.close();connections.append(parent);processes.append(process)
        results=[]
        for iteration in range(2):
            output=tmp_path/str(iteration);output.mkdir()
            results.append(run_group(connections,'init_stoch',actor,3,3,31001000,41001000,output))
        assert results[0]==results[1]
    finally:
        for connection in connections:connection.send(('close',None))
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():process.terminate();process.join()
        for connection in connections:connection.close()
