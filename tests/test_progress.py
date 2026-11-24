from experiment.progress import ProgressLogger
from experiment.interception import evaluation_interval_for_budget
import pytest


def test_progress_reports_training_mode_learning_and_eta():
    now=[0.0]
    progress=ProgressLogger('maddpg',1000,100,attack_mode='direct_rush',clock=lambda:now[0])
    progress.episode(dict(attack_mode='direct_rush',outcome='all_captured',captured_count=5,team_return=20.0))
    progress.update(dict(actor_loss=-1.0,critic_loss=2.0,td_abs=0.5))
    now[0]=2.0
    assert not progress.ready(50)
    assert progress.ready(100)
    snapshot=progress.snapshot(100,1,1,replay_size=100,warmup=200)
    lines=ProgressLogger.lines(snapshot)
    assert '100/1,000' in lines[0] and 'replay 100/200 warmup' in lines[0]
    assert 'direct n=1 all=100.0%' in lines[1]
    assert 'split n=' not in lines[1]
    assert 'critic_loss=2.0000' in lines[2]
    assert not progress.ready(100)


def test_single_mode_training_log_shows_only_its_mode():
    progress=ProgressLogger('mappo',100,10,attack_mode='split_attack',clock=lambda:1.0)
    progress.episode(dict(attack_mode='split_attack',outcome='penetrated',captured_count=1,team_return=-20.0))
    line=ProgressLogger.lines(progress.snapshot(10,1,1))[1]
    assert 'split n=1' in line
    assert 'direct n=' not in line


def test_network_diagnostics_survive_progress_aggregation():
    for algorithm, metrics, expected in (
        ('mappo', dict(actor_grad_norm=.2, critic_grad_norm=.3,
                       policy_log_std_mean=-.5, policy_std_mean=.6,
                       advantage_raw_mean=2., advantage_raw_std=3.,
                       value_norm_std=4.), ('policy_std_mean=0.9000', 'value_norm_std=6.0000')),
        ('maddpg', dict(actor_grad_norm=.4, critic_grad_norm=.5,
                        actor_clip_rate=0., critic_clip_rate=1.,active_critic_count=1.,
                        q_mean=-2., q_std=1.5, target_q_mean=-1.),
         ('actor_clip=0.0%', 'critic_clip=100.0%', 'active_critics=1.0')),
    ):
        progress=ProgressLogger(algorithm,100,10,attack_mode='direct_rush',clock=lambda:1.0)
        progress.update(metrics)
        progress.update({key:(value if key in ('actor_clip_rate','critic_clip_rate','active_critic_count')
                              else value*2) for key,value in metrics.items()})
        learning=progress.snapshot(10,0,2)['learning']
        assert learning['actor_grad_norm']==pytest.approx(metrics['actor_grad_norm']*1.5)
        diagnostics=ProgressLogger.lines(progress.snapshot(10,0,2))[3]
        assert 'actor_grad_norm=' in diagnostics and 'critic_grad_norm=' in diagnostics
        for label in expected:
            key=label.split('=')[0]
            assert label in diagnostics


def test_five_percent_eval_schedule_and_reward_line_are_single_mode():
    assert evaluation_interval_for_budget(2_000_000,4)==100_000
    assert evaluation_interval_for_budget(101,1)==6
    summary=dict(all_capture_rate=0.0,penetration_rate=1.0,capture_efficiency=0.8,
                 episodes=20,mean_team_return=-20.5,
                 mean_reward_components=dict(capture=80.0,penetration=-100.0,
                                             approach=0.5,heading=0.3,boundary=-1.0))
    line=ProgressLogger.evaluation(100_000,{'direct_rush':summary})
    assert '[eval 100,000] direct:' in line
    assert 'split' not in line
    assert 'capture=+80.00' in line and 'penetration=-100.00' in line
    assert 'approach=+0.50' in line and 'heading=+0.30' in line and 'boundary=-1.00' in line
    assert 'return=-20.50' in line
    with pytest.raises(ValueError,match='only the training attack mode'):
        ProgressLogger.evaluation(100_000,{'direct_rush':summary,'split_attack':summary})


def test_maddpg_terminal_uses_exactly_last_100_updates_and_four_lines():
    progress=ProgressLogger('maddpg',200,10,attack_mode='jia_straight',attacker_count=1,
                            clock=lambda:1.0)
    for index in range(101):
        progress.update(dict(actor_loss=float(index),critic_loss=2.,td_abs=.2,q_mean=3.,
                             actor_grad_norm=.2,critic_grad_norm=12.,
                             actor_clip_rate=float(index==100),critic_clip_rate=float(index%2==0),
                             active_critic_count=1,critic_grad_norm_mean_agent=12.,
                             critic_grad_norm_max_agent=12.,q_std=5.,target_q_mean=4.))
    progress.episode(dict(attack_mode='jia_straight',outcome='penetrated',captured_count=0,
                          actual_penetrated_count=0,time_limit_occurred=True,team_return=-100.))
    row=progress.snapshot(101,1,101,replay_size=101)
    lines=ProgressLogger.lines(row)
    assert len(lines)==4
    assert row['learning']['actor_loss']==pytest.approx(50.5)
    assert row['learning']['actor_clip_rate']==pytest.approx(.01)
    assert row['learning']['critic_clip_rate']==pytest.approx(.5)
    assert row['recent']['jia_straight']['penetration_rate']==0
    assert row['recent']['jia_straight']['time_limit_rate']==1
    assert 'fail=100.0%' in lines[1] and 'timeout=100.0%' in lines[1]
    assert 'actor_clip=1.0%' in lines[3] and 'critic_clip=50.0%' in lines[3]
    assert 'q_mean=3.0000' in lines[2]
    assert all(field not in '\n'.join(lines) for field in
               ('critic_grad_norm_mean_agent','critic_grad_norm_max_agent','q_std','target_q_mean'))


def test_timeout_hidden_when_no_real_timeout():
    progress=ProgressLogger('maddpg',10,1,attack_mode='direct_rush',clock=lambda:1.0)
    progress.episode(dict(attack_mode='direct_rush',outcome='penetrated',captured_count=0,
                          actual_penetrated_count=1,time_limit_occurred=False,team_return=-100.))
    assert 'timeout=' not in ProgressLogger.lines(progress.snapshot(1,1,0))[1]
