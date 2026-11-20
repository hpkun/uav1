from experiment.progress import ProgressLogger


def test_progress_reports_modes_learning_and_eta():
    now=[0.0]
    progress=ProgressLogger('maddpg',1000,100,clock=lambda:now[0])
    progress.episode(dict(attack_mode='direct_rush',outcome='all_captured',captured_count=5,team_return=20.0))
    progress.episode(dict(attack_mode='split_attack',outcome='penetrated',captured_count=2,team_return=-50.0))
    progress.update(dict(actor_loss=-1.0,critic_loss=2.0,td_abs=0.5))
    now[0]=2.0
    assert not progress.ready(50)
    assert progress.ready(100)
    snapshot=progress.snapshot(100,2,1,replay_size=100,warmup=200)
    lines=ProgressLogger.lines(snapshot)
    assert '100/1,000' in lines[0] and 'ETA' in lines[0]
    assert 'direct n=1 all=100.0%' in lines[1]
    assert 'split n=1 all=0.0% breach=100.0% cap=40.0%' in lines[1]
    assert 'critic_loss=2.0000' in lines[2] and 'replay=100/200 warmup' in lines[2]
    assert not progress.ready(100)
