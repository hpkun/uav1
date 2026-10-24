"""Full vector batches accept arbitrary budgets without hiding actual samples."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

import algorithm.maac.runner as module
from algorithm.train_maac import prepare_output_directory


def configs():
    return tuple(yaml.safe_load(Path(p).read_text(encoding='utf-8')) for p in (
        'configs/combat_environment_v350.yaml','configs/maac_4v4_v350.yaml'))


def make_runner(monkeypatch,tmp_path,budget,envs=12):
    ec,ac=configs()
    monkeypatch.setattr(module.torch.cuda,'is_available',lambda:True)
    trainer=SimpleNamespace(observation_dim=45,num_agents=4,action_dim=3,sampled_steps=0,
        actor_updates=0,policy_updates=0,critic_updates=0,target_updates=0,save=Mock())
    monkeypatch.setattr(module,'MAACTrainer',lambda *a:trainer)
    vector=SimpleNamespace(reset=lambda:np.zeros((envs,4,45)),close=Mock())
    monkeypatch.setattr(module,'ParallelVectorEnv',lambda *a:vector)
    monkeypatch.setattr(module,'JointReplayBuffer',lambda *a,**k:[])
    return module.MAACTrainingRunner(ec,ac,tmp_path,num_envs=envs,total_sampled_steps=budget)


@pytest.mark.parametrize('budget,envs,expected',[(2000000,12,2000004),(3000000,12,3000000),
    (25,12,36),(1,12,12),(12,12,12),(13,12,24),(32,12,36),(25,1,25)])
def test_requested_and_effective_budget(monkeypatch,tmp_path,budget,envs,expected):
    runner=make_runner(monkeypatch,tmp_path,budget,envs)
    assert runner.total_steps==budget and runner.effective_total_steps==expected
    assert expected%envs==0 and 0<=expected-budget<envs


@pytest.mark.parametrize('budget,envs',[(0,12),(-1,12),(25,0),(25,-1)])
def test_invalid_budgets_still_fail(monkeypatch,tmp_path,budget,envs):
    with pytest.raises(ValueError,match='positive'):make_runner(monkeypatch,tmp_path,budget,envs)


@pytest.mark.parametrize('budget,expected',[(25,36),(24,24),(1,12)])
def test_run_stops_after_final_batch_and_saves_actual_count(monkeypatch,tmp_path,budget,expected):
    runner=make_runner(monkeypatch,tmp_path,budget)
    calls=[]
    def collect():
        runner.trainer.sampled_steps+=12;calls.append(12);return []
    def evaluate():
        row={'sampled_steps':runner.trainer.sampled_steps}
        runner.evaluation_history.append(row);return row
    runner.collect_step=collect;runner.evaluate=evaluate;runner.log_train=lambda *a:None
    summary=runner.run()
    assert sum(calls)==expected and summary['sampled_steps']==expected
    assert summary['requested_total_sampled_steps']==budget
    assert summary['effective_total_sampled_steps']==expected
    assert summary['sampled_steps_overshoot']==expected-budget
    runner.trainer.save.assert_any_call(tmp_path/f'checkpoint_{expected}.pt')
    runner.vector.close.assert_called_once()


def test_failed_startup_snapshots_reused_without_overwrite(tmp_path):
    ec,ac=configs();prepare_output_directory(tmp_path,ec,ac)
    # Preserve bytes/mtime on retry, not just their decoded config values.
    before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    prepare_output_directory(tmp_path,ec,ac)
    assert before=={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in tmp_path.iterdir()}


@pytest.mark.parametrize('artifact',['run_config.json','latest.pt','final.pt','training_metrics.jsonl','train.log'])
def test_existing_run_never_reused(tmp_path,artifact):
    ec,ac=configs();prepare_output_directory(tmp_path,ec,ac)
    (tmp_path/artifact).write_bytes(b'existing result')
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(RuntimeError,match='fresh'):prepare_output_directory(tmp_path,ec,ac)
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}


def test_nonmatching_or_incomplete_snapshots_rejected(tmp_path):
    ec,ac=configs();prepare_output_directory(tmp_path,ec,ac)
    changed=deepcopy(ac);changed['training']['gamma']=.98
    with pytest.raises(RuntimeError,match='mismatch'):prepare_output_directory(tmp_path,ec,changed)
    (tmp_path/'algorithm_config.yaml').unlink()
    with pytest.raises(RuntimeError,match='fresh'):prepare_output_directory(tmp_path,ec,ac)
