"""Fresh-run identity guards are independent of matching tensor dimensions."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
import yaml
from env.config import environment_dimensions
from algorithm.mappo.runner import MAPPOTrainingRunner


def load(version='338',algorithm='mappo'):
    env=yaml.safe_load(Path(f'configs/combat_environment_v{version}.yaml').read_text(encoding='utf-8'))
    cfg=yaml.safe_load(Path(f'configs/{algorithm}_4v4_v{version}.yaml').read_text(encoding='utf-8'))
    return env,cfg


@pytest.fixture(autouse=True)
def cuda_required():
    assert torch.cuda.is_available(),'CUDA mandatory for runner/checkpoint tests'
    torch.set_num_threads(1)


@pytest.mark.parametrize('case',['v338','historical_no_identity','historical_declared_identity'])
def test_matching_or_historical_configuration_constructs_cuda_runner(tmp_path,case):
    env,cfg=load('338' if case=='v338' else '337')
    if case!='v338':
        assert 'environment' not in cfg
    if case=='historical_declared_identity':
        cfg['environment']={'environment_version':'3.37','team_size':4}
    runner=MAPPOTrainingRunner(env,cfg,num_envs=1,total_sampled_steps=4,
                              device='cuda',output_dir=tmp_path,smoke=True)
    try:
        assert str(runner.trainer.device).startswith('cuda')
        assert runner.observations.shape==(1,4,45)
        actions=runner.trainer.act(runner.observations,runner.alive_masks,deterministic=True)
        result=runner.vector.step_batch(actions)
        assert actions.shape==(1,4,3) and result.observations.shape==(1,4,45)
        assert np.isfinite(result.observations).all()
        assert not runner.trainer.actor_optimizer.state and not runner.trainer.critic_optimizer.state
    finally:
        runner.vector.close()


@pytest.mark.parametrize('case',['v337_env_v338_config','wrong_version','wrong_team','missing','null','extra_key'])
def test_wrong_identity_rejected_before_trainer_workers_or_output(tmp_path,case):
    env,cfg=load()
    if case=='v337_env_v338_config':env,_=load('337')
    elif case=='wrong_version':cfg['environment']['environment_version']='3.37'
    elif case=='wrong_team':cfg['environment']['team_size']=5
    elif case=='missing':del cfg['environment']
    elif case=='null':cfg['environment']=None
    else:cfg['environment']['unexpected']=True
    assert environment_dimensions(load('337')[0])==environment_dimensions(load('338')[0])==(45,3,4)
    assert tuple(cfg['network'][k] for k in ('observation_dim','action_dim','num_agents'))==(45,3,4)
    output=tmp_path/'must_not_be_created'
    with patch('algorithm.mappo.runner.MAPPOTrainer') as trainer, \
         patch('algorithm.mappo.runner.ParallelVectorEnv') as workers:
        with pytest.raises(RuntimeError,match='MAPPO algorithm/environment identity mismatch') as exc:
            MAPPOTrainingRunner(env,cfg,num_envs=1,total_sampled_steps=4,
                                device='cuda',output_dir=output,smoke=True)
        trainer.assert_not_called();workers.assert_not_called()
    assert 'expected=' in str(exc.value) and 'actual=' in str(exc.value)
    assert not output.exists()


def test_network_dimension_check_still_precedes_identity(tmp_path):
    env,cfg=load();cfg['network']['observation_dim']=44
    cfg['environment']['environment_version']='3.37'
    with patch('algorithm.mappo.runner.MAPPOTrainer') as trainer, \
         patch('algorithm.mappo.runner.ParallelVectorEnv') as workers:
        with pytest.raises(ValueError,match='network/environment dimension mismatch'):
            MAPPOTrainingRunner(env,cfg,device='cuda',output_dir=tmp_path)
        trainer.assert_not_called();workers.assert_not_called()


def test_madsac_diagnostic_ranges_cover_v338():
    from algorithm.madsac.protocol import validate_madsac_config
    env,cfg=load(algorithm='madsac')
    cfg['network']['actor_hidden_layers']=[128,128]
    with pytest.raises(RuntimeError,match='MADSAC v3.26-v3.38 requires'):
        validate_madsac_config(env,cfg)
    unsupported=yaml.safe_load(Path('configs/combat_environment_v24.yaml').read_text(encoding='utf-8'))
    with pytest.raises(RuntimeError,match='v3.30-v3.38 4v4/45D'):
        validate_madsac_config(unsupported,cfg)


def test_maac_diagnostic_reports_registered_range():
    from algorithm.maac.protocol import validate_config_pair
    _,cfg=load(algorithm='maac')
    unsupported=yaml.safe_load(Path('configs/combat_environment_v328.yaml').read_text(encoding='utf-8'))
    with pytest.raises(RuntimeError,match='registered supported environment identity.*v3.30-v3.38'):
        validate_config_pair(unsupported,cfg)
