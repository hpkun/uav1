"""Formal identity-only migration and frozen source/target provenance contracts."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import yaml

from env.combat_env import MultiUAVCombatEnv
from algorithm.common.evaluator import aggregate_combat_records
from algorithm.common.protocol import config_sha256
from tools.verify_v339_training import NAMES, validate
from tools.compare_v338_v339_blue import provenance, paired_delta

ROOT=Path(__file__).resolve().parents[1]


def configs(name,version='339'):
    return tuple(yaml.safe_load((ROOT/f'configs/{prefix}_v{version}.yaml').read_text(encoding='utf-8'))
                 for prefix in ('combat_environment',name+'_4v4'))


@pytest.mark.parametrize('name',NAMES)
def test_formal_config_identity_only_parity(name):
    _,old=configs(name,'338'); _,new=configs(name)
    assert new['environment']==dict(environment_version='3.39',team_size=4)
    new['environment']['environment_version']='3.38'
    assert new==old
    if name=='mappo':
        assert old['training']['gae_lambda']==.95


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('version',['338','339'])
def test_formal_protocol_accepts_matching_identity(name,version):
    env,c=configs(name,version)
    if name in ('mappo_ca','ea_mappo','eaa_mappo'):
        from algorithm.eaa_mappo.protocol import validate_config,contract,checkpoint_family
        validate_config(env,c)
        assert contract(env,c)['environment_version']==env['environment_version']
        assert checkpoint_family(env)=='eaa_mappo_v'+version
    elif name=='maac':
        from algorithm.maac.protocol import validate_config_pair
        validate_config_pair(env,c)
    elif name=='maddpg':
        from algorithm.maddpg.protocol import validate_config
        validate_config(env,c)
    elif name=='madsac':
        from algorithm.madsac.protocol import validate_madsac_config
        validate_madsac_config(env,c)
    else:
        from algorithm.mappo.runner import MAPPOTrainingRunner
        with patch('algorithm.mappo.runner.MAPPOTrainer',side_effect=RuntimeError('identity passed')):
            with pytest.raises(RuntimeError,match='identity passed'):
                MAPPOTrainingRunner(env,c,device='cuda',output_dir=ROOT/'outputs/v339_identity_test')


@pytest.mark.parametrize('name',NAMES)
def test_wrong_and_missing_formal_identity_rejected(name,tmp_path):
    from algorithm.mappo.runner import MAPPOTrainingRunner
    from algorithm.eaa_mappo.protocol import validate_config as eaa
    from algorithm.maac.protocol import validate_config_pair as maac
    from algorithm.maddpg.protocol import validate_config as maddpg
    from algorithm.madsac.protocol import validate_madsac_config as madsac
    env,c=configs(name)
    validators={'mappo_ca':eaa,'ea_mappo':eaa,'eaa_mappo':eaa,'maac':maac,'maddpg':maddpg,'madsac':madsac}
    for identity in (None,dict(environment_version='3.38',team_size=4),dict(environment_version='3.39',team_size=5)):
        changed=deepcopy(c)
        if identity is None:
            changed.pop('environment')
        else:
            changed['environment']=identity
        with pytest.raises((ValueError,RuntimeError),match='identity'):
            if name=='mappo':
                MAPPOTrainingRunner(env,changed,device='cuda',output_dir=tmp_path)
            else:
                validators[name](env,changed)


def terminal_record(version):
    env=MultiUAVCombatEnv(configs('mappo',version)[0]);env.reset(9)
    env.max_steps=1
    _,reward,_,_,info=env.step(np.zeros((4,3),np.float32))
    return dict(episode_return=float(reward.sum()),mean_agent_episode_return=float(reward.mean()),**info)


def test_shared_evaluator_v339_metric_schema_matches_v338():
    a=aggregate_combat_records([terminal_record('338')])
    b=aggregate_combat_records([terminal_record('339')])
    assert a.keys()==b.keys()
    for key in ('average_episode_paper_reward_total','average_episode_safety_total',
                'average_episode_extended_reward_total','average_blue_target_switches',
                'red_ceiling_losses','average_red_noncombat_losses'):
        assert key in b
    for name in ('r1','r2','r3','r4','individual_event','team_casualty','boundary'):
        assert f'average_episode_{name}_total' in b


def test_runner_v339_rollout_metrics_and_startup_version(tmp_path):
    from algorithm.mappo.runner import MAPPOTrainingRunner
    from types import SimpleNamespace
    # Use the unchanged accumulation/finalization methods with real environment info.
    env=MultiUAVCombatEnv(configs('mappo')[0]);env.reset(4)
    _,_,_,_,info=env.step(np.zeros((4,3),np.float32))
    runner=object.__new__(MAPPOTrainingRunner)
    runner.env_config=env.config
    runner._begin_rollout_metrics()
    runner._accumulate_rollout_metrics(SimpleNamespace(infos=[info],rewards=np.zeros((1,4))),[])
    for key in ('mean_r1_reward','mean_r2_reward','mean_r3_reward','mean_r4_reward',
                'mean_paper_reward','mean_safety_reward','mean_extended_reward'):
        assert key in runner._rollout_metrics['sums']


def test_environment_config_only_existing_policy_difference():
    a,_=configs('mappo','338');b,_=configs('mappo')
    assert {k for k in a if a[k]!=b[k]}=={'environment_version','blue_policy'}


def test_cross_environment_provenance_never_relabels_source():
    a,_=configs('mappo','338');b,ac=configs('mappo')
    loaded=dict(algorithm_config_sha256='source_algorithm_hash',checkpoint_sha256='source_checkpoint_hash')
    result=provenance(a,b,loaded)
    assert result['source_checkpoint_environment_version']=='3.38'
    assert result['target_environment_version']=='3.39'
    assert result['source_environment_config_sha256']==config_sha256(a)
    assert result['target_environment_config_sha256']==config_sha256(b)
    assert result['source_algorithm_config_sha256']=='source_algorithm_hash'
    assert result['cross_environment_inference'] is True


def test_frozen_loader_validates_source_and_freezes_weights():
    import torch
    from types import SimpleNamespace
    from tools.compare_v338_v339_blue import source_policy
    assert torch.cuda.is_available(),'CUDA mandatory'
    source,_=configs('mappo','338')
    actor=torch.nn.Linear(45,3).cuda()
    trainer=SimpleNamespace(actor=actor)
    job={'checkpoint':'source.pt'}
    with patch('tools.audit_red_speed.load_policy',return_value=(trainer,source,{'source':True})) as official:
        loaded,env,meta=source_policy(job)
    official.assert_called_once_with(job)
    assert loaded is trainer and env is source and meta=={'source':True}
    assert not actor.training and not any(p.requires_grad for p in actor.parameters())


def test_paired_bootstrap_uses_episodes_and_signed_delta():
    result=paired_delta([1,2,3],[2,3,4],np.random.default_rng(4),100)
    assert result==dict(delta=1.,ci95=[1.,1.],episodes=3)


def test_shared_scenario_is_one_bootstrap_unit_across_policies():
    result=paired_delta([0,0,0,0],[1,1,-1,-1],np.random.default_rng(5),1000,
                        clusters=[30001000,30001000,30001001,30001001])
    assert result['delta']==0 and result['episodes']==4
    assert result['bootstrap_scenario_units']==2
    assert result['ci95']==[-1.,1.]


def test_summary_preserves_groups_and_clusters_shared_scenarios(tmp_path):
    import json
    from tools.compare_v338_v339_blue import summarize
    rows=[]
    for training_seed in (1,2):
        for episode_seed in (30001000,30001001):
            for version in ('3.38','3.39'):
                rows.append(dict(algorithm='MAPPO',training_seed=training_seed,
                    episode_seed=episode_seed,target_environment=version,red_win=int(version=='3.38'),
                    episode_return=1.,episode_length=100,blue_attack_kills=0,red_attack_kills=0,
                    overshoot=0,blue_speed_histogram=[0,1,0]))
    summarize(rows,tmp_path)
    result=json.loads((tmp_path/'summary.json').read_text())
    overall=next(r for r in result['paired_deltas'] if r['level']=='all')
    assert overall['metrics']['red_win']['delta']==-1
    assert overall['metrics']['red_win']['episodes']==4
    assert overall['metrics']['red_win']['bootstrap_scenario_units']==2


def test_batched_audit_keeps_independent_episode_results():
    from tools.compare_v338_v339_blue import episode,episode_batch
    class ZeroPolicy:
        def act(self,obs,alive,deterministic=True):
            return np.zeros((*obs.shape[:-1],3),np.float32)
    env,_=configs('mappo')
    policy=ZeroPolicy()
    from tools.combat_visualization import RecordingCombatEnv
    original_reset=RecordingCombatEnv.reset
    def short_reset(e,seed=None):
        result=original_reset(e,seed)
        e.max_steps=3  # Runtime unit fixture; the strict formal YAML remains valid.
        return result
    with patch.object(RecordingCombatEnv,'reset',short_reset):
        expected=[episode(policy,env,seed) for seed in (30001000,30001001)]
        assert episode_batch(policy,env,[30001000,30001001])==expected


def test_process_audit_collectors_match_local_canonical_episodes():
    from tools.compare_v338_v339_blue import episode_batch,parallel_episode_batch
    class ZeroPolicy:
        def act(self,obs,alive,deterministic=True):
            return np.zeros((*obs.shape[:-1],3),np.float32)
    env,_=configs('mappo')
    seeds=[30001000,30001001]
    expected=episode_batch(ZeroPolicy(),env,seeds)
    assert parallel_episode_batch(ZeroPolicy(),env,seeds,2)==expected


@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('version',['338','339'])
def test_cuda_readiness_checkpoint_artifacts_both_directions(name,version):
    import torch
    assert torch.cuda.is_available(),'CUDA mandatory for checkpoint validation'
    path=ROOT/f'outputs/v339_training_readiness/{name}_v{version}/latest.pt'
    if not path.exists():
        pytest.skip('run tools/verify_v339_training.py to create bounded CUDA integration artifacts')
    state=torch.load(path,map_location='cuda',weights_only=False)
    env,ac=configs(name,version)
    validate(state,env,ac)
    other='338' if version=='339' else '339'
    env,ac=configs(name,other)
    with pytest.raises((ValueError,RuntimeError),match='environment|family|config_sha256'):
        validate(state,env,ac)


@pytest.mark.parametrize('key',['environment_config_sha256','algorithm_config_sha256'])
def test_v339_mappo_rejects_missing_or_wrong_fingerprints(key):
    import torch
    assert torch.cuda.is_available(),'CUDA mandatory'
    path=ROOT/'outputs/v339_training_readiness/mappo_v339/latest.pt'
    if not path.exists():
        pytest.skip('bounded CUDA readiness checkpoint not generated')
    state=torch.load(path,map_location='cuda',weights_only=False)
    env,ac=configs('mappo')
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
    for validator in (validate_checkpoint_for_evaluation,validate_checkpoint_for_resume):
        for replacement in (None,'wrong'):
            changed=dict(state,extra=dict(state['extra']))
            if replacement is None:
                del changed['extra'][key]
            else:
                changed['extra'][key]=replacement
            with pytest.raises(RuntimeError,match=key):
                validator(changed,env,ac)


@pytest.mark.parametrize('name',['mappo_ca','ea_mappo','eaa_mappo'])
def test_eaa_family_cannot_be_forged_by_only_changing_family(name):
    import torch
    assert torch.cuda.is_available(),'CUDA mandatory'
    path=ROOT/f'outputs/v339_training_readiness/{name}_v338/latest.pt'
    if not path.exists():
        pytest.skip('bounded CUDA readiness checkpoint not generated')
    state=torch.load(path,map_location='cuda',weights_only=False)
    state['checkpoint_family']='eaa_mappo_v339'
    env,ac=configs(name)
    with pytest.raises(RuntimeError,match='environment_version'):
        validate(state,env,ac)
