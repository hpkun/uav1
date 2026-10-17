"""Bounded CUDA readiness runs with FORMAL networks, hyperparameters and snapshots.

Only budget, worker count and number of evaluation episodes shrink. No YAML edits.
Never an exact-resume claim for replay algorithms or the EAA family.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from algorithm.common.protocol import config_sha256
from algorithm.common.evaluator import evaluate
from tools.record_combat_episode import load_recording_policy

NAMES = ('mappo','mappo_ca','ea_mappo','eaa_mappo','maac','maddpg','madsac')


def config(name, version):
    return yaml.safe_load((ROOT/f'configs/{name}_v{version}.yaml').read_text(encoding='utf-8'))


def validate(state, env, ac):
    name = ac['algorithm']
    if name == 'MAPPO':
        from algorithm.common.checkpoint import validate_checkpoint_for_evaluation, validate_checkpoint_for_resume
        validate_checkpoint_for_evaluation(state,env,ac)
        validate_checkpoint_for_resume(state,env,ac)
    elif name in ('MAPPO-CA','EA-MAPPO','EAA-MAPPO'):
        from algorithm.eaa_mappo.protocol import validate_checkpoint
        validate_checkpoint(state,env,ac)
    elif name == 'MAAC':
        from algorithm.maac.protocol import validate_checkpoint
        validate_checkpoint(state,env,ac)
    elif name == 'maddpg':
        from algorithm.maddpg.protocol import validate_checkpoint
        validate_checkpoint(state,env,ac)
    else:
        from algorithm.madsac.protocol import validate_madsac_checkpoint
        validate_madsac_checkpoint(state,env,ac,expected_training_seed=state['extra']['training_seed'])


def run(name,version,output):
    ec, ac = config('combat_environment',version), config(name+'_4v4',version)
    snapshot=deepcopy(ac)
    directory=output/f'{name}_v{version}'
    if directory.exists():
        raise FileExistsError(directory)
    if name == 'maddpg':
        from algorithm.maddpg.runner import MADDPGTrainingRunner
        runner=MADDPGTrainingRunner(ec,ac,num_envs=4,total_sampled_steps=10100,
            device='cuda',seed=1,output_dir=directory,smoke=False)
    else:
        directory.mkdir(parents=True)
        if name in ('mappo','mappo_ca','ea_mappo','eaa_mappo'):
            from algorithm.mappo.runner import MAPPOTrainingRunner
            from algorithm.eaa_mappo.runner import EAAMAPPOTrainingRunner
            cls=MAPPOTrainingRunner if name=='mappo' else EAAMAPPOTrainingRunner
            runner=cls(ec,ac,num_envs=2,total_sampled_steps=512,
                device='cuda',seed=1,output_dir=directory,smoke=False)
            runner.next_evaluation=runner.next_checkpoint=512
        elif name == 'maac':
            from algorithm.maac.runner import MAACTrainingRunner
            runner=MAACTrainingRunner(ec,ac,directory,num_envs=4,total_sampled_steps=1100,
                device='cuda',seed=1,smoke=False)
        else:
            from algorithm.madsac.runner import MADSACTrainingRunner
            runner=MADSACTrainingRunner(ec,ac,num_envs=4,total_sampled_steps=1200,
                device='cuda',seed=1,output_dir=directory,smoke=False)
    for field in ('evaluation_seeds','eval_seeds'):
        if hasattr(runner,field):
            setattr(runner,field,list(getattr(runner,field))[:1])
    try:
        worker_classes=runner.vector.worker_environment_classes.copy()
        for key,value in [('env_config.yaml',ec),('algorithm_config.yaml',ac)]:
            (directory/key).write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
        startup = runner.startup_summary() if hasattr(runner,'startup_summary') else dict(environment_version=ec['environment_version'],device='cuda')
        startup.update(readiness_smoke=True,formal_networks_and_hyperparameters=True,
            num_envs=runner.num_envs,
            bounded_sampled_steps=getattr(runner,'total_sampled_steps',getattr(runner,'total_steps',None)),
            evaluation_episodes_override=1,environment_config_sha256=config_sha256(ec),
            algorithm_config_sha256=config_sha256(ac))
        (directory/'readiness_run_config.json').write_text(json.dumps(startup,indent=2),encoding='utf-8')
        summary=runner.run()
    finally:
        runner.vector.close()
    assert ac==snapshot, 'smoke mutated formal config'
    path=directory/'latest.pt'
    state=torch.load(path,map_location='cuda',weights_only=False)
    validate(state,ec,ac)
    actor_count=state.get('actor_updates',state.get('actor_update_count',0))
    critic_count=state.get('critic_updates',state.get('critic_update_count',0))
    assert actor_count>0 and critic_count>0, (name,actor_count,critic_count)
    assert state['extra']['environment_config_sha256']==config_sha256(ec)
    assert state['extra']['algorithm_config_sha256']==config_sha256(ac)
    reloaded,_,provenance=load_recording_policy(path,ec,ac,'cuda')
    from env.combat_env import MultiUAVCombatEnv
    e=MultiUAVCombatEnv(ec);obs,_=e.reset(31001000)
    a=runner.trainer.act(obs[None],e.red_alive_mask[None],deterministic=True)
    b=reloaded.act(obs[None],e.red_alive_mask[None],deterministic=True)
    np.testing.assert_array_equal(a,b)
    assert np.isfinite(a).all()
    before=evaluate(runner.trainer,ec,[31001000])
    after=evaluate(reloaded,ec,[31001000])
    assert before==after
    assert np.isfinite(list(after.values())).all()
    metrics=runner.last_metrics
    assert metrics and all(np.isfinite(v) for v in metrics.values() if isinstance(v,(float,int)))
    for key in ('average_episode_r1_total','average_episode_r2_total','average_episode_r3_total',
                'average_episode_r4_total','average_episode_paper_reward_total',
                'average_episode_safety_total','average_episode_extended_reward_total',
                'average_blue_target_switches','red_ceiling_losses','red_fire_attempts'):
        assert key in after, key
    other='338' if version=='339' else '339'
    other_ec,other_ac=config('combat_environment',other),config(name+'_4v4',other)
    try:
        validate(state,other_ec,other_ac)
    except (ValueError,RuntimeError) as error:
        cross_rejection=str(error)
        assert any(s in cross_rejection for s in ('environment','family','config_sha256'))
    else:
        raise AssertionError('cross-version checkpoint accepted')
    result=dict(algorithm=name,environment_version=ec['environment_version'],device='cuda',
        sampled_steps=state['sampled_steps'],actor_updates=actor_count,critic_updates=critic_count,
        checkpoint=str(path),provenance=provenance,worker_classes=worker_classes,
        finite_actions=True,finite_losses=True,save_load_eval_exact=True,
        cross_version_rejection=cross_rejection,evaluation=after,optimization_metrics=metrics)
    (directory/'readiness_result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='outputs/v339_training_readiness')
    p.add_argument('--algorithms',nargs='+',choices=NAMES,default=NAMES)
    p.add_argument('--versions',nargs='+',choices=['338','339'],default=['338','339'])
    args=p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory; no CPU fallback')
    torch.set_num_threads(1)
    out=ROOT/args.output_dir;out.mkdir(parents=True,exist_ok=True)
    results=[]
    for name in args.algorithms:
        for version in args.versions:
            result=run(name,version,out)
            results.append(result)
            (out/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
            print(f'READINESS PASS: {name} v{version}',flush=True)


if __name__=='__main__':
    main()
