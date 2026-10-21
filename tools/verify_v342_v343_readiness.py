"""Four bounded CUDA integrations, full configured networks and PPO parameters.

Each run samples exactly 4096 transitions (16x256) and one PPO update; the formal
3M loop is never entered. Checkpoints belong only to this diagnostic directory.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import json
import sys
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from env.config import load_config
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.ippo.runner import IPPOTrainingRunner
from algorithm.mappo.evaluation import evaluate_mappo_checkpoint
from algorithm.ippo.evaluation import evaluate_ippo_checkpoint
from algorithm.common.checkpoint import validate_checkpoint_for_resume,validate_checkpoint_for_evaluation
from algorithm.ippo.protocol import validate_checkpoint as validate_ippo
from algorithm.common.protocol import config_sha256
from tools.record_combat_episode import load_recording_policy,record


def finite(value):
    if isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(tuple,list)):
        for v in value:finite(v)
    elif isinstance(value,(float,int,np.ndarray,np.number)):
        assert np.isfinite(value).all(), 'non-finite diagnostic value'


def algorithm_config(algo,version):
    return yaml.safe_load((ROOT/f'configs/{algo}_4v4_v{version}.yaml').read_text(encoding='utf-8'))


def run(algo,version,out):
    ec=load_config(ROOT/f'configs/combat_environment_v{version}.yaml');ac=algorithm_config(algo,version)
    out.mkdir(parents=True,exist_ok=False)
    (out/'algorithm_config.yaml').write_text(yaml.safe_dump(ac,sort_keys=False),encoding='utf-8')
    (out/'env_config.yaml').write_text(yaml.safe_dump(ec,sort_keys=False),encoding='utf-8')
    cls=MAPPOTrainingRunner if algo=='mappo' else IPPOTrainingRunner
    runner=cls(ec,ac,device='cuda',seed=342,output_dir=out,total_sampled_steps=4096,smoke=False)
    handles=[]
    try:
        trainer=runner.trainer;dim=ac['network']['observation_dim']
        assert runner.observations.shape==(16,4,dim)
        assert set(runner.vector.worker_environment_classes)=={f'CombatEnvironmentV{version}'}
        assert set(runner.vector.worker_fixed_policy_classes)=={'DistanceTierNearestTargetBluePolicy'}
        assert trainer.actor_hidden_dim==trainer.critic_hidden_dim==256
        assert runner.rollout_steps==256 and trainer.ppo_epochs==10 and trainer.minibatch_size==512
        initial={side:[p.detach().clone() for p in getattr(trainer,side).parameters()] for side in ('actor','critic')}
        gradients={side:dict(calls=0,max_abs=0.) for side in ('actor','critic')}
        def gradient_hook(side):
            def hook(g):
                if not torch.isfinite(g).all():raise FloatingPointError(side+' invalid gradients')
                gradients[side]['calls']+=1;gradients[side]['max_abs']=max(gradients[side]['max_abs'],float(g.abs().max()))
                return g
            return hook
        for side in initial:
            for p in getattr(trainer,side).parameters():handles.append(p.register_hook(gradient_hook(side)))
        rollout=runner.collect_rollout()
        for value in vars(rollout).values():finite(value)
        assert trainer.sampled_steps==4096 and trainer.ppo_update_count==0
        metrics=trainer.update(rollout);finite(metrics)
        assert trainer.ppo_update_count==1
        change={}
        for side in initial:
            params=list(getattr(trainer,side).parameters())
            assert all(torch.isfinite(p).all() for p in params)
            change[side]=float(sum((a.detach()-b).square().sum() for a,b in zip(params,initial[side])).sqrt())
            assert change[side]>0 and gradients[side]['max_abs']>0
            for state in getattr(trainer,side+'_optimizer').state.values():
                assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor))
        runner.last_metrics=metrics;runner._write_rollout_metrics();finite(runner.summary())
        checkpoint=out/'diagnostic_4096.pt';runner.save_checkpoint(checkpoint)
        state=torch.load(checkpoint,map_location='cuda',weights_only=False)
        validators=(validate_checkpoint_for_resume,validate_checkpoint_for_evaluation) if algo=='mappo' else (validate_ippo,)
        for check in validators:check(state,ec,ac)
        reloaded,_,provenance=load_recording_policy(checkpoint,ec,ac,'cuda')
        for side in initial:
            for a,b in zip(getattr(trainer,side).parameters(),getattr(reloaded,side).parameters()):assert torch.equal(a,b)
        assert reloaded.sampled_steps==4096 and reloaded.ppo_update_count==1
        np.testing.assert_array_equal(trainer.act(runner.observations,runner.alive_masks,deterministic=True),
                                      reloaded.act(runner.observations,runner.alive_masks,deterministic=True))
        rejected=[]
        for other in ('341','342','343'):
            if other==version:continue
            other_ec=load_config(ROOT/f'configs/combat_environment_v{other}.yaml');other_ac=algorithm_config(algo,other)
            for check in validators:
                try:check(state,other_ec,other_ac)
                except (ValueError,RuntimeError) as ex:rejected.append(dict(version=other,validator=check.__name__,reason=str(ex)))
                else:raise AssertionError('cross-version checkpoint silently accepted')
        for key in ('environment_config_sha256','algorithm_config_sha256'):
            bad=deepcopy(state);bad['extra'].pop(key)
            for check in validators:
                try:check(bad,ec,ac)
                except RuntimeError:pass
                else:raise AssertionError('missing fingerprint silently accepted')
        evaluate=evaluate_mappo_checkpoint if algo=='mappo' else evaluate_ippo_checkpoint
        evaluation=evaluate(checkpoint,ac,ec,'cuda',range(10_000_000,10_000_002));finite(evaluation)
        (out/'evaluation.json').write_text(json.dumps(evaluation,indent=2,allow_nan=False),encoding='utf-8')
        metadata=record(reloaded,ec,20_000_342,out/'recording',dict(provenance,algorithm=algo))
        assert metadata['environment_version']==ec['environment_version']
        for event in metadata['events']:
            if event['type']=='fire_attempt':assert event['cone_pass']
        if algo=='ippo':
            from algorithm.ippo.diagnostics import compare_checkpoints
            diagnostic=compare_checkpoints([checkpoint],out/'diagnostic_observations.npz',ec,ac)
            (out/'fixed_observation_check.json').write_text(json.dumps(diagnostic,indent=2,allow_nan=False),encoding='utf-8')
        result=dict(algorithm=algo,version=ec['environment_version'],device='cuda',cuda_name=torch.cuda.get_device_name(),
            sampled_steps=4096,vector_steps=trainer.vector_steps,ppo_updates=trainer.ppo_update_count,
            actor_optimizer_steps=trainer.actor_update_count,critic_optimizer_steps=trainer.critic_update_count,
            observation_shape=list(rollout.observations.shape),gradients=gradients,parameter_change_l2=change,
            full_formal_architecture_and_hyperparameters=True,formal_3m_training_started=False,
            training_smoke_flag=False,scope='explicit 4096 budget; no call to formal run loop',
            checkpoint=str(checkpoint),reload_parameters_and_actions_exact=True,cross_version_rejections=rejected,
            missing_fingerprints_rejected=True,evaluation_episodes=2,recording_steps=metadata['episode_steps'],
            environment_config_sha256=config_sha256(ec),algorithm_config_sha256=config_sha256(ac),metrics=metrics)
        (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        print(f'{algo} v{version}: CUDA 4096 samples, 1 PPO update, reload/evaluation/recording PASS',flush=True)
        return result
    finally:
        for h in handles:h.remove()
        runner.vector.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('smoke output must be fresh')
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    results=[]
    for version in ('342','343'):
        for algo in ('mappo','ippo'):results.append(run(algo,version,args.output/f'{algo}_v{version}'))
    (args.output/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
