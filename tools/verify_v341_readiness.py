"""CUDA formal initialization plus bounded 4096-step, one-update integration smoke.

Preserves formal 256-wide networks and PPO settings; never calls the 3M run loop.
Only this task's diagnostic checkpoint is loaded/evaluated.
"""
from pathlib import Path
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
from algorithm.mappo.factory import build_mappo_trainer
from algorithm.mappo.evaluation import evaluate_mappo_checkpoint
from algorithm.common.checkpoint import validate_checkpoint_for_resume,validate_checkpoint_for_evaluation
from algorithm.common.protocol import config_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/v341_hit_calibration_preflight/readiness')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    torch.set_num_threads(1)
    env=load_config(ROOT/'configs/combat_environment_v341.yaml')
    cfg=yaml.safe_load((ROOT/'configs/mappo_4v4_v341.yaml').read_text())
    formal=MAPPOTrainingRunner(env,cfg,device='cuda',seed=1,output_dir=out/'formal_init')
    try:
        readiness=formal.startup_summary()
        assert formal.trainer.sampled_steps==0 and formal.observations.shape==(16,4,45)
        assert formal.total_sampled_steps==3000000 and formal.num_envs==16
        assert set(formal.vector.worker_environment_classes)=={'CombatEnvironmentV341'}
        assert set(formal.vector.worker_fixed_policy_classes)=={'DistanceTierNearestTargetBluePolicy'}
        readiness.update(cuda_name=torch.cuda.get_device_name(),sampled_steps=0,optimizer_updates=0,
            formal_3m_training_started=False,observation_shape=list(formal.observations.shape),
            worker_classes=formal.vector.worker_environment_classes,
            environment_config_sha256=config_sha256(env),algorithm_config_sha256=config_sha256(cfg))
        (out/'readiness.json').write_text(json.dumps(readiness,indent=2),encoding='utf-8')
    finally:formal.vector.close()
    # smoke=False retains the formal architecture/hyperparameters. The explicit
    # 4096 budget and separate output/checkpoint identify this as a diagnostic.
    runner=MAPPOTrainingRunner(env,cfg,device='cuda',seed=341,output_dir=out/'smoke_4096',
                              total_sampled_steps=4096,smoke=False)
    handles=[]
    try:
        trainer=runner.trainer
        before={side:[p.detach().clone() for p in getattr(trainer,side).parameters()] for side in ('actor','critic')}
        grads={side:dict(calls=0,max_abs=0.,all_finite=True) for side in ('actor','critic')}
        def hook_for(side):
            def hook(grad):
                if not torch.isfinite(grad).all():raise FloatingPointError(side+' non-finite gradient')
                grads[side]['calls']+=1;grads[side]['max_abs']=max(grads[side]['max_abs'],float(grad.abs().max()))
                return grad
            return hook
        for side in ('actor','critic'):
            for param in getattr(trainer,side).parameters():handles.append(param.register_hook(hook_for(side)))
        rollout=runner.collect_rollout() # exactly 256 vector steps * 16 environments
        for value in vars(rollout).values():assert np.isfinite(value).all()
        assert trainer.sampled_steps==4096 and trainer.ppo_update_count==0
        metrics=trainer.update(rollout)
        assert trainer.ppo_update_count==1 and np.isfinite(list(metrics.values())).all()
        changes={}
        for side in ('actor','critic'):
            params=list(getattr(trainer,side).parameters())
            assert all(torch.isfinite(p).all() for p in params)
            changes[side]=float(sum((p-q).square().sum() for p,q in zip(params,before[side])).sqrt())
            assert changes[side]>0 and grads[side]['calls']>0 and grads[side]['max_abs']>0
            for state in getattr(trainer,side+'_optimizer').state.values():
                assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor))
        checkpoint=out/'smoke_4096'/'diagnostic_4096.pt';runner.save_checkpoint(checkpoint)
        state=torch.load(checkpoint,map_location='cuda',weights_only=False)
        validate_checkpoint_for_resume(state,env,cfg);validate_checkpoint_for_evaluation(state,env,cfg)
        reloaded=build_mappo_trainer(cfg,'cuda');reloaded.load(checkpoint)
        for side in ('actor','critic'):
            for a,b in zip(getattr(trainer,side).parameters(),getattr(reloaded,side).parameters()):assert torch.equal(a,b)
        assert reloaded.sampled_steps==4096 and reloaded.ppo_update_count==1
        obs=runner.observations;alive=runner.alive_masks
        np.testing.assert_array_equal(trainer.act(obs,alive,deterministic=True),reloaded.act(obs,alive,deterministic=True))
        rejected=[]
        for version in ('339','340'):
            old_env=load_config(ROOT/f'configs/combat_environment_v{version}.yaml')
            old_cfg=yaml.safe_load((ROOT/f'configs/mappo_4v4_v{version}.yaml').read_text())
            for check in (validate_checkpoint_for_resume,validate_checkpoint_for_evaluation):
                try:check(state,old_env,old_cfg)
                except RuntimeError as e:
                    assert 'environment_version' in str(e);rejected.append(version+'/'+check.__name__)
                else:raise AssertionError('cross-version checkpoint accepted')
        evaluation=evaluate_mappo_checkpoint(checkpoint,cfg,env,'cuda',range(10000000,10000002))
        def check_finite(value):
            if isinstance(value,dict):
                for v in value.values():check_finite(v)
            elif isinstance(value,(list,tuple)):
                for v in value:check_finite(v)
            elif isinstance(value,(float,int)):assert np.isfinite(value)
        check_finite(evaluation)
        (out/'smoke_evaluation.json').write_text(json.dumps(evaluation,indent=2),encoding='utf-8')
        result=dict(device='cuda',cuda_name=torch.cuda.get_device_name(),purpose='bounded integration smoke only',
            formal_3m_training_started=False,smoke_sampled_steps=trainer.sampled_steps,
            vector_steps=trainer.vector_steps,ppo_updates=trainer.ppo_update_count,
            actor_updates=trainer.actor_update_count,critic_updates=trainer.critic_update_count,
            network_widths=[trainer.actor_hidden_dim,trainer.critic_hidden_dim],num_envs=runner.num_envs,
            rollout_shape=list(rollout.observations.shape),gradients=grads,parameter_change_l2=changes,
            all_parameters_gradients_optimizer_states_metrics_finite=True,
            checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_reload_parameters_exact=True,
            deterministic_actions_exact_after_reload=True,cross_version_rejections=rejected,
            test_evaluation_episodes=2,old_models_loaded=False,
            checkpoint_training_smoke_flag=False,
            smoke_flag_explanation='False preserves full formal architecture; explicit 4096 target, diagnostic filename and directory isolate this smoke',
            metrics=metrics)
        (out/'smoke_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2),flush=True)
    finally:
        for handle in handles:handle.remove()
        runner.vector.close()


if __name__=='__main__':main()
