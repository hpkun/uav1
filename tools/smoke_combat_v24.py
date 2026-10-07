"""CUDA integration checks gated by the completed 1000-episode v2.4 audit."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit-report',type=Path,default=ROOT/'outputs/v24_combat_audit/v24/audit_report.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/v24_validation/cuda_smoke')
    args=parser.parse_args()
    if not args.audit_report.is_file():
        raise RuntimeError('Complete the v2.4 1000-episode difficulty audit before CUDA update smoke')
    report=json.loads(args.audit_report.read_text())
    # Read the actual episodes again, so the gate also uses current diagnostics.
    from tools.audit_combat_v24 import summarize
    from env.config import load_config
    episode_path=args.audit_report.parent/'episodes.jsonl'
    rows=[json.loads(line) for line in episode_path.read_text().splitlines()]
    report.update(summarize(rows,load_config(ROOT/'configs/combat_environment_v24.yaml')))
    if report['environment_version']!='2.4' or report['episodes']<1000 or not report['audit_passed']:
        raise RuntimeError('v2.4 1000-episode difficulty audit must pass before CUDA update smoke')
    # Imported only in the coordinator: spawned environment workers remain light.
    import numpy as np
    import torch
    import yaml
    from algorithm.mappo.runner import MAPPOTrainingRunner
    from algorithm.mappo.evaluation import evaluate_mappo_checkpoint
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation
    from algorithm.stea_mappo.runner import STEAMAPPOTrainingRunner
    from algorithm.stea_mappo.evaluation import evaluate_stea_mappo_checkpoint
    from algorithm.stea_mappo.protocol import validate_checkpoint
    from algorithm.stea_mappo.trainer import sequence_chunks
    from algorithm.common.protocol import config_sha256
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is mandatory for v2.4 smoke')
    torch.set_num_threads(1)
    env=yaml.safe_load((ROOT/'configs/combat_environment_v24.yaml').read_text())
    if report['config_sha256']!=config_sha256(env): raise RuntimeError('audited environment config changed')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    results={}
    for name,runner_type,validator,evaluator in (
        ('mappo',MAPPOTrainingRunner,validate_checkpoint_for_evaluation,evaluate_mappo_checkpoint),
        ('stea_mappo',STEAMAPPOTrainingRunner,validate_checkpoint,evaluate_stea_mappo_checkpoint)):
        cfg=yaml.safe_load((ROOT/f'configs/{name}_8v8.yaml').read_text())
        folder=args.output_dir/name
        with_runner=runner_type(env,cfg,num_envs=2,total_sampled_steps=512,device='cuda',seed=31,output_dir=folder,smoke=False)
        try:
            runner=with_runner
            rollout=runner.collect_rollout()
            assert rollout.observations.shape==(256,2,8,104)
            assert rollout.actions.shape==(256,2,8,3)
            assert np.isfinite(rollout.observations).all() and np.isfinite(rollout.rewards).all()
            if name=='stea_mappo':
                assert rollout.actor_hidden_states.shape==(256,2,8,128)
                assert not runner.actor_hidden_states[runner.alive_masks==0].any()
                tensors={key:runner.trainer.tensor(value) for key,value in vars(rollout).items()}
                packed=runner.trainer._packed(tensors,sequence_chunks(256,2,32))
                sequence_shape=list(packed['observations'].shape)
                assert sequence_shape==[16,32,8,104]
                # Both reset paths must erase arbitrary prior memory, including dead UAVs.
                actor=runner.trainer.actor
                observation=torch.as_tensor(runner.observations,device='cuda')
                alive=torch.as_tensor(runner.alive_masks,device='cuda')
                hidden=torch.randn(2,8,128,device='cuda')
                start=torch.ones(2,device='cuda')
                with torch.no_grad():
                    first=actor.distribution_step(observation,hidden,alive,start)
                    zero=actor.distribution_step(observation,torch.zeros_like(hidden),alive,start)
                assert torch.equal(first[0].mean,zero[0].mean) and torch.equal(first[1],zero[1])
                assert not first[1][alive==0].any()
            metrics=runner.trainer.update(rollout)
            assert np.isfinite(list(metrics.values())).all()
            gradients={key:bool(torch.isfinite(p.grad).all()) for key,p in runner.trainer.actor.named_parameters() if p.grad is not None}
            assert gradients and all(gradients.values())
            if name=='stea_mappo':
                required={key for key,p in runner.trainer.actor.named_parameters()
                    if p.requires_grad and ('attention' in key or 'gru' in key)}
                assert required and required.issubset(gradients),'Missing attention/GRU gradients'
            assert all(torch.isfinite(p.grad).all() for p in runner.trainer.critic.parameters() if p.grad is not None)
            checkpoint=folder/'latest.pt';runner.save_checkpoint(checkpoint)
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            validator(state,env,cfg)
            runner.trainer.load(checkpoint)
            result={'cuda_device':torch.cuda.get_device_name(),'startup':runner.startup_summary(),
                'observation_shape':list(rollout.observations.shape),'action_shape':list(rollout.actions.shape),
                'checkpoint':str(checkpoint),'metadata':state['extra'],
                'metrics':metrics,'actor_gradients_finite':gradients,
                'actor_parameter_count':sum(p.numel() for p in runner.trainer.actor.parameters()),
                'critic_parameter_count':sum(p.numel() for p in runner.trainer.critic.parameters())}
            if name=='stea_mappo':
                result.update(hidden_shape=list(rollout.actor_hidden_states.shape),
                    sequence_shape=sequence_shape,death_and_episode_hidden_reset_passed=True)
        finally:
            with_runner.vector.close()
        result['checkpoint_evaluation']=evaluator(checkpoint,cfg,env,'cuda',[30_000_000,30_000_001])
        results[name]=result
        (args.output_dir/'smoke_report.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({'algorithm':name,'sampled_steps':512,'metrics':metrics}),flush=True)

if __name__=='__main__': main()
