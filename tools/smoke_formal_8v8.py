"""Bounded CUDA acceptance of the four unchanged formal PPO protocols.

Exactly 512 sampled steps per algorithm and two deterministic episodes after
checkpoint reload. Existing --smoke modes reduce MAPPO/STEA widths; this tool
instead caps a formal-width runner and never starts the configured 3M budget.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import numpy as np
import torch
import yaml
from algorithm.train_mappo import TeeOutput, ensure_fresh_output_directory
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.rmappo.runner import RMAPPOTrainingRunner
from algorithm.ea_mappo.runner import EAMAPPOTrainingRunner
from algorithm.stea_mappo.runner import STEAMAPPOTrainingRunner
from algorithm.mappo.evaluation import evaluate_mappo_checkpoint
from algorithm.rmappo.evaluation import evaluate_rmappo_checkpoint
from algorithm.ea_mappo.evaluation import evaluate_ea_mappo_checkpoint
from algorithm.stea_mappo.evaluation import evaluate_stea_mappo_checkpoint
from algorithm.rmappo.protocol import require_cuda
from algorithm.common.protocol import config_sha256

ALGORITHMS={
    'mappo':('mappo_8v8_formal',MAPPOTrainingRunner,evaluate_mappo_checkpoint),
    'rmappo':('rmappo_8v8',RMAPPOTrainingRunner,evaluate_rmappo_checkpoint),
    'ea-mappo':('ea_mappo_8v8',EAMAPPOTrainingRunner,evaluate_ea_mappo_checkpoint),
    'stea-mappo':('stea_mappo_8v8_formal',STEAMAPPOTrainingRunner,evaluate_stea_mappo_checkpoint),
}


def fingerprint(state):
    digest=hashlib.sha256()
    for key,value in state.items():
        digest.update(key.encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_smokes(output, names, seed=31):
    require_cuda('cuda')
    output=Path(output)
    if not output.is_absolute(): output=ROOT/output
    if (output/'summary.json').exists():
        raise RuntimeError('Use a fresh validation output directory; results will not be overwritten')
    env_path=ROOT/'configs/combat_environment_v24.yaml'
    env_bytes=env_path.read_bytes()
    env=yaml.safe_load(env_bytes)
    records={};reference=None
    for name in names:
        stem,runner_class,evaluator=ALGORITHMS[name]
        config_path=ROOT/f'configs/{stem}.yaml'
        cfg=yaml.safe_load(config_path.read_text())
        run=output/name
        ensure_fresh_output_directory(run)
        runner=runner_class(env,cfg,num_envs=16,total_sampled_steps=512,device='cuda',
            seed=seed,output_dir=run,smoke=False)
        try:
            initial={k:v.detach().cpu().clone() for k,v in runner.trainer.critic.state_dict().items()}
            if reference is None: reference=initial
            else:
                assert initial.keys()==reference.keys()
                assert all(torch.equal(v,reference[k]) for k,v in initial.items())
            startup=runner.startup_summary()
            assert runner.trainer.critic.value_network[0].in_features==936
            assert startup['critic_parameter_count']==305921
            assert runner.trainer.ppo_epochs==10 and runner.trainer.minibatch_size==512
            assert runner.rollout_steps==256 and runner.trainer.target_kl==.015
            (run/'env_config.yaml').write_bytes(env_bytes)
            (run/'algorithm_config.yaml').write_bytes(config_path.read_bytes())
            (run/'run_config.json').write_text(json.dumps({**startup,
                'validation_only':True,'num_envs':16,'smoke':False,
                'configured_total_sampled_steps':cfg['training']['total_sampled_steps'],
                'validation_total_sampled_steps':512,'validation_evaluation_episodes':2,
                'environment_config_sha256':config_sha256(env),
                'algorithm_config_sha256':config_sha256(cfg)},indent=2))
            with (run/'train.log').open('w') as log, redirect_stdout(TeeOutput(sys.stdout,log)):
                print(f'[CUDA-SMOKE] algorithm={cfg["algorithm"]} | cap=512 | formal_dimensions_and_ppo=true',flush=True)
                print(runner.start_log_line(),flush=True)
                summary=runner.run()
                print(runner.done_log_line(summary),flush=True)
            (run/'run_summary.json').write_text(json.dumps(summary,indent=2))
        finally:
            runner.vector.close()
        assert summary['sampled_steps']==512 and summary['rollout_updates']==1
        metrics=summary['last_update_metrics']
        assert np.isfinite(list(metrics.values())).all()
        assert metrics['actor_grad_norm']>0 and metrics['critic_grad_norm']>0
        if name in ('rmappo','stea-mappo'):
            assert metrics['pre_update_ratio_max_abs_error']<1e-4
        assert all(torch.isfinite(p).all() for m in (runner.trainer.actor,runner.trainer.critic) for p in m.parameters())
        with torch.no_grad():
            values=runner.trainer.critic(torch.ones(2,8,104,device='cuda'),torch.ones(2,8,device='cuda'))
            assert torch.isfinite(values).all()
        checkpoint=run/'latest.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        extra=state['extra']
        assert tuple(extra[k] for k in ('observation_dim','action_dim','num_agents'))==(104,3,8)
        assert extra['environment_version']=='2.4' and extra['training_total_sampled_steps']==512
        seeds=range(int(cfg['implementation']['evaluation_seed_base']),int(cfg['implementation']['evaluation_seed_base'])+2)
        evaluation=evaluator(checkpoint,cfg,env,'cuda',seeds)
        assert evaluation['evaluation_episodes']==2 and evaluation['protocol_complete']
        (run/'reloaded_evaluation.json').write_text(json.dumps(evaluation,indent=2))
        records[name]={'algorithm':cfg['algorithm'],'sampled_steps':512,'num_envs':16,
            'formal_network_and_ppo':True,'configured_rollout_steps':256,'actual_final_rollout_steps':32,
            'actor_parameter_count':startup['actor_parameter_count'],
            'critic_parameter_count':startup['critic_parameter_count'],'critic_input_dim':936,
            'critic_initial_sha256':fingerprint(initial),'finite_gradients_parameters_and_values':True,
            'metrics':metrics,'evaluation_after_checkpoint_reload':evaluation}
        print(f'[SMOKE-DONE] {cfg["algorithm"]}: epochs={metrics["effective_ppo_epochs"]}, '
              f'last_KL={metrics["last_epoch_mean_kl"]:.7f}, eval_episodes=2',flush=True)
    assert env_path.read_bytes()==env_bytes,'v2.4 environment config must remain unchanged'
    result={'cuda_device':torch.cuda.get_device_name(),'same_seed_initial_critics_identical':True,
        'seed':seed,'formal_training_started':False,'records':records}
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm',choices=['all',*ALGORITHMS],default='all')
    parser.add_argument('--seed',type=int,default=31)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.seed<0: raise ValueError('seed must be nonnegative')
    run_smokes(args.output,list(ALGORITHMS) if args.algorithm=='all' else [args.algorithm],args.seed)


if __name__=='__main__': main()
