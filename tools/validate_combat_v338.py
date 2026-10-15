"""CUDA configuration/checkpoint/evaluation smoke; no optimizer or training run."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from copy import deepcopy
import argparse
import gc
import json
import numpy as np
import torch
import yaml
from env.factory import make_combat_environment


def run(output):
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory for checkpoint/evaluation smoke; no CPU fallback')
    torch.set_num_threads(1)
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'refusing nonempty smoke output {out}')
    out.mkdir(parents=True,exist_ok=True)
    ec=yaml.safe_load((ROOT/'configs/combat_environment_v338.yaml').read_text(encoding='utf-8'))
    old_ec=yaml.safe_load((ROOT/'configs/combat_environment_v337.yaml').read_text(encoding='utf-8'))
    rows={}
    for name in ('mappo','madsac','maac'):
        cfg=yaml.safe_load((ROOT/f'configs/{name}_4v4_v338.yaml').read_text(encoding='utf-8'))
        old_cfg=yaml.safe_load((ROOT/f'configs/{name}_4v4_v337.yaml').read_text(encoding='utf-8'))
        directory=out/name;directory.mkdir()
        checkpoint=directory/'initial_untrained.pt'
        runner=None
        if name=='mappo':
            from algorithm.mappo.runner import MAPPOTrainingRunner
            from algorithm.common.checkpoint import validate_checkpoint_for_evaluation as validator
            runner=MAPPOTrainingRunner(ec,cfg,num_envs=1,total_sampled_steps=24,device='cuda',output_dir=directory,smoke=False)
            trainer=runner.trainer
            runner.save_checkpoint(checkpoint)
            loader=trainer.load
        elif name=='madsac':
            from algorithm.madsac.runner import MADSACTrainingRunner
            from algorithm.madsac.protocol import validate_madsac_checkpoint as validator
            from algorithm.madsac.trainer import MADSAC_IMPL_VERSION
            assert MADSAC_IMPL_VERSION==5
            runner=MADSACTrainingRunner(ec,cfg,num_envs=1,total_sampled_steps=24,device='cuda',output_dir=directory,smoke=False)
            trainer=runner.trainer
            runner.save_checkpoint(checkpoint)
            loader=trainer.load_for_evaluation
        else:
            from algorithm.maac.trainer import MAACTrainer
            from algorithm.maac.protocol import validate_checkpoint as validator
            trainer=MAACTrainer(ec,cfg,device='cuda',seed=0,smoke=False)
            trainer.save(checkpoint)
            loader=trainer.load
        try:
            state=torch.load(checkpoint,map_location='cuda',weights_only=False)
            validator(state,ec,cfg)
            loader(checkpoint)
            rejected={}
            # Both identities are explicit even though old and new observations share 45D.
            old_state=deepcopy(state);old_state['extra']['environment_version']='3.37'
            for label,s,e,c in (('v337_checkpoint_on_v338',old_state,ec,cfg),
                                ('v338_checkpoint_on_v337',state,old_ec,old_cfg)):
                try:validator(s,e,c)
                except RuntimeError as exc:rejected[label]=str(exc)
                else:raise AssertionError(f'{name} accepted crossed observation semantics')
            env=make_combat_environment(ec);obs,_=env.reset(33800)
            for _ in range(16):
                if name=='madsac':actions=trainer.act(obs[None],env.red_alive_mask[None],True)[0]
                else:actions=trainer.act(obs,env.red_alive_mask,True)
                assert actions.shape==(4,3) and np.isfinite(actions).all()
                tensor=torch.as_tensor(obs[None],device='cuda',dtype=torch.float32)
                alive=torch.as_tensor(env.red_alive_mask[None],device='cuda',dtype=torch.float32)
                with torch.no_grad():
                    if name=='mappo':q=trainer.critic(tensor,alive)
                    elif name=='madsac':q=trainer.critic1(tensor,torch.as_tensor(actions[None],device='cuda',dtype=torch.float32),alive)
                    else:q=trainer.critic(tensor,torch.as_tensor(actions[None],device='cuda',dtype=torch.float32),alive)
                assert torch.isfinite(q).all()
                obs,_,term,trunc,_=env.step(actions)
                assert obs.shape==(4,45)
                if term or trunc:obs,_=env.reset(33801)
            if name=='madsac':
                from algorithm.madsac.evaluation import evaluate_madsac
                evaluation=evaluate_madsac(trainer,ec,[10000000],mode='deterministic')
            else:
                from algorithm.common.evaluator import evaluate
                evaluation=evaluate(trainer,ec,[10000000])
            assert evaluation['evaluation_episodes']==1
            optimizers=[v for k,v in vars(trainer).items() if k.endswith('_optimizer')]
            assert all(not optimizer.state for optimizer in optimizers),'smoke must not optimize'
            rows[name]=dict(device=str(next(trainer.actor.parameters()).device),observation_dim=45,
                checkpoint_protocol='PASS',checkpoint_save_load='PASS',cross_version_rejections=rejected,
                reset_step_forward_count=16,optimizer_steps=0,evaluation=evaluation,
                implementation_version=state.get('implementation_version',state.get('mappo_impl_version')))
            print(f'[PASS] {name} CUDA 45D actor/critic, protocol, load, 16 steps, 1 episode eval',flush=True)
        finally:
            if runner is not None:runner.vector.close()
        del state,old_state,loader,trainer,runner
        gc.collect();torch.cuda.empty_cache()
    result=dict(environment_version='3.38',gpu=torch.cuda.get_device_name(0),python=sys.executable,
                algorithms=rows,formal_training_started=False,optimizer_steps=0)
    (out/'smoke.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='outputs/v338_validation/cuda_smoke')
    run(parser.parse_args().output_dir)
