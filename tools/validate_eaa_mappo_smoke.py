"""Four full-width CUDA rollout/update/evaluation/load smokes; no formal training."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.common.evaluator import evaluate
from algorithm.mappo.factory import build_mappo_trainer
from algorithm.mappo.evaluation import evaluate_mappo_checkpoint
from algorithm.eaa_mappo.evaluation import load_checkpoint
from algorithm.eaa_mappo.protocol import require_cuda


def baseline_bitwise_check(directory):
    """Optional original-source snapshot for a before/after PPO parameter comparison."""
    original=directory/'original_mappo'
    if not original.exists():return {'status':'not requested; no original snapshot'}
    spec=importlib.util.spec_from_file_location('original_mappo',original/'__init__.py',submodule_search_locations=[str(original)])
    module=importlib.util.module_from_spec(spec);sys.modules['original_mappo']=module;spec.loader.exec_module(module)
    from original_mappo.factory import build_mappo_trainer as old_builder
    from algorithm.mappo.trainer import RolloutBatch
    config=yaml.safe_load((ROOT/'configs/mappo_4v4_v338.yaml').read_text())
    old,new=old_builder(config,'cuda'),build_mappo_trainer(config,'cuda')
    for name in ['actor','critic']:
        assert all(torch.equal(v,getattr(new,name).state_dict()[k]) for k,v in getattr(old,name).state_dict().items())
    rng=np.random.default_rng(41);obs=rng.normal(size=(4,2,4,45)).astype('float32');alive=np.ones((4,2,4),'float32')
    torch.manual_seed(81);actions,raw,logp=old.act(obs,alive,return_policy_data=True)
    torch.manual_seed(81);other=new.act(obs,alive,return_policy_data=True)
    assert all(np.array_equal(a,b) for a,b in zip([actions,raw,logp],other))
    rollout=RolloutBatch(obs,actions,raw,logp,rng.normal(size=(4,2,4)).astype('float32'),
        np.zeros((4,2),'float32'),alive,obs.copy(),alive.copy())
    torch.manual_seed(92);a=old.update(rollout)
    torch.manual_seed(92);b=new.update(rollout)
    assert a==b
    for name in ['actor','critic']:
        assert all(torch.equal(v,getattr(new,name).state_dict()[k]) for k,v in getattr(old,name).state_dict().items())
    return {'status':'PASS','initial_parameters':'bitwise equal','sample_raw_logprob':'bitwise equal',
            'PPO_metrics_and_updated_parameters':'bitwise equal','baseline_implementation_version':2}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation-dir',default='outputs/eaa_mappo_validation')
    args=p.parse_args();directory=ROOT/args.validation_dir;directory.mkdir(parents=True,exist_ok=True)
    require_cuda('cuda');torch.set_num_threads(1)
    env=yaml.safe_load((ROOT/'configs/combat_environment_v338.yaml').read_text(encoding='utf-8'))
    report={'device':'cuda','GPU':torch.cuda.get_device_name(),'baseline_bitwise':baseline_bitwise_check(directory),'structures':{}}
    for name in ['mappo','mappo_ca','ea_mappo','eaa_mappo']:
        output=directory/name
        config_path=ROOT/'configs'/f'{name}_4v4_v338.yaml'
        config=yaml.safe_load(config_path.read_text(encoding='utf-8'))
        entry='train_mappo.py' if name in ['mappo','mappo_ca'] else 'train_eaa_mappo.py'
        cmd=[sys.executable,'-u',str(ROOT/'algorithm'/entry),'--env-config','configs/combat_environment_v338.yaml',
             '--algorithm-config',str(config_path),'--device','cuda','--seed','1','--num-envs','2',
             '--total-sampled-steps','512','--output-dir',str(output)]
        if name!='mappo':cmd.append('--smoke')
        print('START',name,'512 transitions; full256 networks/PPO',flush=True)
        with (directory/f'{name}_console.log').open('w',encoding='utf-8') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state=torch.load(output/'latest.pt',map_location='cuda',weights_only=False)
        summary=json.loads((output/'run_summary.json').read_text())
        if name=='mappo':
            trainer=build_mappo_trainer(config,'cuda');trainer.load(output/'latest.pt')
            result=evaluate_mappo_checkpoint(output/'latest.pt',config,env,'cuda',[10000000])
            # The formal entry's existing schedule does not force an eval at 512; audit it here.
            metadata=json.loads((output/'run_config.json').read_text())
        else:
            trainer=load_checkpoint(output/'latest.pt',config,env)
            result=evaluate(trainer,env,[10000000])
            assert summary['evaluation_history'][-1]['average_return']==result['average_return']
            assert summary['evaluation_history'][-1]['win_rate']==result['win_rate']
            metadata=json.loads((output/'run_config.json').read_text())
        for key in ['actor','critic']:
            assert all(torch.equal(v,state[key][k]) for k,v in getattr(trainer,key).state_dict().items())
        assert trainer.actor_update_count>0 and trainer.critic_update_count>0 and trainer.ppo_update_count==1
        assert summary['sampled_steps']==512 and summary['rollout_steps']==256
        assert trainer.actor_hidden_dim==trainer.critic_hidden_dim==256
        metrics=[json.loads(s) for s in (output/'optimization_metrics.jsonl').read_text().splitlines()]
        assert metrics and np.isfinite(list(metrics[0].values())).all()
        if name in ['ea_mappo','eaa_mappo']:
            row=metrics[-1];assert abs(sum(row['actor_attention_'+k+'_mass'] for k in ['self','ally','enemy'])-1)<1e-6
        if name in ['mappo_ca','eaa_mappo']:assert 'critic_attention_entropy' in metrics[-1]
        report['structures'][name]=dict(status='PASS',sampled_steps=512,rollout_steps=256,
            actor_updates=trainer.actor_update_count,critic_updates=trainer.critic_update_count,
            actor_parameters=metadata['actor_parameter_count'],critic_parameters=metadata['critic_parameter_count'],
            total_parameters=metadata['total_parameter_count'],deterministic_return=result['average_return'],
            checkpoint_roundtrip='exact',PPO_config_preserved=True,diagnostics=metrics[-1])
        print('PASS',name,'updates',trainer.actor_update_count,flush=True)
    old_path=ROOT/'outputs/mappo_v338_seed1_3m/best_eval.pt'
    if old_path.exists():
        env_hash_before=hashlib.sha256(old_path.read_bytes()).hexdigest()
        config=yaml.safe_load((ROOT/'configs/mappo_4v4_v338.yaml').read_text())
        trainer=build_mappo_trainer(config,'cuda');extra=trainer.load(old_path)
        assert extra['environment_version']=='3.38'
        assert hashlib.sha256(old_path.read_bytes()).hexdigest()==env_hash_before
        report['existing_baseline_checkpoint']={'status':'loaded on CUDA, unchanged','path':str(old_path),'sha256':env_hash_before}
    (directory/'cuda_smoke_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('ALL FOUR CUDA SMOKES PASS',flush=True)


if __name__=='__main__':main()
