"""Short CUDA validation with unchanged formal v4 schedule and checkpoint protocol."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import argparse
import hashlib
import json
import math
import torch
import yaml
from algorithm.madsac.runner import MADSACTrainingRunner
from algorithm.madsac.factory import build_madsac_trainer
from algorithm.madsac.protocol import validate_madsac_checkpoint


def expected_counts(total, num_envs=24, batch=1024, steps_per_update=100, num_updates=4, policy_delay=2):
    if total % num_envs: raise ValueError('total must be divisible by num_envs')
    first=max(math.ceil(batch/num_envs),math.ceil(steps_per_update/num_envs))
    stride=math.ceil(steps_per_update/num_envs)
    vectors=total//num_envs
    cycles=max(0,1+(vectors-first)//stride)
    delayed=cycles//policy_delay
    rounds=cycles*num_updates
    return dict(sampled_steps=total,vector_steps=vectors,update_cycle_count=cycles,
        critic_update_count=rounds,actor_update_count=delayed*num_updates,
        critic_optimizer_steps=2*rounds,actor_optimizer_steps=delayed*num_updates,
        target_update_count=delayed,replay_sample_count_critic=rounds,
        replay_sample_count_actor=delayed*num_updates,
        transitions_since_update=(vectors-first-(cycles-1)*stride)*num_envs if cycles else total,
        first_update_sampled_steps=first*num_envs if cycles else None)


def run(output,total=4800):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is mandatory; no CPU fallback')
    torch.set_num_threads(1)
    out=Path(output)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'refusing non-empty run directory {out}')
    out.mkdir(parents=True,exist_ok=True)
    c=yaml.safe_load((ROOT/'configs/madsac_4v4_v337.yaml').read_text(encoding='utf-8'))
    e=yaml.safe_load((ROOT/'configs/combat_environment_v337.yaml').read_text(encoding='utf-8'))
    for name,value in (('algorithm_config.yaml',c),('env_config.yaml',e)):
        (out/name).write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
    runner=MADSACTrainingRunner(e,c,total_sampled_steps=total,device='cuda',output_dir=out,smoke=False)
    actual=dict(critic1=0,critic2=0,actor=0,sample=0,target_network_polyak_calls=0)
    handles=[]
    for name in ('critic1','critic2','actor'):
        def hook(opt,args,kwargs,key=name): actual[key]+=1
        handles.append(getattr(runner.trainer,name+'_optimizer').register_step_post_hook(hook))
    original_sample=runner.replay.sample
    def sample(*args,**kwargs):
        actual['sample']+=1
        return original_sample(*args,**kwargs)
    runner.replay.sample=sample
    import algorithm.madsac.trainer as trainer_module
    from unittest.mock import patch
    with patch.object(trainer_module,'polyak_update',wraps=trainer_module.polyak_update) as polyak:
        summary=runner.run()
        actual['target_network_polyak_calls']=polyak.call_count
    for handle in handles: handle.remove()
    expected=expected_counts(total)
    checks={k:summary[k]==v for k,v in expected.items()}
    checks.update(critic1_steps=actual['critic1']==expected['critic_update_count'],
        critic2_steps=actual['critic2']==expected['critic_update_count'],
        actor_steps=actual['actor']==expected['actor_update_count'],
        samples=actual['sample']==expected['replay_sample_count_critic']+expected['replay_sample_count_actor'],
        target_polyak_calls=actual['target_network_polyak_calls']==3*expected['target_update_count'])
    assert all(checks.values()),checks
    path=out/'final.pt'
    state=torch.load(path,map_location='cuda',weights_only=False)
    validate_madsac_checkpoint(state,e,c)
    restored=build_madsac_trainer(c,'cuda',seed=19)
    restored.load_for_evaluation(path)
    obs=torch.zeros(2,4,45,device='cuda')
    for name in ('actor','critic1','critic2','target_actor','target_critic1','target_critic2'):
        for a,b in zip(getattr(runner.trainer,name).state_dict().values(),getattr(restored,name).state_dict().values()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    legacy=ROOT/'outputs/madsac_v3_v326_seed1_3m/best_eval.pt'
    legacy_rejected=False
    if legacy.exists():
        try: restored.load_for_evaluation(legacy)
        except RuntimeError as exc:
            assert 'v3' in str(exc) and 'v4' in str(exc)
            legacy_rejected=True
        else: raise AssertionError('legacy v3 checkpoint accepted')
    # Keep the configured 20 evaluation seeds and unchanged combat environment.
    evaluation=runner.evaluate()
    result=dict(expected=expected,actual_optimizer_hooks=actual,checks=checks,summary=summary,
        checkpoint_save_load='PASS',old_v3_checkpoint_rejection=legacy_rejected,
        evaluation=evaluation,gpu=torch.cuda.get_device_name(0),python=sys.executable,
        long_training_started=False,only_total_steps_shortened=True)
    before=ROOT/'outputs/madsac_v4_validation/protected_hashes_before.json'
    if before.exists():
        hashes=json.loads(before.read_text(encoding='utf-8'))
        changed=[p for p,h in hashes.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
        assert not changed,changed
        result['protected_hashes_unchanged']=dict(files=len(hashes),changed=changed)
    (out/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result['expected'],indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='outputs/madsac_v4_validation/cuda_smoke')
    p.add_argument('--total-sampled-steps',type=int,default=4800)
    a=p.parse_args();run(a.output_dir,a.total_sampled_steps)
