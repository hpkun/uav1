"""4800-transition CUDA audit: full v5 architecture, batch and 24 workers.

Only runtime budget shrinks; no smoke, historical weights or 3M run is used.
"""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import argparse
import json
import sys
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.madsac.runner import MADSACTrainingRunner
from algorithm.madsac.factory import build_madsac_trainer
from algorithm.madsac.protocol import validate_madsac_checkpoint
from algorithm.madsac.evaluation import evaluate_madsac
from tools.validate_madsac_v4 import expected_counts
import algorithm.madsac.trainer as module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/madsac_v341_paper_audit/readiness')
    args=p.parse_args();out=args.output_dir
    if out.exists() and any(out.iterdir()):raise FileExistsError('readiness directory must be fresh')
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=True)
    e=yaml.safe_load((ROOT/'configs/combat_environment_v341.yaml').read_text())
    c=yaml.safe_load((ROOT/'configs/madsac_4v4_v341.yaml').read_text());snapshot=deepcopy(c)
    for name,value in (('env_config.yaml',e),('algorithm_config.yaml',c)):
        (out/name).write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
    runner=MADSACTrainingRunner(e,c,num_envs=24,total_sampled_steps=4800,device='cuda',seed=3,output_dir=out,smoke=False)
    t=runner.trainer;names=('actor','critic1','critic2','target_actor','target_critic1','target_critic2')
    before={n:{k:v.detach().clone() for k,v in getattr(t,n).state_dict().items()} for n in names}
    actual=dict(actor=0,critic1=0,critic2=0,samples=0);grads={n:dict(calls=0,max_abs=0.) for n in names[:3]};handles=[];cycles=[]
    def grad_hook(name):
        def hook(g):
            if not torch.isfinite(g).all():raise FloatingPointError(name+' gradient is nonfinite')
            grads[name]['calls']+=1;grads[name]['max_abs']=max(grads[name]['max_abs'],float(g.abs().max()))
            return g
        return hook
    for name in names[:3]:
        for param in getattr(t,name).parameters():handles.append(param.register_hook(grad_hook(name)))
        def step_hook(opt,args,kwargs,key=name):actual[key]+=1
        handles.append(getattr(t,name+'_optimizer').register_step_post_hook(step_hook))
    original_sample=runner.replay.sample;original_update=runner.maybe_update
    def sample(size):
        assert size==1024;actual['samples']+=1;return original_sample(size)
    def update():
        result=original_update()
        if result:cycles.append(dict(vector_step=t.vector_steps,sampled_steps=t.sampled_steps,**t.counters()))
        return result
    startup=runner.startup_summary();assert startup['hidden_dim']==256 and startup['attention_heads']==2
    assert runner.observations.shape==(24,4,45) and len(runner.evaluation_seeds)==20
    assert set(runner.vector.worker_environment_classes)=={'CombatEnvironmentV341'}
    (out/'readiness_run_config.json').write_text(json.dumps(dict(startup,formal_config_budget=3000000,
        bounded_readiness_budget=4800,training_seed=3,formal_3m_training_started=False),indent=2),encoding='utf-8')
    try:
        with patch.object(runner.replay,'sample',side_effect=sample),patch.object(runner,'maybe_update',side_effect=update), \
             patch.object(module,'polyak_update',wraps=module.polyak_update) as polyak:
            summary=runner.run();actual['network_polyak_calls']=polyak.call_count
        expected=expected_counts(4800)
        for k,v in expected.items():assert summary[k]==v,(k,summary[k],v)
        assert actual==dict(actor=64,critic1=128,critic2=128,samples=192,network_polyak_calls=48),actual
        assert [r['vector_step'] for r in cycles]==list(range(43,201,5))
        delta={}
        for name in names:
            state=getattr(t,name).state_dict();assert all(torch.isfinite(v).all() for v in state.values())
            delta[name]=sum(float((state[k]-before[name][k]).square().sum()) for k in state)**.5
            assert delta[name]>0,(name,delta)
        for name in names[:3]:
            assert grads[name]['calls']>0 and grads[name]['max_abs']>0
            for state in getattr(t,name+'_optimizer').state.values():
                assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor))
        rng=t.policy_generator.get_state().clone();global_rng=torch.get_rng_state().clone()
        cuda_rng=[s.clone() for s in torch.cuda.get_rng_state_all()]
        stochastic=runner.evaluate() # original 20 fixed scenarios, original stochastic protocol
        assert torch.equal(rng,t.policy_generator.get_state()) and torch.equal(global_rng,torch.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
        path=out/'final.pt';runner.save_checkpoint(path)
        state=torch.load(path,map_location='cuda',weights_only=False)
        validate_madsac_checkpoint(state,e,c,expected_training_seed=3)
        reloaded=build_madsac_trainer(c,'cuda',seed=19);reloaded.load_for_evaluation(path)
        for name in names:
            for a,b in zip(getattr(t,name).state_dict().values(),getattr(reloaded,name).state_dict().values()):
                assert torch.equal(a,b)
        deterministic=evaluate_madsac(t,e,[31001000,31001001],'deterministic')
        after=evaluate_madsac(reloaded,e,[31001000,31001001],'deterministic')
        assert deterministic==after
        # Evaluation on the original 20 stochastic seeds must also reproduce exactly.
        stochastic_reload=evaluate_madsac(reloaded,e,runner.evaluation_seeds,'stochastic',runner.evaluation_policy_seed)
        assert all(stochastic[k]==v for k,v in stochastic_reload.items())
        rejections={}
        for fault in ('version','seed','env_hash','algorithm_hash','v3','v4'):
            forged=deepcopy(state)
            if fault=='version':forged['extra']['environment_version']='3.39'
            elif fault=='seed':forged['extra']['training_seed']=0
            elif fault=='env_hash':forged['extra']['environment_config_sha256']='invalid'
            elif fault=='algorithm_hash':forged['extra']['algorithm_config_sha256']='invalid'
            else:forged['implementation_version']=int(fault[1:])
            try:validate_madsac_checkpoint(forged,e,c,expected_training_seed=3)
            except RuntimeError as exc:rejections[fault]=str(exc)
            else:raise AssertionError('forged checkpoint accepted: '+fault)
        try:reloaded.resume(path)
        except RuntimeError as exc:rejections['exact_resume']=str(exc)
        else:raise AssertionError('exact resume enabled')
        assert c==snapshot
        result=dict(expected=expected,actual_optimizer_hooks=actual,gradient_hooks=grads,parameter_change_l2=delta,
            counters_match=True,seed=3,config_seed=c['training']['seed'],num_envs=24,hidden_dim=256,heads=2,batch=1024,
            replay_size=len(runner.replay),schedule=[100,4,2],cycles=cycles,stochastic_evaluation_20=stochastic,
            stochastic_checkpoint_reload_exact=True,deterministic_test_evaluation_2=deterministic,
            six_network_checkpoint_reload_exact=True,evaluation_training_and_global_rng_unchanged=True,
            checkpoint_rejections=rejections,finite_optimization_metrics=runner.last_metrics,
            formal_3m_training_started=False,old_models_loaded=False,smoke=False,
            gpu=torch.cuda.get_device_name(),summary=summary)
        (out/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(dict(expected=expected,actual=actual,changes=delta,stochastic_win_rate=stochastic['win_rate'],
                             deterministic_test_win_rate=deterministic['win_rate']),indent=2),flush=True)
    finally:
        for h in handles:h.remove()
        runner.vector.close()


if __name__=='__main__':main()
