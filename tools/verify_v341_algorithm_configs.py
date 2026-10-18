"""Bounded CUDA readiness with formal MAAC/EA/EAA networks and frozen configs.

Reuses v339 readiness collection, update, save/load, deterministic evaluation and
cross-version checks. Adds actual gradient/attention/optimizer instrumentation.
"""
from pathlib import Path
from unittest.mock import patch
import argparse
import json
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.verify_v339_training import run,config,validate
from algorithm.maac.runner import MAACTrainingRunner
from algorithm.eaa_mappo.runner import EAAMAPPOTrainingRunner

NAMES=('maac','ea_mappo','eaa_mappo')


def instrument(original,proof):
    def wrapped(runner):
        trainer=runner.trainer
        assert not trainer.smoke if hasattr(trainer,'smoke') else not runner.smoke
        before={side:{name:p.detach().clone() for name,p in getattr(trainer,side).named_parameters()}
                for side in ('actor','critic')}
        grads={side:{} for side in ('actor','critic')};handles=[]
        def hook_for(side,name):
            def hook(g):
                if not torch.isfinite(g).all():raise FloatingPointError(side+'/'+name+' invalid gradient')
                entry=grads[side].setdefault(name,dict(calls=0,max_abs=0.))
                entry['calls']+=1;entry['max_abs']=max(entry['max_abs'],float(g.abs().max()))
                return g
            return hook
        for side in ('actor','critic'):
            for name,p in getattr(trainer,side).named_parameters():handles.append(p.register_hook(hook_for(side,name)))
        try:
            if isinstance(runner,MAACTrainingRunner):
                t=runner.config['training']
                assert runner.replay.capacity==t['replay_capacity']==1000000
                assert runner.batch_size==1024 and runner.learning_starts==1024
                assert runner.update_interval==100 and trainer.baseline_samples==16
                assert trainer.actor.backbone[0].out_features==128 and trainer.critic.heads==4
                effective=dict(replay_capacity=runner.replay.capacity,minibatch_size=runner.batch_size,
                    learning_starts=runner.learning_starts,sampled_steps_per_update=runner.update_interval,
                    baseline_samples=trainer.baseline_samples,hidden_dim=128,attention_heads=4)
            else:
                assert trainer.actor_hidden_dim==trainer.critic_hidden_dim==256
                assert runner.rollout_steps==256 and trainer.ppo_epochs==10 and trainer.minibatch_size==512
                assert trainer.actor.entity_embedding_dim==64 and trainer.actor.attention_heads==4
                effective=dict(actor_hidden=256,critic_hidden=256,entity_embedding=64,entity_heads=4,
                    critic_type=trainer.critic_type,critic_heads=trainer.attention_heads if trainer.critic_type=='attention' else None,
                    rollout_steps=runner.rollout_steps,ppo_epochs=trainer.ppo_epochs,minibatch_size=trainer.minibatch_size)
            result=original(runner)
            changes={};attention={}
            for side in ('actor','critic'):
                params=dict(getattr(trainer,side).named_parameters())
                assert all(torch.isfinite(p).all() for p in params.values())
                diffs={name:float((p.detach()-before[side][name]).square().sum()) for name,p in params.items()}
                changes[side]=sum(diffs.values())**.5
                assert changes[side]>0 and grads[side] and any(g['max_abs']>0 for g in grads[side].values())
                for state in getattr(trainer,side+'_optimizer').state.values():
                    assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor))
                att_names=[n for n in params if n.split('.')[0] in ('wq','wk','wv','query','key','value')]
                if att_names:
                    assert all(n in grads[side] for n in att_names)
                    assert any(grads[side][n]['max_abs']>0 for n in att_names)
                    assert sum(diffs[n] for n in att_names)>0
                    attention[side]=dict(parameters=att_names,change_l2=sum(diffs[n] for n in att_names)**.5,
                                         gradients_finite=True,nonzero_gradient=True)
            proof.update(effective=effective,parameter_change_l2=changes,gradient_parameters=grads,
                attention=attention,parameters_gradients_optimizer_states_finite=True,
                actor_class=type(trainer.actor).__name__,critic_class=type(trainer.critic).__name__,
                parameter_counts={side:sum(p.numel() for p in getattr(trainer,side).parameters()) for side in ('actor','critic')},
                ppo_updates=getattr(trainer,'ppo_update_count',None))
            return result
        finally:
            for h in handles:h.remove()
    return wrapped


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/v341_algorithm_config_readiness')
    args=p.parse_args();out=args.output_dir.resolve()
    if out.exists() and any((out/f'{n}_v341').exists() for n in NAMES):raise FileExistsError('readiness algorithm directories must be fresh')
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=True);results=[]
    for name in NAMES:
        cls=MAACTrainingRunner if name=='maac' else EAAMAPPOTrainingRunner
        proof={}
        original_init=cls.__init__
        def formal_workers(self,*args,**kwargs):
            # Keep even the formal worker count: only budget/evaluation scope shrink.
            kwargs.update(num_envs=12 if name=='maac' else 16,
                          total_sampled_steps=1200 if name=='maac' else 4096)
            original_init(self,*args,**kwargs)
        with patch.object(cls,'__init__',formal_workers),patch.object(cls,'run',instrument(cls.run,proof)):
            result=run(name,'341',out)
        result.update(instrumentation=proof,formal_3m_training_started=False,
                      cuda_name=torch.cuda.get_device_name(),smoke_flag=False)
        (out/f'{name}_v341'/'instrumented_result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        results.append(result)
        (out/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
        print('READINESS PASS:',name,flush=True)
    env=config('combat_environment','341');rejections=[]
    states={name:torch.load(out/f'{name}_v341/latest.pt',map_location='cuda',weights_only=False) for name in NAMES}
    for source in ('ea_mappo','eaa_mappo'):
        target='eaa_mappo' if source=='ea_mappo' else 'ea_mappo'
        try:validate(states[source],env,config(target+'_4v4','341'))
        except (RuntimeError,ValueError) as e:rejections.append(dict(source=source,target=target,error=str(e)))
        else:raise AssertionError('attention variants accepted each other checkpoint')
    from algorithm.maac.protocol import validate_checkpoint_for_resume
    try:validate_checkpoint_for_resume(states['maac'],env,config('maac_4v4','341'))
    except RuntimeError as e:
        assert 'exact resume unsupported' in str(e);resume=str(e)
    else:raise AssertionError('MAAC exact resume unexpectedly enabled')
    (out/'checkpoint_isolation.json').write_text(json.dumps(dict(cross_variant_rejections=rejections,
        maac_exact_resume_rejection=resume,replay_buffer_in_checkpoint=False),indent=2),encoding='utf-8')
    print('All three bounded CUDA runs and checkpoint isolation passed',flush=True)


if __name__=='__main__':main()
