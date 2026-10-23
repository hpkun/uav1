"""CUDA-only, bounded formal-parameter updates and isolated historical audit."""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import io
import json
import subprocess
import sys
import zipfile
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from env.config import load_config
from algorithm.maac.runner import MAACTrainingRunner
from algorithm.maac.protocol import validate_checkpoint as validate_maac
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation as validate_mappo
from algorithm.common.evaluator import evaluate
from tools.verify_v342_v343_readiness import run as ppo_run
from tools.record_combat_episode import load_recording_policy, record
from tools.combat_replay_report import export_analysis
from tools.combat_trace import read_trace
from tools.audit_v342_v343_history import PROBE


def dump(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def configs(algo,v='348'):
    return load_config(ROOT/f'configs/combat_environment_v{v}.yaml'), yaml.safe_load(
        (ROOT/f'configs/{algo}_4v4_v{v}.yaml').read_text())


def history(out):
    baseline=out/'baseline_head';baseline.mkdir()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    raw=subprocess.check_output(['git','archive','--format=zip','HEAD','env','configs','algorithm'],cwd=ROOT)
    with zipfile.ZipFile(io.BytesIO(raw)) as z: z.extractall(baseline)
    probe=PROBE.replace("'31','32','37','39','330','332','336','341'","'338','342','344','346','347'")
    probe=probe.replace('digest.update(bytes((term,trunc)))', '''
                for side in ('red','blue'):
                    if hasattr(e,'dwell'):
                        for name in ('targets','counts','sampled_targets','sampled_counts','completed_this_step'):
                            digest.update(getattr(e.dwell[side],name).tobytes())
                    digest.update(getattr(e,side+'_fire_cooldown_steps').tobytes())
                for key in sorted(info):
                    if key.endswith('_rewards'): digest.update(info[key].tobytes())
                digest.update(bytes((term,trunc)))'''.strip())
    current=json.loads(subprocess.check_output([sys.executable,'-c',probe],cwd=ROOT,text=True))
    old=json.loads(subprocess.check_output([sys.executable,'-c',probe],cwd=baseline,text=True))
    assert current==old,'historical complete trajectories changed'
    dump(out/'historical_trajectories.json',dict(baseline_revision=revision,complete_episodes=len(old),
        observations_states_rewards_ammo_cooldown_rng_dwell_exact=True,results=current))
    print(f'{len(old)} historical complete episodes exactly match HEAD',flush=True)


def maac_update(out):
    ec,ac=configs('maac');out.mkdir()
    runner=MAACTrainingRunner(ec,ac,out,total_sampled_steps=12000,smoke=False)
    handles=[]
    try:
        t=runner.trainer
        assert runner.observations.shape==(12,4,45) and runner.batch_size==1024
        assert t.baseline_samples==16 and runner.replay.capacity==1000000
        before={name:[p.detach().clone() for p in getattr(t,name).parameters()]
                for name in ('actor','critic','target_actor','target_critic')}
        grads={k:dict(calls=0,max_abs=0.) for k in ('actor','critic','attention')}
        def hook(name):
            def receive(g):
                assert torch.isfinite(g).all()
                grads[name]['calls']+=1
                grads[name]['max_abs']=max(grads[name]['max_abs'],float(g.abs().max()))
            return receive
        for name in ('actor','critic'):
            for p in getattr(t,name).parameters(): handles.append(p.register_hook(hook(name)))
        for p in t.critic.shared_parameters(): handles.append(p.register_hook(hook('attention')))
        infos=[]
        for _ in range(1000):
            infos.extend(runner.collect_step())
            if t.sampled_steps>=1024 and len(runner.records)>=12: break
        assert len(runner.records)>=12 and t.sampled_steps<=12000
        d=t.batch_tensors(runner.replay.sample(1024))
        assert d['observations'].shape==(1024,4,45)
        y=t.td_targets(d['rewards'],d['next_observations'],d['dones'],d['next_alive_masks'])
        mask=d['dones'][:,None].bool()|~d['next_alive_masks'].bool()
        torch.testing.assert_close(y[mask],d['rewards'][mask],rtol=0,atol=0)
        runner.train_cycle()
        assert (t.critic_updates,t.actor_updates,t.target_updates)==(4,4,1)
        changes={}
        for name,initial in before.items():
            changes[name]=float(sum((p-q).square().sum() for p,q in zip(getattr(t,name).parameters(),initial)).sqrt().detach())
            assert changes[name]>0
        assert all(v['calls']>0 and v['max_abs']>0 for v in grads.values())
        for opt in (t.actor_optimizer,t.critic_optimizer):
            assert opt.state
            assert all(torch.isfinite(x).all() for s in opt.state.values() for x in s.values() if isinstance(x,torch.Tensor))
        runner.write_metrics(infos)
        path=out/'diagnostic_bounded.pt';t.save(path)
        reloaded,state,provenance=load_recording_policy(path,ec,ac,'cuda')
        validate_maac(state,ec,ac)
        np.testing.assert_array_equal(t.act(runner.observations,runner.vector.current_alive_masks),
            reloaded.act(runner.observations,runner.vector.current_alive_masks))
        for name in before:
            for p,q in zip(getattr(t,name).parameters(),getattr(reloaded,name).parameters()): assert torch.equal(p,q)
        runner.eval_seeds=[10000000,10000001];ev=runner.evaluate()
        m=record(reloaded,ec,20000348,out/'recording',provenance)
        export_analysis(out/'recording/episode_trace.npz',out/'recording/metadata.json',out/'recording',('csv','report'))
        result=dict(sampled_steps=t.sampled_steps,completed_episodes=len(runner.records),
            critic_updates=t.critic_updates,actor_updates=t.actor_updates,target_updates=t.target_updates,
            gradients=grads,parameter_change_l2=changes,evaluation=ev,metrics=runner.last_metrics,
            observation_dim=45,device='cuda',formal_hyperparameters_unchanged=True,
            formal_3m_training_started=False,checkpoint_reload_exact=True)
        dump(out/'summary.json',result)
        return result
    finally:
        for h in handles: h.remove()
        runner.vector.close()
        assert all(not p.is_alive() for p in runner.vector._processes)


def check_checkpoints_and_repeat(out):
    results=[]
    for algo,filename in (('mappo','diagnostic_4096.pt'),('maac','diagnostic_bounded.pt')):
        directory=out/algo;ec,ac=configs(algo)
        validator=validate_mappo if algo=='mappo' else validate_maac
        trainer,state,provenance=load_recording_policy(directory/filename,ec,ac,'cuda')
        for version in (('342','346','347') if algo=='mappo' else ('341','346','347')):
            try: validator(state,*configs(algo,version))
            except (RuntimeError,ValueError): pass
            else: raise AssertionError('cross-version checkpoint accepted')
        for key in ('environment_config_sha256','algorithm_config_sha256','observation_dim','action_dim','num_agents'):
            bad=deepcopy(state);bad['extra'][key]='invalid'
            try: validator(bad,ec,ac)
            except (RuntimeError,ValueError): pass
            else: raise AssertionError('corrupt checkpoint metadata accepted')
        ev=evaluate(trainer,ec,[10000002])
        assert ev==evaluate(trainer,ec,[10000002])
        metas=[record(trainer,ec,20000349,directory/f'repeat_{i}',provenance) for i in range(2)]
        traces=[read_trace(directory/f'repeat_{i}/episode_trace.npz',metas[i]) for i in range(2)]
        for key in traces[0]: np.testing.assert_array_equal(traces[0][key],traces[1][key])
        m=metas[0]
        assert m['events']==metas[1]['events']
        assert m['dwell_metrics_applicable'] is False and not any(e['type']=='dwell_complete' for e in m['events'])
        assert all(e['cone_pass'] for e in m['events'] if e['type']=='fire_attempt')
        for side in ('red','blue'):
            assert m[side+'_fire_attempts']==m[side+'_ammo_used']
        error=float(np.abs(traces[0]['local_rewards']-traces[0]['reward_components'].sum(axis=1)).max())
        assert error<1e-10
        results.append(dict(algorithm=algo,cross_version_rejected=True,corrupt_metadata_rejected=True,
            deterministic_evaluation_exact=True,repeat_trace_actions_events_exact=True,reward_error=error,
            attempts={s:m[s+'_fire_attempts'] for s in ('red','blue')},
            hits={s:m[s+'_weapon_hits'] for s in ('red','blue')}))
    # Read-only validation of actual historical training artifacts in both directions.
    history_results=[]
    for algo,version in (('mappo','347'),('mappo','346'),('maac','346')):
        path=ROOT/f'outputs/{algo}_v{version}_seed1_3m/best_eval.pt'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        trainer,state,_=load_recording_policy(path,*configs(algo,version),'cuda')
        ev=evaluate(trainer,configs(algo,version)[0],[10000000])
        try: (validate_mappo if algo=='mappo' else validate_maac)(state,*configs(algo))
        except (RuntimeError,ValueError): pass
        else: raise AssertionError('historical checkpoint accepted as v3.48')
        assert digest==hashlib.sha256(path.read_bytes()).hexdigest()
        history_results.append(dict(algorithm=algo,version=version,sha256=digest,read_only=True,evaluation=ev))
    dump(out/'historical_checkpoints.json',history_results)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('history','cuda'),required=True)
    args=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA mandatory; no CPU fallback')
    if args.output.exists(): raise FileExistsError('fresh output required')
    args.output.mkdir(parents=True);torch.set_num_threads(1)
    if args.mode=='history': history(args.output);return
    results=dict(mappo=ppo_run('mappo','348',args.output/'mappo'),maac=maac_update(args.output/'maac'))
    results['checks']=check_checkpoints_and_repeat(args.output)
    results['cuda_name']=torch.cuda.get_device_name()
    results['formal_3m_training_started']=False
    dump(args.output/'summary.json',results)
    print('Both bounded CUDA updates, repeat evaluations/recordings, checkpoint contracts PASS',flush=True)


if __name__=='__main__': main()
