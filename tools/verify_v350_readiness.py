"""Bounded CUDA integration and read-only historical parity for reward v3.50.

No formal run loop: MAPPO 4096 samples/one PPO update; MAAC 1200 samples/one
existing 4-critic + 4-policy cycle. Configured networks/hyperparameters stay full.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


HISTORY_PROBE = r'''
import hashlib,json
import numpy as np
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config
def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value))
result={}
for version in ('342','346','347','348','349'):
 for seed in (1,12):
  for mode in ('zero','pursuit'):
   e=MultiUAVCombatEnv(load_config(f'configs/combat_environment_v{version}.yaml'))
   obs,info=e.reset(seed);h=hashlib.sha256(obs.tobytes());events=[]
   original=e._entry_attempts
   def entries(*args):
    attempts=original(*args);events.append((args[-1],attempts));return attempts
   e._entry_attempts=entries
   while True:
    events.clear()
    action=np.zeros((4,3),dtype=np.float32) if mode=='zero' else e.fixed_policy.team_actions(e.red,e.blue)
    obs,reward,term,trunc,info=e.step(action)
    for array in (obs,reward,np.array([s.as_array() for s in e.red+e.blue]),e.red_ammo,e.blue_ammo,
                  e.red_fire_cooldown_steps,e.blue_fire_cooldown_steps):h.update(array.tobytes())
    payload=dict(info=info,events=events,rng=e.rng.bit_generator.state,term=term,trunc=trunc)
    h.update(json.dumps(payload,default=serial,sort_keys=True).encode())
    if term or trunc:break
   result[f'{version}_{seed}_{mode}']=dict(sha256=h.hexdigest(),steps=e.steps,outcome=info['termination_reason'])
print(json.dumps(result))
'''


def history(out):
    baseline = out/'baseline_source'
    old = json.loads(subprocess.check_output([sys.executable, '-c', HISTORY_PROBE], cwd=baseline, text=True))
    new = json.loads(subprocess.check_output([sys.executable, '-c', HISTORY_PROBE], cwd=ROOT, text=True))
    assert old == new, 'historical full trajectories/info/events/rewards/RNG changed'
    dump(out/'historical_exact.json', dict(complete_episodes=len(new), all_exact=True, results=new))
    print(f'PASS history: {len(new)} complete episodes exact', flush=True)


def configs(algo, version='350'):
    from env.config import load_config
    return load_config(ROOT/f'configs/combat_environment_v{version}.yaml'), yaml.safe_load(
        (ROOT/f'configs/{algo}_4v4_v{version}.yaml').read_text(encoding='utf-8'))


def maac(out):
    from algorithm.maac.runner import MAACTrainingRunner
    from tools.record_combat_episode import load_recording_policy
    ec,ac=configs('maac');directory=out/'maac';directory.mkdir()
    runner=MAACTrainingRunner(ec,ac,directory,total_sampled_steps=1200,device='cuda',smoke=False)
    try:
        t=runner.trainer
        assert t.device.type=='cuda' and t.baseline_samples==16 and runner.batch_size==1024
        assert runner.replay.capacity==1000000
        assert t.actor.backbone[0].out_features==128
        initial={name:[p.detach().clone() for p in getattr(t,name).parameters()]
                 for name in ('actor','critic','target_actor','target_critic')}
        infos=[]
        for _ in range(100):infos.extend(runner.collect_step())
        assert t.sampled_steps==1200
        runner.train_cycle()
        assert (t.critic_updates,t.actor_updates,t.target_updates)==(4,4,1)
        changes={}
        for name,parameters in initial.items():
            current=list(getattr(t,name).parameters())
            assert all(torch.isfinite(p).all() for p in current)
            changes[name]=float(sum((p.detach()-q).square().sum() for p,q in zip(current,parameters)).sqrt())
            assert changes[name]>0
        assert all(np.isfinite(v) for v in runner.last_metrics.values())
        for optimizer in (t.actor_optimizer,t.critic_optimizer):
            assert optimizer.state
            assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values() if isinstance(v,torch.Tensor))
        runner.write_metrics(infos)
        path=directory/'diagnostic_1200.pt';t.save(path)
        reloaded,_,_=load_recording_policy(path,ec,ac,'cuda')
        for name in initial:
            for p,q in zip(getattr(t,name).parameters(),getattr(reloaded,name).parameters()):assert torch.equal(p,q)
        runner.eval_seeds=[10000000,10000001];evaluation=runner.evaluate()
        summary=dict(device='cuda',sampled_steps=1200,critic_updates=4,actor_updates=4,target_updates=1,
                     full_configured_networks_and_hyperparameters=True,formal_run_loop_entered=False,
                     parameter_change_l2=changes,optimization_metrics=runner.last_metrics,evaluation=evaluation,
                     checkpoint_reload_exact=True)
        dump(directory/'summary.json',summary)
        print('PASS MAAC: 1200 samples, one configured update cycle',flush=True)
    finally:runner.vector.close()


def checkpoint_audit(out):
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation as check_mappo
    from algorithm.maac.protocol import validate_checkpoint as check_maac
    from algorithm.common.evaluator import evaluate
    from tools.record_combat_episode import load_recording_policy, record
    from tools.combat_trace import read_trace
    from tools.combat_replay_report import export_analysis
    rows=[]
    for algo,filename,validator in (('mappo','diagnostic_4096.pt',check_mappo),('maac','diagnostic_1200.pt',check_maac)):
        ec,ac=configs(algo);directory=out/algo
        trainer,state,provenance=load_recording_policy(directory/filename,ec,ac,'cuda')
        for version in ('348','349'):
            try:validator(state,*configs(algo,version))
            except (ValueError,RuntimeError):pass
            else:raise AssertionError('cross-version checkpoint accepted')
        for key in ('environment_config_sha256','algorithm_config_sha256','reward_semantics','outcome_semantics'):
            bad=deepcopy(state);bad['extra'].pop(key)
            try:validator(bad,ec,ac)
            except (ValueError,RuntimeError):pass
            else:raise AssertionError('missing identity accepted: '+key)
        a=trainer.act(np.zeros((1,4,45),np.float32),np.ones((1,4),bool),deterministic=True)
        b=trainer.act(np.zeros((1,4,45),np.float32),np.ones((1,4),bool),deterministic=True)
        np.testing.assert_array_equal(a,b)
        ev=evaluate(trainer,ec,[10000002]);assert ev==evaluate(trainer,ec,[10000002])
        metas=[record(trainer,ec,20000350,directory/f'repeat_{i}',dict(provenance,algorithm=algo)) for i in range(2)]
        traces=[read_trace(directory/f'repeat_{i}/episode_trace.npz',metas[i]) for i in range(2)]
        for key in traces[0]:np.testing.assert_array_equal(traces[0][key],traces[1][key])
        assert metas[0]['events']==metas[1]['events']
        assert metas[0]['pbrs_enabled'] is False and metas[0]['dwell_metrics_applicable'] is False
        assert not metas[0]['draw']
        error=float(np.abs(traces[0]['local_rewards']-traces[0]['reward_components'].sum(axis=1)).max())
        assert error<1e-10
        export_analysis(directory/'repeat_0/episode_trace.npz',directory/'repeat_0/metadata.json',directory/'repeat_0',('csv','report'))
        rows.append(dict(algorithm=algo,cross_version_rejected=True,missing_identity_rejected=True,
                         deterministic_policy_actions_exact=True,seeded_environment_trace_exact=True,
                         reward_max_error=error,evaluation=ev))
    historical=[]
    for algo,validator in (('mappo',check_mappo),('maac',check_maac)):
        # Actual old trained checkpoints are loaded on CUDA, never rewritten.
        path=ROOT/f'outputs/{algo}_v348_seed1_3m/best_eval.pt'
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        ec=yaml.safe_load((path.parent/'env_config.yaml').read_text(encoding='utf-8'))
        ac=yaml.safe_load((path.parent/'algorithm_config.yaml').read_text(encoding='utf-8'))
        trainer,state,_=load_recording_policy(path,ec,ac,'cuda')
        assert trainer.device.type=='cuda'
        for version in ('349','350'):
            try:validator(state,*configs(algo,version))
            except (ValueError,RuntimeError):pass
            else:raise AssertionError('old checkpoint accepted as a different environment')
        assert before==hashlib.sha256(path.read_bytes()).hexdigest()
        historical.append(dict(algorithm=algo,path=str(path),sha256=before,original_load_pass=True,
                               cross_version_rejected=True,unchanged=True))
    dump(out/'checkpoint_replay_audit.json',dict(new_version=rows,historical=historical))
    print('PASS checkpoint identity, seeded evaluation/replay and historical model loads',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('history','mappo','maac','checkpoints'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    torch.set_num_threads(1)
    if a.stage=='mappo':
        from tools.verify_v342_v343_readiness import run
        run('mappo','350',out/'mappo')
    elif a.stage=='maac':maac(out)
    elif a.stage=='history':history(out)
    else:checkpoint_audit(out)


if __name__=='__main__':main()
