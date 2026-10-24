"""CUDA-only bounded MAAC v3.42 integration, no formal training loop."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import subprocess
import sys

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def configs(version):
    return tuple(yaml.safe_load((ROOT/p).read_text(encoding='utf-8')) for p in (
        f'configs/combat_environment_v{version}.yaml',f'configs/maac_4v4_v{version}.yaml'))


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def smoke(out):
    # Spawn workers import this module. Keep parent-only CUDA modules out of
    # their imports; environment workers do not need separate PyTorch runtimes.
    import torch
    from algorithm.maac.runner import MAACTrainingRunner
    from algorithm.maac.protocol import validate_checkpoint
    from tools.record_combat_episode import load_recording_policy,record
    ec,ac=configs('342');directory=out/'cuda_smoke';directory.mkdir(exist_ok=False)
    for name,value in (('env_config.yaml',ec),('algorithm_config.yaml',ac)):
        (directory/name).write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
    runner=MAACTrainingRunner(ec,ac,directory,total_sampled_steps=1200,device='cuda',smoke=False)
    hooks=[]
    try:
        t=runner.trainer
        assert runner.num_envs==12 and runner.observations.shape==(12,4,45)
        assert set(runner.vector.worker_environment_classes)=={'CombatEnvironmentV342'}
        assert runner.batch_size==1024 and t.baseline_samples==16 and runner.replay.capacity==1000000
        assert t.actor.backbone[0].out_features==128 and t.critic.hidden_dim==128 and t.critic.heads==4
        names=('actor','critic','target_actor','target_critic')
        before={name:[p.detach().clone() for p in getattr(t,name).parameters()] for name in names}
        gradients={name:dict(calls=0,max_abs=0.) for name in ('actor','critic')}
        def hook(name):
            def receive(g):
                assert torch.isfinite(g).all()
                gradients[name]['calls']+=1
                gradients[name]['max_abs']=max(gradients[name]['max_abs'],float(g.abs().max()))
            return receive
        for name in gradients:
            for p in getattr(t,name).parameters():hooks.append(p.register_hook(hook(name)))
        infos=[]
        for _ in range(100):infos.extend(runner.collect_step())
        assert t.sampled_steps==len(runner.replay)==1200
        batch=runner.replay.sample(1024)
        assert batch.observations.shape==(1024,4,45) and batch.actions.shape==(1024,4,3)
        runner.train_cycle()
        assert (t.critic_updates,t.actor_updates,t.target_updates)==(4,4,1)
        assert all(np.isfinite(v) for v in runner.last_metrics.values())
        assert all(g['calls']>0 and g['max_abs']>0 for g in gradients.values())
        changes={}
        for name in names:
            params=list(getattr(t,name).parameters());assert all(torch.isfinite(p).all() for p in params)
            changes[name]=float(sum((p.detach()-q).square().sum() for p,q in zip(params,before[name])).sqrt())
            assert changes[name]>0
        for optimizer in (t.actor_optimizer,t.critic_optimizer):
            assert optimizer.state
            assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values() if isinstance(v,torch.Tensor))
        runner.write_metrics(infos)
        path=directory/'diagnostic_1200.pt';t.save(path)
        reloaded,state,provenance=load_recording_policy(path,ec,ac,'cuda')
        for name in names:
            for p,q in zip(getattr(t,name).parameters(),getattr(reloaded,name).parameters()):assert torch.equal(p,q)
        action=t.act(runner.observations,runner.vector.current_alive_masks,deterministic=True)
        np.testing.assert_array_equal(action,reloaded.act(runner.observations,runner.vector.current_alive_masks,deterministic=True))
        np.testing.assert_array_equal(action,t.act(runner.observations,runner.vector.current_alive_masks,deterministic=True))
        for version in ('341','348','350'):
            other_ec,other_ac=configs(version)
            try:validate_checkpoint(state,other_ec,other_ac)
            except RuntimeError:pass
            else:raise AssertionError('cross-version v342 checkpoint accepted')
            from algorithm.maac.protocol import identity
            old=deepcopy(state);old['extra']=identity(other_ec,other_ac)
            validate_checkpoint(old,other_ec,other_ac)
            try:validate_checkpoint(old,ec,ac)
            except RuntimeError:pass
            else:raise AssertionError('historical checkpoint identity accepted as v342')
        for key in ('environment_version','environment_config_sha256','algorithm_config_sha256','observation_dim','action_dim','num_agents'):
            bad=deepcopy(state);bad['extra'].pop(key)
            try:validate_checkpoint(bad,ec,ac)
            except RuntimeError:pass
            else:raise AssertionError('missing checkpoint identity accepted: '+key)
        runner.eval_seeds=[10000000,10000001]
        first=runner.evaluate();second=runner.evaluate()
        assert first==second and first['fire_rate']==first['red_attempt_episode_rate']
        meta=record(reloaded,ec,20000342,directory/'recording',dict(provenance,algorithm='maac'))
        assert meta['environment_version']=='3.42'
        result=dict(device='cuda',cuda_device=torch.cuda.get_device_name(),sampled_steps=1200,
            critic_updates=4,actor_updates=4,target_updates=1,gradients=gradients,
            parameter_change_l2=changes,optimization_metrics=runner.last_metrics,
            full_original_network_and_hyperparameters=True,reload_parameters_actions_exact=True,
            cross_version_rejected_both_directions=True,missing_checkpoint_identity_rejected=True,
            deterministic_evaluation_repeat_exact=True,evaluation=first,recording_steps=meta['episode_steps'],
            formal_training_started=False)
        dump(out/'cuda_summary.json',result)
        print('PASS CUDA: 1200 samples, 4 critic + 4 actor + 1 target; reload/evaluation/identity',flush=True)
    finally:
        for h in hooks:h.remove()
        runner.vector.close()


def history(out):
    from tools.verify_v350_readiness import HISTORY_PROBE
    probe=HISTORY_PROBE.replace("('342','346','347','348','349')","('341','342','348','350')").replace('(1,12)','(1,)')
    probe=probe.replace('print(json.dumps(result))', '''
from algorithm.maac.protocol import identity,validate_config_pair
import yaml
identities={}
for version in ('341','348','350'):
 ec=load_config(f'configs/combat_environment_v{version}.yaml')
 ac=yaml.safe_load(open(f'configs/maac_4v4_v{version}.yaml',encoding='utf-8'))
 validate_config_pair(ec,ac);identities[version]=identity(ec,ac)
print(json.dumps(dict(trajectories=result,identities=identities)))
''')
    a=json.loads(subprocess.check_output([sys.executable,'-c',probe],cwd=ROOT,text=True))
    b=json.loads(subprocess.check_output([sys.executable,'-c',probe],cwd=out/'baseline_source',text=True))
    assert a==b,'historical behavior or checkpoint identity changed'
    dump(out/'history.json',dict(complete_episodes=len(a['trajectories']),all_exact=True,**a))
    print('PASS history: complete states/rewards/observations/weapons/RNG and existing MAAC identities exact',flush=True)


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('smoke','history'),required=True);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    torch.set_num_threads(1)
    (smoke if args.stage=='smoke' else history)(args.output.resolve())


if __name__=='__main__':main()
