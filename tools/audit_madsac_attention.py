"""CUDA read-only attention audit on real environment inputs and existing weights."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import argparse
import numpy as np
import torch
import yaml
from algorithm.madsac.factory import build_madsac_trainer
from algorithm.madsac.protocol import validate_madsac_checkpoint
from env.factory import make_combat_environment
from tools.madsac_audit_reference import require_cuda,dump,sha256,attention_audit,gradient_audit


def collect(env_config, trainer, count, seed, random_policy=False, reset_each=False):
    env=make_combat_environment(env_config)
    obs,_=env.reset(seed)
    rng=np.random.default_rng(seed)
    observations=[]; actions=[]; masks=[]; episode=0
    for _ in range(count):
        alive=env.red_alive_mask.copy()
        action=(rng.uniform(-1,1,(env.team_size,3)).astype(np.float32)*alive[:,None]
                if random_policy else trainer.act(obs[None],alive[None])[0])
        observations.append(obs.copy()); actions.append(action.copy()); masks.append(alive)
        obs,_,terminated,truncated,_=env.step(action)
        if terminated or truncated or reset_each:
            episode+=1; obs,_=env.reset(seed+episode)
    return tuple(np.stack(x) for x in (observations,actions,masks))


def run(output, config, env_config, transitions=2400):
    from algorithm.madsac.trainer import MADSAC_IMPL_VERSION
    if MADSAC_IMPL_VERSION != 3:
        raise RuntimeError('Legacy implementation-v3 audit is frozen; use tools/validate_madsac_v4.py for v4')
    require_cuda(); torch.set_num_threads(1)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    c=yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    e=yaml.safe_load(Path(env_config).read_text(encoding='utf-8'))
    random=[]; gradients={}
    for seed in range(5):
        trainer=build_madsac_trainer(c,'cuda',seed=seed)
        reset=collect(e,trainer,128,1000+seed*1000,True,True)
        rollout=collect(e,trainer,384,20000+seed*1000,True)
        data=tuple(np.concatenate((x,y)) for x,y in zip(reset,rollout))
        for name in ('critic1','critic2'):
            label=f'random_seed{seed}_{name}'
            random.append(attention_audit(getattr(trainer,name),data,output,label))
            gradients[label]=gradient_audit(getattr(trainer,name),data)
        print(f'[RANDOM INIT] seed={seed} done',flush=True)
    dump(output/'attention_random_init.json',dict(category='RANDOM INIT',input_source='128 legal resets + 384 random-policy transitions per seed',results=random))
    trainer=build_madsac_trainer(c,'cuda',seed=0)
    data=collect(e,trainer,transitions,50000)
    np.savez_compressed(output/'real_env_inputs.npz',observations=data[0],actions=data[1],alive=data[2])
    real=[]
    for name in ('critic1','critic2'):
        label='real_env_'+name
        real.append(attention_audit(getattr(trainer,name),data,output,label))
        gradients[label]=gradient_audit(getattr(trainer,name),data)
    dump(output/'attention_real_env.json',dict(category='REAL ENV INPUT',transitions=transitions,trained=False,results=real))
    print('[REAL ENV INPUT] done',flush=True)
    checkpoint=[]; rejected=[]
    # Discover existing formal runs. Never treat a fresh audit checkpoint as pre-existing training evidence.
    for directory in sorted((ROOT/'outputs').glob('*')):
        if not directory.is_dir() or 'madsac' not in directory.name.lower() or 'audit' in directory.name.lower(): continue
        for basename in ('best_eval.pt','final.pt'):
            path=directory/basename
            if not path.exists(): continue
            try:
                state=torch.load(path,map_location='cuda',weights_only=False)
                extra=state.get('extra',{})
                if state.get('algorithm')!='madsac' or extra.get('training_smoke'): continue
                ec,cc=extra['environment_config'],extra['algorithm_config']
                validate_madsac_checkpoint(state,ec,cc,expected_training_seed=extra['training_seed'])
                tr=build_madsac_trainer(cc,'cuda',seed=extra['training_seed'])
                # load_for_evaluation intentionally loads actors only. Explicitly load each critic state here.
                for name in ('actor','target_actor','critic1','critic2','target_critic1','target_critic2'):
                    getattr(tr,name).load_state_dict(state[name],strict=True)
                inputs=collect(ec,tr,1024,60000)
                results=[]
                for name in ('critic1','critic2'):
                    label=f'checkpoint_{directory.name}_{path.stem}_{name}'
                    results.append(attention_audit(getattr(tr,name),inputs,output,label))
                    gradients[label]=gradient_audit(getattr(tr,name),inputs)
                checkpoint.append(dict(path=str(path),sha256=sha256(path),sampled_steps=state['sampled_steps'],
                    environment_version=ec['environment_version'],agents=cc['network']['num_agents'],results=results))
                print(f'[TRAINED CHECKPOINT] {path.name} steps={state["sampled_steps"]} done',flush=True)
            except (RuntimeError,KeyError,ValueError) as exc:
                rejected.append(dict(path=str(path),reason=str(exc)))
    dump(output/'attention_checkpoint.json',dict(category='TRAINED CHECKPOINT',results=checkpoint,rejected=rejected,
        note='Historical checkpoints use their own validated environment/config; not v3.37-trained evidence.'))
    dump(output/'gradient_comparison.json',gradients)
    return dict(random_init=len(random),real_env=len(real),checkpoints=len(checkpoint),rejected=rejected)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='outputs/madsac_algorithm_audit')
    p.add_argument('--config',default='configs/madsac_4v4_v337.yaml')
    p.add_argument('--env-config',default='configs/combat_environment_v337.yaml')
    p.add_argument('--transitions',type=int,default=2400)
    a=p.parse_args(); print(run(a.output_dir,a.config,a.env_config,a.transitions))
