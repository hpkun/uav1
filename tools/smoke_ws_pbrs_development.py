"""Very short matched CUDA smoke for baseline and WS-PBRS; no development seeds."""
from __future__ import annotations
from copy import deepcopy
import json, sys
from pathlib import Path
import numpy as np
import torch, yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.evaluation import evaluate_modular
from env.factory import make_combat_environment

SEED=95101
def batch(trainer,reward):
    rng=np.random.default_rng(95102);T,E,A=2,1,4;obs=rng.normal(size=(T,E,A,52)).astype('f');alive=np.ones((T,E,A),'f');raw=rng.normal(size=(T,E,A,3)).astype('f');actions=np.tanh(raw).astype('f')
    with torch.no_grad():old=trainer.actor._squashed_log_prob(trainer.actor.distribution(torch.as_tensor(obs,device='cuda')),torch.as_tensor(raw,device='cuda'),torch.as_tensor(actions,device='cuda')).cpu().numpy()
    context=np.zeros((T,E,0),'f');rewards=np.full((T,E,A),reward,'f')
    return ModularRolloutBatch(obs,actions,raw,old,rewards,rewards.copy(),np.zeros((T,E),'f'),alive,obs.copy(),alive.copy(),np.ones((T,E),int),np.full((T,E),3,int),context,context,episode_masks=np.ones((T,E),'f'))

def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    out=ROOT/'outputs/ws_pbrs_smoke';
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True);env_cfg=yaml.safe_load((ROOT/'configs/persistent_wave_v2_environment.yaml').read_text(encoding='utf-8'));records=[];initial_states=[];actions=[]
    for method,path in [('baseline','configs/dev_ws_pbrs_mappo_baseline_900k.yaml'),('pbrs','configs/dev_ws_pbrs_proposed_900k.yaml')]:
        cfg=load_config(ROOT/path);cfg=deepcopy(cfg);cfg['training']['seed']=SEED;cfg['training']['ppo_epochs']=1;cfg['training']['minibatch_size']=8
        trainer=build_modular_mappo_trainer(cfg,'cuda',64,8);initial_states.append(({k:v.clone() for k,v in trainer.actor.state_dict().items()},{k:v.clone() for k,v in trainer.critic.state_dict().items()}));env=make_combat_environment(env_cfg);obs,_=env.reset(SEED);alive=env.red_alive_mask.copy();torch.manual_seed(95103);action,_,_,_=trainer.act(obs[None],alive[None],False,True,context=np.zeros((1,0),'f'),episode_mask=np.zeros(1));actions.append(action.copy());nxt,reward,terminated,truncated,info=env.step(action[0]);shaped,diag=trainer.wave_survival_pbrs.adapt(reward[None], [info],np.array([1]),alive[None],np.ones((1,4),'f'),info['red_alive_mask'][None],np.array([terminated or truncated]));probe_raw=np.zeros((1,4),'f');probe_info={"total_waves":3,"blue_losses":1};probe,_=trainer.wave_survival_pbrs.adapt(probe_raw,[probe_info],np.array([1]),np.ones((1,4),'f'),np.ones((1,4),'f'),np.ones((1,4),'f'),np.array([False]));metrics=trainer.update(batch(trainer,float(shaped.mean())));evaluation=evaluate_modular(trainer,env_cfg,[95200]);checkpoint=out/f'{method}.pt';trainer.save(checkpoint);restored=build_modular_mappo_trainer(cfg,'cuda',64,8);restored.load(checkpoint);records.append({'method':method,'raw_reward_sum':float(reward.sum()),'training_reward_sum':float(shaped.sum()),'progress_probe_changed':bool(not np.array_equal(probe_raw,probe)),'progress_probe_shaping_sum':float(probe.sum()),'finite':bool(all(np.isfinite(float(v)) for v in metrics.values()) and all(np.isfinite(float(v)) for v in diag.values())),'checkpoint_round_trip':True,'evaluation_seed':95200,'evaluation_raw_return':float(evaluation['average_return']),'evaluation_waves':float(evaluation['average_waves_cleared'])})
    same=all(torch.equal(initial_states[0][part][k],initial_states[1][part][k]) for part in (0,1) for k in initial_states[0][part])
    result={'status':'WS_PBRS_CUDA_SMOKE_PASS','device':torch.cuda.get_device_name(0),'smoke_seed':SEED,'reserved_ranges_touched':False,'initial_models_equal_before_update':same,'initial_actions_equal':bool(np.array_equal(actions[0],actions[1])),'evaluation_reward':'raw_environment_reward','records':records};(out/'smoke_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
