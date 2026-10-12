"""CUDA checkpoint version isolation and real v3.16 vector auto-reset."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
from algorithm.rmappo.protocol import validate_checkpoint as rm_check
from algorithm.ea_mappo.protocol import validate_checkpoint as ea_check
from algorithm.stea_mappo.protocol import validate_checkpoint as stea_check
from tools.audit_combat_v39 import check_decomposition


def check_checkpoints(smoke_output):
    old=load_config(ROOT/'configs/combat_environment_v315.yaml')
    new=load_config(ROOT/'configs/combat_environment_v316.yaml')
    records={}
    for name,stem,check in [('mappo','mappo',validate_checkpoint_for_evaluation),('rmappo','rmappo',rm_check),('ea-mappo','ea_mappo',ea_check),('stea-mappo','stea_mappo',stea_check)]:
        cfg=yaml.safe_load((ROOT/f'configs/{stem}_5v5_v316.yaml').read_bytes())
        prior=torch.load(ROOT/f'outputs/v315_validation/cuda_smoke/{name}/latest.pt',map_location='cuda',weights_only=False)
        current=torch.load(smoke_output/name/'latest.pt',map_location='cuda',weights_only=False)
        check(prior,old,cfg);check(current,new,cfg)
        for state,env in [(prior,new),(current,old)]:
            try:check(state,env,cfg)
            except RuntimeError as error:assert 'environment_version' in str(error)
            else:raise AssertionError('cross-version checkpoint accepted')
        if name=='mappo':
            validate_checkpoint_for_resume(current,new,cfg)
            try:validate_checkpoint_for_resume(prior,new,cfg)
            except RuntimeError as error:assert 'environment_version' in str(error)
            else:raise AssertionError('v3.15 MAPPO resume accepted in v3.16')
        records[name]=dict(old_accepted=True,new_accepted=True,cross_version_rejected_both_directions=True)
    cfg=yaml.safe_load((ROOT/'configs/mappo_5v5_v316.yaml').read_bytes())
    prior=torch.load(ROOT/'outputs/mappo_v315_seed1_3m/best_eval.pt',map_location='cuda',weights_only=False)
    validate_checkpoint_for_evaluation(prior,old,cfg)
    try:validate_checkpoint_for_evaluation(prior,new,cfg)
    except RuntimeError as error:assert 'environment_version' in str(error)
    else:raise AssertionError('v3.15 3m best checkpoint accepted in v3.16')
    records['mappo_v315_3m_best']=dict(old_accepted=True,cross_version_rejected=True,new_episode_generated=False)
    return records


def check_vector():
    vector=ParallelVectorEnv(1,ROOT/'configs/combat_environment_v316.yaml',base_seed=45000000)
    try:
        obs=vector.reset()
        assert obs.shape==(1,5,56) and not obs[:,:,4].any()
        assert vector.worker_environment_classes==['CombatEnvironmentV316']
        action=np.tile([.5,0,0],(1,5,1));total=np.zeros(5)
        for step in range(1,1001):
            result=vector.step_batch(action)
            assert np.isfinite(result.observations).all() and np.isfinite(result.rewards).all()
            assert not result.observations[:,:,4].any() and not result.transition_next_observations[:,:,4].any()
            total+=result.rewards[0];check_decomposition(result.rewards[0],result.infos[0],total)
            if result.terminated[0] or result.truncated[0]:break
        else:raise AssertionError('episode did not end')
        assert np.array_equal(vector.episode_indices,[1])
        seed=int(vector.last_reset_seeds[0]);fresh=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v316.yaml')
        reset,_=fresh.reset(seed);np.testing.assert_array_equal(result.observations[0],reset.astype(np.float32))
        expected,reward,_,_,_=fresh.step(action[0]);nxt=vector.step_batch(action)
        np.testing.assert_array_equal(nxt.observations[0],expected.astype(np.float32));np.testing.assert_array_equal(nxt.rewards[0],reward)
        return dict(worker_class=vector.worker_environment_classes[0],first_episode_steps=step,reset_seed=seed,
                    phi_slot_zero_every_step_and_auto_reset=True,reset_and_first_step_match_fresh=True,reward_decomposition=True)
    finally:vector.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/v316_validation')
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    args.output.mkdir(parents=True,exist_ok=True)
    for name,check in [('checkpoint_isolation',lambda:check_checkpoints(args.output/'cuda_smoke')),('vector_reset',check_vector)]:
        result=check();(args.output/f'{name}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(name,json.dumps(result),flush=True)


if __name__=='__main__':main()
