"""CUDA checkpoint isolation and real multiprocessing reset acceptance."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import argparse
import json
import numpy as np
import torch
import yaml
from env.config import load_config
from env.combat_env import MultiUAVCombatEnv
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation, validate_checkpoint_for_resume
from algorithm.rmappo.protocol import validate_checkpoint as rm_check
from algorithm.ea_mappo.protocol import validate_checkpoint as ea_check
from algorithm.stea_mappo.protocol import validate_checkpoint as stea_check
from tools._capture_v33_reference import canonical
from tools.audit_combat_v39 import check_decomposition


def check_checkpoint_isolation():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for checkpoint audit')
    old=load_config(ROOT/'configs/combat_environment_v313.yaml')
    new=load_config(ROOT/'configs/combat_environment_v314.yaml')
    rows={}
    for name,stem,checker in [('mappo','mappo',validate_checkpoint_for_evaluation),
                             ('rmappo','rmappo',rm_check),('ea-mappo','ea_mappo',ea_check),
                             ('stea-mappo','stea_mappo',stea_check)]:
        checkpoint=ROOT/f'outputs/v313_validation/cuda_smoke/{name}/latest.pt'
        if not checkpoint.exists():
            raise RuntimeError(f'required existing v3.13 checkpoint missing: {checkpoint}')
        state=torch.load(checkpoint,map_location='cuda',weights_only=False)
        old_cfg=yaml.safe_load((ROOT/f'configs/{stem}_5v5_v313.yaml').read_bytes())
        new_cfg=yaml.safe_load((ROOT/f'configs/{stem}_5v5_v314.yaml').read_bytes())
        checker(state,old,old_cfg)
        try:
            checker(state,new,new_cfg)
        except RuntimeError as error:
            assert 'environment_version' in str(error)
            reason=str(error)
        else:
            raise AssertionError('v3.13 checkpoint accepted under v3.14')
        if name=='mappo':
            validate_checkpoint_for_resume(state,old,old_cfg)
            try:
                validate_checkpoint_for_resume(state,new,new_cfg)
            except RuntimeError as error:
                assert 'environment_version' in str(error)
            else:
                raise AssertionError('v3.13 MAPPO resume accepted under v3.14')
        rows[name]=dict(checkpoint=str(checkpoint),old_version='3.13',new_version='3.14',
            old_checkpoint_accepted=True,cross_version_rejected=True,
            reason=reason,observation_dim=state['extra']['observation_dim'])
    return rows


def check_vector_reset():
    vector=ParallelVectorEnv(1,ROOT/'configs/combat_environment_v314.yaml',base_seed=45000000)
    try:
        obs=vector.reset()
        assert obs.shape==(1,5,56) and not obs[:,:,4].any()
        assert vector.worker_environment_classes==['CombatEnvironmentV314']
        action=np.tile([.5,0,0],(1,5,1))
        seen_nonzero_phi=False
        accumulated=np.zeros(5)
        for step in range(1,1001):
            result=vector.step_batch(action)
            seen_nonzero_phi |= bool(result.transition_next_observations[:,:,4].any())
            assert np.isfinite(result.observations).all() and np.isfinite(result.rewards).all()
            accumulated+=result.rewards[0]
            check_decomposition(result.rewards[0],result.infos[0],accumulated)
            if result.terminated[0] or result.truncated[0]:
                break
        else:
            raise AssertionError('bounded vector episode failed to finish')
        assert seen_nonzero_phi
        assert not result.observations[:,:,4].any()
        assert np.array_equal(vector.episode_indices,[1])
        seed=int(vector.last_reset_seeds[0])
        fresh=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v314.yaml')
        reset_obs,_=fresh.reset(seed)
        np.testing.assert_array_equal(result.observations[0],reset_obs.astype(np.float32))
        obs,reward,term,trunc,info=fresh.step(action[0])
        nxt=vector.step_batch(action)
        np.testing.assert_array_equal(nxt.observations[0],obs.astype(np.float32))
        np.testing.assert_array_equal(nxt.rewards[0],reward)
        assert canonical(nxt.infos[0])==canonical(info)
        return dict(observation_shape=[1,5,56],worker_class=vector.worker_environment_classes[0],
            first_episode_steps=step,nonzero_phi_before_reset=seen_nonzero_phi,
            auto_reset_phi_zero=True,reset_seed=seed,reset_equal_fresh=True,
            first_step_equal_fresh=True,finite_rewards=True,reward_decomposition=True)
    finally:
        vector.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/v314_validation')
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    args.output.mkdir(parents=True,exist_ok=True)
    for name,result in [('checkpoint_isolation',check_checkpoint_isolation()),('vector_reset',check_vector_reset())]:
        (args.output/f'{name}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(name,json.dumps(result,indent=2))


if __name__=='__main__':
    main()
