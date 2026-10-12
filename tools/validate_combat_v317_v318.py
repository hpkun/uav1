"""Zero-training CUDA runner/checkpoint interfaces and real vector auto-reset."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.config import load_config
from env.combat_env import MultiUAVCombatEnv
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.rmappo.runner import RMAPPOTrainingRunner
from algorithm.ea_mappo.runner import EAMAPPOTrainingRunner
from algorithm.stea_mappo.runner import STEAMAPPOTrainingRunner
from algorithm.rmappo.protocol import validate_checkpoint as rm_check
from algorithm.ea_mappo.protocol import validate_checkpoint as ea_check
from algorithm.stea_mappo.protocol import validate_checkpoint as stea_check
from tools.record_combat_episode import load_recording_policy
from tools.audit_combat_v39 import check_decomposition

ALGORITHMS=[('mappo','mappo',MAPPOTrainingRunner,validate_checkpoint_for_evaluation),
            ('rmappo','rmappo',RMAPPOTrainingRunner,rm_check),
            ('ea-mappo','ea_mappo',EAMAPPOTrainingRunner,ea_check),
            ('stea-mappo','stea_mappo',STEAMAPPOTrainingRunner,stea_check)]


def checkpoint_interfaces(output):
    envs={v:load_config(ROOT/f'configs/combat_environment_v{v}.yaml') for v in ('316','317','318')}
    rows=[]
    for name,stem,runner_class,check in ALGORITHMS:
        cfg=yaml.safe_load((ROOT/f'configs/{stem}_5v5_v316.yaml').read_bytes())
        prior=torch.load(ROOT/f'outputs/v316_validation/cuda_smoke/{name}/latest.pt',map_location='cuda',weights_only=False)
        check(prior,envs['316'],cfg)
        for version in ('317','318'):
            try:check(prior,envs[version],cfg)
            except RuntimeError as error:assert 'environment_version' in str(error)
            else:raise AssertionError('historical checkpoint accepted in new version')
            run=output/'interface_checkpoints'/f'v{version}'/name
            # Construct official formal-width runner, but never call train/collect/update.
            runner=runner_class(envs[version],cfg,num_envs=1,total_sampled_steps=1,
                                seed=31,device='cuda',output_dir=run,smoke=False)
            try:
                assert runner.trainer.sampled_steps==0
                assert runner.vector.worker_environment_classes==[f'CombatEnvironmentV{version}']
                checkpoint=run/'untrained.pt';runner.save_checkpoint(checkpoint)
                state=torch.load(checkpoint,map_location='cuda',weights_only=False)
                assert state['sampled_steps']==0
                check(state,envs[version],cfg)
                if name=='mappo':validate_checkpoint_for_resume(state,envs[version],cfg)
                for other in ('316','317','318'):
                    if other==version:continue
                    try:check(state,envs[other],cfg)
                    except RuntimeError as error:assert 'environment_version' in str(error)
                    else:raise AssertionError('cross-version untrained checkpoint accepted')
                trainer,_,_=load_recording_policy(checkpoint,envs[version],cfg,'cuda')
                assert trainer.sampled_steps==0
                rows.append(dict(version=f'3.{version[-2:]}',algorithm=name,official_runner_constructed=True,
                    worker_class=runner.vector.worker_environment_classes[0],sampled_steps=0,train_called=False,
                    matching_checkpoint_accepted=True,all_other_versions_rejected=True,recording_loader_accepted=True))
            finally:runner.vector.close()
    return rows


def vector_reset(version):
    vector=ParallelVectorEnv(1,ROOT/f'configs/combat_environment_v{version}.yaml',base_seed=45000000)
    try:
        obs=vector.reset();assert obs.shape==(1,5,56) and not obs[:,:,4].any()
        assert vector.worker_environment_classes==[f'CombatEnvironmentV{version}']
        action=np.tile([.5,0,0],(1,5,1));total=np.zeros(5)
        for step in range(1,1001):
            result=vector.step_batch(action)
            assert np.isfinite(result.observations).all() and np.isfinite(result.rewards).all()
            assert not result.observations[:,:,4].any() and not result.transition_next_observations[:,:,4].any()
            total+=result.rewards[0];check_decomposition(result.rewards[0],result.infos[0],total)
            if result.terminated[0] or result.truncated[0]:break
        else:raise AssertionError('vector episode did not end')
        assert np.array_equal(vector.episode_indices,[1])
        seed=int(vector.last_reset_seeds[0]);fresh=MultiUAVCombatEnv(ROOT/f'configs/combat_environment_v{version}.yaml')
        reset,_=fresh.reset(seed);np.testing.assert_array_equal(result.observations[0],reset.astype(np.float32))
        if version=='318':assert all(s.armed for s in fresh.red_fire_states+fresh.blue_fire_states)
        expected,reward,_,_,_=fresh.step(action[0]);nxt=vector.step_batch(action)
        np.testing.assert_array_equal(nxt.observations[0],expected.astype(np.float32));np.testing.assert_array_equal(nxt.rewards[0],reward)
        return dict(worker_class=vector.worker_environment_classes[0],first_episode_steps=step,reset_seed=seed,
            finite=True,zero_phi_observation=True,reset_and_first_step_match_fresh=True,reward_decomposition=True)
    finally:vector.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/v317_v318_validation');args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    args.output.mkdir(parents=True,exist_ok=True)
    records=checkpoint_interfaces(args.output)
    (args.output/'checkpoint_interfaces.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('checkpoint_interfaces',json.dumps(records),flush=True)
    records={v:vector_reset(v) for v in ('317','318')}
    (args.output/'vector_reset.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('vector_reset',json.dumps(records),flush=True)


if __name__=='__main__':main()
