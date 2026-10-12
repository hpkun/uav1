"""CUDA checkpoint isolation, old-actor offline ZERO regression and vector reset."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from env.config import load_config
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.v315_observation import build_team_observations_v315
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
from algorithm.rmappo.protocol import validate_checkpoint as rm_check
from algorithm.ea_mappo.protocol import validate_checkpoint as ea_check
from algorithm.stea_mappo.protocol import validate_checkpoint as stea_check
from tools.record_combat_episode import load_recording_policy
from tools.audit_combat_v39 import check_decomposition


def require_cuda():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')


def check_checkpoint_isolation():
    require_cuda();old=load_config(ROOT/'configs/combat_environment_v314.yaml');new=load_config(ROOT/'configs/combat_environment_v315.yaml');rows={}
    for name,stem,check in [('mappo','mappo',validate_checkpoint_for_evaluation),('rmappo','rmappo',rm_check),('ea-mappo','ea_mappo',ea_check),('stea-mappo','stea_mappo',stea_check)]:
        path=ROOT/f'outputs/v314_validation/cuda_smoke/{name}/latest.pt'
        if not path.exists():raise RuntimeError(f'required existing v3.14 checkpoint missing: {path}')
        state=torch.load(path,map_location='cuda',weights_only=False)
        oldcfg=yaml.safe_load((ROOT/f'configs/{stem}_5v5_v314.yaml').read_bytes());newcfg=yaml.safe_load((ROOT/f'configs/{stem}_5v5_v315.yaml').read_bytes())
        check(state,old,oldcfg)
        try:check(state,new,newcfg)
        except RuntimeError as error:
            assert 'environment_version' in str(error);reason=str(error)
        else:raise AssertionError('v3.14 checkpoint accepted under v3.15')
        if name=='mappo':
            validate_checkpoint_for_resume(state,old,oldcfg)
            try:validate_checkpoint_for_resume(state,new,newcfg)
            except RuntimeError as error:assert 'environment_version' in str(error)
            else:raise AssertionError('cross-version MAPPO resume accepted')
        rows[name]=dict(checkpoint=str(path),old_accepted=True,cross_version_rejected=True,reason=reason)
    pilot=torch.load(ROOT/'outputs/mappo_v314_seed1_500k/checkpoint_500000.pt',map_location='cuda',weights_only=False)
    cfg=yaml.safe_load((ROOT/'configs/mappo_5v5_v314.yaml').read_bytes());validate_checkpoint_for_evaluation(pilot,old,cfg)
    try:validate_checkpoint_for_evaluation(pilot,new,cfg)
    except RuntimeError as error:assert 'environment_version' in str(error)
    else:raise AssertionError('old 500k pilot accepted under v3.15')
    rows['mappo_500k']=dict(old_accepted=True,cross_version_rejected=True)
    return rows


def check_old_zero_counterfactual():
    require_cuda();source=ROOT/'outputs/v314_closed_loop_diagnosis/reconstructed_actor_inputs.npz'
    with np.load(source) as data:keys=data['record_keys'];oldobs=data['observations'];prior=data['zero_actions']
    # Load only under its correct v3.14 checkpoint contract; never evaluate a
    # complete v3.15 episode with this old policy.
    cfg=load_config(ROOT/'configs/combat_environment_v314.yaml');alg=yaml.safe_load((ROOT/'configs/mappo_5v5_v314.yaml').read_bytes())
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):trainer,_,_=load_recording_policy(ROOT/'outputs/mappo_v314_seed1_500k/checkpoint_500000.pt',cfg,alg,'cuda')
    cache={};newobs=[];expected=[];indices=[]
    segments=[(10000000,4,29,40),(10000001,3,34,151),(10000002,4,85,141)]
    for i,(seed,agent,step) in enumerate(keys):
        if not any(seed==s and agent==j and lo<=step<=hi for s,j,lo,hi in segments):continue
        if int(seed) not in cache:
            with np.load(ROOT/f'outputs/mappo_v314_seed1_500k_replay_analysis/seed_{seed}/episode_trace.npz') as trace:cache[int(seed)]={key:trace[key] for key in trace.files}
        t=cache[int(seed)];frame=int(step)-1
        teams=[[AircraftState(*v,alive=bool(al)) for v,al in zip(t[f'{side}_kinematics'][frame],t[f'{side}_alive'][frame])] for side in ['red','blue']]
        obs=build_team_observations_v315(*teams,cfg['observation'])[int(agent)-1].astype(np.float32)
        want=oldobs[i].copy();want[4]=0.;np.testing.assert_array_equal(obs,want)
        newobs.append(obs);expected.append(prior[i]);indices.append(i)
    with torch.no_grad():actual=trainer.actor.deterministic(torch.as_tensor(np.asarray(newobs),device='cuda')).cpu().numpy()
    residual=float(abs(actual-np.asarray(expected)).max());assert residual<1e-5
    return dict(segments=segments,observation_rows=len(indices),observation_exactly_equals_zero_case=True,actor_max_abs_error=residual,matches_prior_zero_counterfactual=True,old_model_used_only_for_offline_forward=True,new_episode_generated=False)


def check_new_checkpoints():
    require_cuda()
    new=load_config(ROOT/'configs/combat_environment_v315.yaml')
    old=load_config(ROOT/'configs/combat_environment_v314.yaml')
    rows={}
    for name,stem,check in [('mappo','mappo',validate_checkpoint_for_evaluation),('rmappo','rmappo',rm_check),('ea-mappo','ea_mappo',ea_check),('stea-mappo','stea_mappo',stea_check)]:
        path=ROOT/f'outputs/v315_validation/cuda_smoke/{name}/latest.pt'
        state=torch.load(path,map_location='cuda',weights_only=False)
        cfg=yaml.safe_load((ROOT/f'configs/{stem}_5v5_v315.yaml').read_bytes())
        check(state,new,cfg)
        if name=='mappo':validate_checkpoint_for_resume(state,new,cfg)
        try:check(state,old,cfg)
        except RuntimeError as error:assert 'environment_version' in str(error)
        else:raise AssertionError('v3.15 checkpoint accepted under v3.14')
        rows[name]=dict(checkpoint=str(path),new_accepted=True,reverse_cross_version_rejected=True)
    return rows


def check_vector_reset():
    vector=ParallelVectorEnv(1,ROOT/'configs/combat_environment_v315.yaml',base_seed=45000000)
    try:
        obs=vector.reset();assert obs.shape==(1,5,56) and not obs[:,:,4].any()
        assert vector.worker_environment_classes==['CombatEnvironmentV315']
        action=np.tile([.5,0,0],(1,5,1));accumulated=np.zeros(5);phi_seen=False
        for step in range(1,1001):
            result=vector.step_batch(action)
            assert not result.observations[:,:,4].any() and not result.transition_next_observations[:,:,4].any()
            assert np.isfinite(result.observations).all() and np.isfinite(result.rewards).all()
            phi_seen|=bool(np.asarray(result.infos[0]['executed_red_phi']).any())
            accumulated+=result.rewards[0];check_decomposition(result.rewards[0],result.infos[0],accumulated)
            if result.terminated[0] or result.truncated[0]:break
        else:raise AssertionError('vector episode did not terminate')
        assert phi_seen and np.array_equal(vector.episode_indices,[1])
        seed=int(vector.last_reset_seeds[0]);fresh=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v315.yaml');reset,_=fresh.reset(seed)
        np.testing.assert_array_equal(result.observations[0],reset.astype(np.float32))
        expected,reward,term,trunc,info=fresh.step(action[0]);nxt=vector.step_batch(action)
        np.testing.assert_array_equal(nxt.observations[0],expected.astype(np.float32));np.testing.assert_array_equal(nxt.rewards[0],reward)
        assert not nxt.observations[:,:,4].any()
        return dict(worker_class=vector.worker_environment_classes[0],observation_shape=[1,5,56],first_episode_steps=step,executed_phi_nonzero=True,every_transition_phi_slot_zero=True,auto_reset_phi_slot_zero=True,reset_seed=seed,reset_equal_fresh=True,first_step_equal_fresh=True,reward_decomposition=True)
    finally:vector.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'outputs/v315_validation');args=parser.parse_args();require_cuda();args.output.mkdir(parents=True,exist_ok=True)
    for name,check in [('checkpoint_isolation',check_checkpoint_isolation),('new_checkpoint_acceptance',check_new_checkpoints),('old_zero_counterfactual',check_old_zero_counterfactual),('vector_reset',check_vector_reset)]:
        result=check();(args.output/f'{name}.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(name,json.dumps(result),flush=True)


if __name__=='__main__':main()
