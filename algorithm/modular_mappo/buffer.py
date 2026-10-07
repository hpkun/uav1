"""Rollout schema retaining raw rewards, wave identity and recurrent state."""
from dataclasses import dataclass
import numpy as np

@dataclass
class ModularRolloutBatch:
    observations:np.ndarray; actions:np.ndarray; raw_actions:np.ndarray
    old_log_probs:np.ndarray; rewards:np.ndarray; raw_environment_rewards:np.ndarray
    dones:np.ndarray; alive_masks:np.ndarray; next_observations:np.ndarray
    next_alive_masks:np.ndarray; wave_indices:np.ndarray; total_waves:np.ndarray
    contexts:np.ndarray; next_contexts:np.ndarray
    actor_hidden_before_step:np.ndarray|None=None
    critic_hidden_before_step:np.ndarray|None=None
    episode_masks:np.ndarray|None=None
    remaining_horizons:np.ndarray|None=None
    next_remaining_horizons:np.ndarray|None=None
    wave_transition_flags:np.ndarray|None=None
    iw_supervision_observations:object|None=None
    iw_supervision_alive_masks:object|None=None
    iw_supervision_credit_waves:object|None=None
    iw_supervision_horizons:object|None=None
    iw_supervision_targets:object|None=None
    iw_supervision_segment_ids:object|None=None
    iw_supervision_boundary_flags:object|None=None
    iw_supervision_segments:object|None=None
    caiw_supervision_segments:object|None=None
    brsc_supervision_boundaries:object|None=None
    hta_options:np.ndarray|None=None
    hta_next_options:np.ndarray|None=None
    hta_manager_transitions:object|None=None
    marc_success_segments:object|None=None
    actor_recurrent_phase_reset_flags:np.ndarray|None=None

def contiguous_chunks(time_steps:int,num_envs:int,sequence_length:int):
    return [(env,start,min(start+sequence_length,time_steps)) for env in range(num_envs) for start in range(0,time_steps,sequence_length)]

def wave_segmented_chunks(wave_indices,phase_reset_flags,episode_masks,sequence_length):
    """Cover every step once, with explicit pre-action reset provenance."""
    waves=np.asarray(wave_indices);flags=np.asarray(phase_reset_flags)
    if int(sequence_length)<=0:raise ValueError("sequence_length must be positive")
    episodes=np.asarray(episode_masks)
    if waves.ndim!=2 or flags.shape!=waves.shape or episodes.shape!=waves.shape:
        raise RuntimeError("wave-segmented rollout requires matching [T,E] waves/reset flags/episode masks")
    if not np.isin(waves,[1,2,3]).all() or not np.isin(flags,[0,1]).all():
        raise RuntimeError("invalid wave/reset provenance")
    if np.any((episodes==0)&(flags!=1)):
        raise RuntimeError("episode start lacks explicit phase reset")
    T,E=waves.shape;chunks=[]
    for env in range(E):
        start=0
        for t in range(1,T):
            boundary=waves[t,env]!=waves[t-1,env] or episodes[t,env]==0
            if bool(flags[t,env])!=bool(boundary):
                raise RuntimeError("phase reset provenance inconsistent with episode/wave boundary")
            if boundary or t-start==sequence_length:
                chunks.append((env,start,t));start=t
        if start<T:chunks.append((env,start,T))
    return chunks

def recurrent_batch_plan(time_steps:int,num_envs:int,sequence_length:int,minibatch_size:int,ppo_epochs:int=1):
    chunks=len(contiguous_chunks(time_steps,num_envs,sequence_length))
    sequences_per_minibatch=max(1,minibatch_size//sequence_length)
    minibatches_per_epoch=int(np.ceil(chunks/sequences_per_minibatch))
    return {"sequence_chunks":chunks,"sequences_per_minibatch":sequences_per_minibatch,
            "recurrent_minibatches_per_epoch":minibatches_per_epoch,
            "optimizer_steps":minibatches_per_epoch*ppo_epochs}

def episode_contiguous_chunks(episode_masks,sequence_length):
    """Ordered joint recurrent chunks: split on real episode starts, not waves."""
    episodes=np.asarray(episode_masks)
    if episodes.ndim!=2 or not np.isin(episodes,[0,1]).all() or int(sequence_length)<=0:
        raise ValueError("recurrent chunks require binary [T,E] episode masks and positive length")
    T,E=episodes.shape;chunks=[]
    for env in range(E):
        start=0
        for t in range(1,T):
            if episodes[t,env]==0 or t-start==sequence_length:
                chunks.append((env,start,t));start=t
        if start<T:chunks.append((env,start,T))
    return chunks

def recurrent_alive_mean(values,alive_mask,valid_time_mask):
    mask=alive_mask*valid_time_mask[...,None]
    return (values*mask).sum()/mask.sum().clamp_min(1.0)

__all__=["ModularRolloutBatch","contiguous_chunks","recurrent_batch_plan","recurrent_alive_mean"]
