from __future__ import annotations
import numpy as np,pytest,torch,yaml
from pathlib import Path

from algorithm.mappo.networks import SharedMAPPOActor
from algorithm.mappo.trainer import compute_gae
from env.persistent_env import PersistentWaveCombatEnv

ROOT=Path(__file__).resolve().parents[1]

def gae(reward,value,next_value,done,next_alive,alive=1.0):
 return compute_gae(torch.tensor([[[reward]]]),torch.tensor([[[value]]]),torch.tensor([[[next_value]]]),torch.tensor([[done]]),torch.tensor([[[alive]]]),torch.tensor([[[next_alive]]]),.9,.95)

def test_live_transition_bootstraps():
 advantage,returns=gae(1.,2.,3.,0.,1.);assert advantage.item()==pytest.approx(1.7);assert returns.item()==pytest.approx(3.7)

def test_episode_terminal_stops_bootstrap_but_retains_reward():
 advantage,returns=gae(1.,2.,999.,1.,1.);assert advantage.item()==-1.;assert returns.item()==1.

def test_individual_death_stops_bootstrap_but_retains_death_step_reward():
 advantage,returns=gae(-10.,2.,999.,0.,0.);assert advantage.item()==-12.;assert returns.item()==-10.

def test_already_dead_slot_is_fully_masked():
 advantage,returns=gae(7.,2.,3.,0.,0.,alive=0.);assert advantage.item()==0. and returns.item()==0.

def test_real_intermediate_wave_clear_is_nonterminal_and_returns_fresh_wave_observation():
 config=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"));env=PersistentWaveCombatEnv(config);env.reset(88_910_001)
 for state in env.blue:state.alive=False
 observation,reward,terminated,truncated,info=env.step(np.zeros((4,3),np.float32))
 assert info["wave_cleared_this_step"] and info["spawned_next_wave"] and info["wave_index"]==2
 assert not terminated and not truncated and observation.shape==(4,52) and info["blue_survivors"]==4

def test_real_final_wave_clear_is_terminal():
 config=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"));env=PersistentWaveCombatEnv(config);env.reset(88_910_002);env.wave_index=env.total_waves;env.waves_cleared=env.total_waves-1
 for state in env.blue:state.alive=False
 _,_,terminated,truncated,info=env.step(np.zeros((4,3),np.float32))
 assert terminated and not truncated and not info["spawned_next_wave"] and info["waves_cleared"]==3

def test_squashed_entropy_sample_consumes_rng_and_negative_differential_entropy_is_legal():
 actor=SharedMAPPOActor(hidden_dim=16);observations=torch.zeros(8,52);actions,raw,_,_=actor.sample(observations);before=torch.get_rng_state().clone();_,entropy=actor.evaluate_actions(observations,actions,raw);after=torch.get_rng_state()
 assert not torch.equal(before,after) and torch.isfinite(entropy).all()
