from copy import deepcopy
import numpy as np
import torch
from pathlib import Path

from algorithm.modules.wave_survival_pbrs import WaveSurvivalPotentialShapingModule
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.train_modular_mappo import load_config
from tools.preflight_ws_pbrs_development import validate_ws_pbrs_only_reward_shaping_diff


def transition(module, *, wave=1, blue=(1,1,1,1), red=(1,1,1,1), next_red=(1,1,1,1), losses=0, done=False):
    raw=np.arange(4,dtype=np.float32)[None]
    info={"total_waves":3,"blue_losses":losses}
    shaped,_=module.adapt(raw,[info],np.array([wave]),np.array([red],np.float32),np.array([blue],np.float32),np.array([next_red],np.float32),np.array([done]))
    return raw,shaped


def test_disabled_is_exact_identity_and_has_no_parameters():
    m=WaveSurvivalPotentialShapingModule({"enabled":False},.999);raw,out=transition(m)
    assert np.array_equal(raw,out) and not hasattr(m,"parameters")


def test_progress_survival_and_wave_refresh_are_continuous():
    m=WaveSurvivalPotentialShapingModule({"enabled":True,"coefficient":1.,"potential":"progress_times_survival","terminal_zero":True},.999)
    transition(m,wave=1,blue=(1,1,0,0),losses=2);assert np.allclose(m.last_transition["phi_pre"],1/6)
    transition(m,wave=1,blue=(1,0,0,0),red=(1,1,1,0),next_red=(1,1,1,0),losses=4)
    assert np.allclose(m.last_transition["phi_pre"][0,:3],3/16)
    assert np.allclose(m.last_transition["phi_next"][0,:3],1/4)


def test_agent_death_and_episode_terminal_use_zero_next_potential():
    m=WaveSurvivalPotentialShapingModule({"enabled":True,"coefficient":1.,"potential":"progress_times_survival","terminal_zero":True},.999)
    transition(m,wave=2,losses=5,next_red=(1,1,0,1));assert m.last_transition["phi_next"][0,2]==0 and m.last_transition["phi_next"][0,0]>0
    transition(m,wave=3,losses=12,next_red=(1,1,0,1),done=True);assert np.array_equal(m.last_transition["phi_next"],np.zeros((1,4)))


def test_timeout_and_telescoping_and_bounds():
    gamma=.999;phi=np.array([0.,.1,.3,.2,0.]);f=gamma*phi[1:]-phi[:-1]
    assert np.isclose(sum(gamma**t*f[t] for t in range(len(f))),-phi[0]+gamma**(len(phi)-1)*phi[-1])
    m=WaveSurvivalPotentialShapingModule({"enabled":True,"coefficient":1.,"potential":"progress_times_survival","terminal_zero":True},gamma)
    transition(m,wave=3,blue=(0,0,0,1),red=(1,1,0,0),next_red=(1,0,0,0),losses=11,done=True)
    assert np.all(np.isfinite(m.last_transition["shaping"])) and np.abs(m.last_transition["shaping"]).max()<=1+1e-6
    assert m.last_transition["phi_next"].max()==0


def test_configs_only_differ_by_shaping_and_initial_models_match():
    base=load_config('configs/dev_ws_pbrs_mappo_baseline_900k.yaml');proposed=load_config('configs/dev_ws_pbrs_proposed_900k.yaml')
    validate_ws_pbrs_only_reward_shaping_diff(base,proposed)
    a=build_modular_mappo_trainer(deepcopy(base),'cpu');b=build_modular_mappo_trainer(deepcopy(proposed),'cpu')
    assert a.actor.state_dict().keys()==b.actor.state_dict().keys() and a.critic.state_dict().keys()==b.critic.state_dict().keys()
    assert all(torch.equal(x,y) for x,y in zip(a.actor.state_dict().values(),b.actor.state_dict().values()))
    assert all(torch.equal(x,y) for x,y in zip(a.critic.state_dict().values(),b.critic.state_dict().values()))
    obs=np.zeros((1,4,52),np.float32);alive=np.ones((1,4),np.float32)
    da=a.actor.distribution(torch.as_tensor(obs));db=b.actor.distribution(torch.as_tensor(obs))
    assert torch.equal(da.mean,db.mean) and torch.equal(da.stddev,db.stddev)


def test_launcher_restored_to_historical_strict_serial_order():
    text=Path('tools/run_ws_pbrs_development.sh').read_text(encoding='utf-8')
    assert 'for seed in 5101 5102 5103' in text
    assert 'MAX_PARALLEL' not in text and 'run_seed_pair' not in text
    start=text.index('run_seed()')
    end=text.index('for seed in 5101 5102 5103')
    seed_function=text[start:end]
    assert seed_function.index('run_one baseline') < seed_function.index('run_one pbrs')
