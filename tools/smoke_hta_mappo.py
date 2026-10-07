"""Synthetic HTA-MAPPO V1 smoke; no environment rollout or formal seeds."""
from pathlib import Path
import sys,tempfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modules import compute_smdp_gae,discounted_macro_reward


def build(config,seed=8804001):
    cfg=load_config(config);cfg["training"]["seed"]=seed
    return build_modular_mappo_trainer(cfg,"cpu",32,1000)


def main():
    plain=build("configs/diag_mappo_learnability_common_3m.yaml")
    hta=build("configs/dev_hta_mappo_v1_3m.yaml")
    checks={}
    checks["plain_actor_equal"]=all(torch.equal(v,hta.actor.state_dict()[k]) for k,v in plain.actor.state_dict().items())
    checks["plain_critic_equal"]=all(torch.equal(v,hta.critic.state_dict()[k]) for k,v in plain.critic.state_dict().items())
    checks["residual_zero"]=all(torch.count_nonzero(p)==0 for head in hta.actor.option_mean_residuals for p in head.parameters())
    checks["critic_projection_zero"]=torch.count_nonzero(hta.critic.mission_context_projection)==0
    obs=np.zeros((2,4,52),np.float32);alive=np.ones((2,4),np.float32)
    checks["uniform_manager"]=np.allclose(hta.manager_act(obs,alive,True),0)
    rng=torch.get_rng_state().clone();hta.manager_act(obs,alive,False);checks["manager_rng_isolated"]=torch.equal(rng,torch.get_rng_state())
    with torch.no_grad():hta.actor.option_mean_residuals[2].bias.fill_(.2)
    z0=torch.zeros(2,4,dtype=torch.long);z2=torch.full((2,4),2);tobs=torch.as_tensor(obs)
    checks["option_controllable"]=not torch.equal(hta.actor.distribution(tobs,option_ids=z0).mean,hta.actor.distribution(tobs,option_ids=z2).mean)
    checks["macro_reward_exact"]=np.isclose(discounted_macro_reward([1.,2.,3.],.9),1+.9*2+.9**2*3)
    advantage,_,_=compute_smdp_gae([[1.]],[[.5]],[[2.]],[3],[[1.]],[[0.]],.9,.95)
    checks["rollout_bootstrap_no_trace"]=np.isclose(advantage[0,0],1+.9**3*2-.5)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"hta.pt";hta.save(path);restored=build("configs/dev_hta_mappo_v1_3m.yaml")
        restored.load(path);checks["checkpoint_roundtrip"]=torch.equal(hta.manager_actor.logits(tobs),restored.manager_actor.logits(tobs)) and torch.equal(hta.hta_manager_generator.get_state(),restored.hta_manager_generator.get_state())
    failed=[key for key,value in checks.items() if not bool(value)]
    if failed:raise RuntimeError("HTA synthetic smoke failed: "+", ".join(failed))
    print("HTA_SYNTHETIC_SMOKE_PASS")


if __name__=="__main__":main()

