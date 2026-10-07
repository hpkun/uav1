"""Small CPU synthetic smoke for IWSC mechanics; never runs the environment."""
from __future__ import annotations
import json,sys,tempfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer,asymmetric_tactical_projection

CFG={"enabled":True,"max_waves":3,"quality_critic_learning_rate":.003,"actor_credit_coefficient":1.,
 "replay_segments_per_wave":128,"max_states_per_segment":16,"critic_minibatch_size":32,"critic_updates_per_rollout":4,
 "min_completed_segments_per_wave":4,"min_target_std":.05,"huber_delta":1.,"wave_balanced_supervision":True,
 "wave_balanced_actor_loss":True,"gradient_projection":"asymmetric_tactical_preserving"}

def main():
    trainer=ModularMAPPOTrainer(hidden_dim=16,seed=8801,ppo_epochs=1,minibatch_size=8,modules_config={"inter_wave_credit":CFG})
    rng=np.random.default_rng(10);segments=[]
    for wave in (1,2):
      for target in (0.,1.,0.,1.):
        states=[{"observation":np.full((4,52),target+(wave*.1),np.float32),"alive_mask":np.ones(4,np.float32),"horizon":.7,"boundary":i==7} for i in range(8)]
        segments.append(trainer.inter_wave_credit.cap_segment(states,target,wave))
    trainer.inter_wave_credit.ingest(segments)
    sample=trainer.inter_wave_credit.sample_states(1,32,np.random.default_rng(1));tt=lambda x,d=torch.float32:torch.as_tensor(x,dtype=d)
    def loss():return torch.nn.functional.huber_loss(trainer.iw_critic(tt(sample["observations"]),tt(sample["alive_masks"]),tt(sample["credit_waves"],torch.long),tt(sample["horizons"])),tt(sample["targets"])).item()
    before=loss()
    for _ in range(25):trainer._train_iw_critic()
    after=loss()
    t,e,a=4,2,4;obs=rng.normal(size=(t,e,a,52)).astype("f");next_obs=obs+.05;alive=np.ones((t,e,a),"f");raw=rng.normal(size=(t,e,a,3)).astype("f");actions=np.tanh(raw).astype("f")
    with torch.no_grad():old=trainer.actor._squashed_log_prob(trainer.actor.distribution(torch.tensor(obs)),torch.tensor(raw),torch.tensor(actions)).numpy()
    zeros=np.zeros((t,e),"f");ctx=np.zeros((t,e,0),"f");waves=np.asarray([[1,2]]*t)
    batch=ModularRolloutBatch(obs,actions,raw,old,rng.normal(size=(t,e,a)).astype("f"),np.zeros((t,e,a),"f"),zeros,alive,next_obs,alive.copy(),waves,np.full((t,e),3),ctx,ctx,episode_masks=np.ones((t,e),"f"),remaining_horizons=np.full((t,e),.7,"f"),next_remaining_horizons=np.full((t,e),.69,"f"),wave_transition_flags=zeros,iw_supervision_segments=[])
    critic_before=[p.detach().clone() for p in trainer.critic.parameters()];metrics=trainer.update(batch)
    projected,dot=asymmetric_tactical_projection([torch.tensor([1.,0.])],[torch.tensor([-1.,1.])])
    with tempfile.TemporaryDirectory() as directory:
      path=Path(directory)/"roundtrip.pt";trainer.save(path);other=ModularMAPPOTrainer(hidden_dim=16,seed=8801,ppo_epochs=1,minibatch_size=8,modules_config={"inter_wave_credit":CFG});other.load(path)
      roundtrip=all(torch.equal(v,other.iw_critic.state_dict()[k]) for k,v in trainer.iw_critic.state_dict().items())
    checks={"critic_loss_decreased":after<before,"ready_wave1":trainer.inter_wave_credit.ready(1),"ready_wave2":trainer.inter_wave_credit.ready(2),
      "actor_iw_update":metrics["iw_actor_active"]==1,"projection_path":dot<0 and abs(float(torch.dot(torch.tensor([1.,0.]),projected[0])))<1e-6,
      "tactical_critic_updated":any(not torch.equal(a,b) for a,b in zip(critic_before,trainer.critic.parameters())),"checkpoint_roundtrip":roundtrip,
      "finite":bool(np.isfinite(list(metrics.values())).all())}
    if not all(checks.values()):raise RuntimeError(checks)
    print(json.dumps({"before_loss":before,"after_loss":after,"checks":checks,"result":"IWSC_SYNTHETIC_SMOKE_PASS"},indent=2))
if __name__=="__main__":main()
