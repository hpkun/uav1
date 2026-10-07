"""Synthetic mechanism smoke for CAIW-MAPPO V2 (never an RL training run)."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from algorithm.modules import CounterfactualInterWaveCreditModule,W1_TO_W2,binary_auroc,prior_corrected_probability
from algorithm.modular_mappo.networks import InterWaveActionOutcomeCritic
from algorithm.modular_mappo.trainer import antithetic_latent_actions,caiw_trust_cap

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--device",default="cuda");args=parser.parse_args();device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA required but unavailable")
    torch.manual_seed(8802000);rng=np.random.default_rng(8802000);n=512
    obs=torch.as_tensor(rng.normal(size=(n,4,52)),dtype=torch.float32,device=device);actions=torch.as_tensor(rng.uniform(-1,1,size=(n,4,3)),dtype=torch.float32,device=device);alive=torch.ones(n,4,device=device);wave=torch.ones(n,dtype=torch.long,device=device);h=torch.as_tensor(rng.uniform(0,1,size=n),dtype=torch.float32,device=device)
    labels=((obs[:,0,0]+1.5*actions[:,0,0])>0).float();critic=InterWaveActionOutcomeCritic(52,3,32,2).to(device);optimizer=torch.optim.Adam(critic.parameters(),lr=3e-3)
    initial=float(torch.nn.functional.binary_cross_entropy_with_logits(critic(obs,actions,alive,wave,h)[:,0],labels))
    for _ in range(150):
        loss=torch.nn.functional.binary_cross_entropy_with_logits(critic(obs,actions,alive,wave,h)[:,0],labels);optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():
        logits=critic(obs,actions,alive,wave,h)[:,0];prob=torch.sigmoid(logits);final=float(torch.nn.functional.binary_cross_entropy_with_logits(logits,labels));auc=binary_auroc(labels.cpu().numpy(),prob.cpu().numpy());brier=float(((prob-labels)**2).mean());clim=float(((labels-labels.mean())**2).mean());bss=1-brier/clim
        good=actions.clone();good[:,0,0]=1;bad=actions.clone();bad[:,0,0]=-1;q_good=torch.sigmoid(critic(obs,good,alive,wave,h)[:,0]);q_bad=torch.sigmoid(critic(obs,bad,alive,wave,h)[:,0])
    module=CounterfactualInterWaveCreditModule({"enabled":True,"readiness_consecutive_passes":3,"min_validation_segments":64,"min_validation_positive_segments":16,"min_validation_negative_segments":16,"min_validation_auroc":.6,"min_validation_brier_skill":0.})
    vm={"validation_segments":n,"validation_positive":int(labels.sum()),"validation_negative":int(n-labels.sum()),"validation_auroc":auc,"validation_brier_skill":bss}
    ready_sequence=[]
    for step in (1,2,3):module.apply_validation(W1_TO_W2,vm,step);ready_sequence.append(module.task_ready[W1_TO_W2])
    state=torch.random.get_rng_state();samples=antithetic_latent_actions(torch.zeros(8,3),torch.ones(8,3),rng);rng_clean=torch.equal(state,torch.random.get_rng_state())
    trusted,scale,nt,_=caiw_trust_cap([torch.tensor([2.,0.],device=device)],[torch.tensor([4.,0.],device=device)],.25)
    raw_scale=float(torch.tensor(.503)-torch.tensor(.500));roundtrip=CounterfactualInterWaveCreditModule(module.config);roundtrip.load_state_dict(module.state_dict())
    checks={"bce_decreased":final<initial,"auroc":auc,"brier_skill":bss,"gate_sequence":ready_sequence,"good_minus_bad":float((q_good-q_bad).mean()),"raw_scale":raw_scale,"trust_cap_norm":float(torch.linalg.vector_norm(trusted[0])),"trust_limit":float(.25*nt),"checkpoint_roundtrip":roundtrip.task_ready==module.task_ready,"finite":bool(np.isfinite([initial,final,auc,bss]).all()),"global_rng_clean":rng_clean,"antithetic":bool(torch.allclose(samples[:,0]+samples[:,1],torch.zeros_like(samples[:,0])) and torch.allclose(samples[:,2]+samples[:,3],torch.zeros_like(samples[:,2])))}
    passed=checks["bce_decreased"] and auc>.6 and bss>0 and ready_sequence==[False,False,True] and checks["good_minus_bad"]>0 and abs(raw_scale-.003)<1e-6 and checks["trust_cap_norm"]<=checks["trust_limit"]+1e-6 and checks["checkpoint_roundtrip"] and checks["finite"] and checks["global_rng_clean"] and checks["antithetic"]
    print(json.dumps({"status":"CAIW_SYNTHETIC_SMOKE_PASS" if passed else "CAIW_SYNTHETIC_SMOKE_FAIL",**checks},indent=2));raise SystemExit(0 if passed else 2)
if __name__=="__main__":main()
