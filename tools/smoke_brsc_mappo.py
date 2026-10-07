"""Synthetic mechanism smoke for BRSC-MAPPO V1; not an RL training run."""
from __future__ import annotations
import argparse,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from algorithm.modules import (BoundaryRedistributedSegmentCreditModule,W1_BOUNDARY_TO_W2,
 binary_auroc,redistribute_boundary_credit)
from algorithm.modular_mappo.networks import BoundaryStateOutcomeCritic
from algorithm.modular_mappo.trainer import caiw_trust_cap

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--device",default="cuda",choices=("cuda",));args=parser.parse_args();device=torch.device(args.device)
    if not torch.cuda.is_available():raise RuntimeError("CUDA required but unavailable")
    torch.manual_seed(8803000);rng=np.random.default_rng(8803000);n=512
    obs=torch.as_tensor(rng.normal(size=(n,4,52)),dtype=torch.float32,device=device);alive=torch.ones(n,4,device=device);wave=torch.ones(n,dtype=torch.long,device=device);h=torch.as_tensor(rng.uniform(0,1,n),dtype=torch.float32,device=device)
    labels=(obs[:,:,0].mean(1)>0).float()
    global_before=torch.random.get_rng_state()
    with torch.random.fork_rng(devices=[]):critic=BoundaryStateOutcomeCritic(52,32,2).to(device)
    construction_clean=torch.equal(global_before,torch.random.get_rng_state());optimizer=torch.optim.Adam(critic.parameters(),lr=3e-3)
    initial=float(torch.nn.functional.binary_cross_entropy_with_logits(critic(obs,alive,wave,h)[:,0],labels).detach())
    for _ in range(150):
        loss=torch.nn.functional.binary_cross_entropy_with_logits(critic(obs,alive,wave,h)[:,0],labels);optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():
        logits=critic(obs,alive,wave,h)[:,0];prob=torch.sigmoid(logits);final=float(torch.nn.functional.binary_cross_entropy_with_logits(logits,labels));auc=binary_auroc(labels.cpu(),prob.cpu());brier=float(((prob-labels)**2).mean());clim=float(((labels-labels.mean())**2).mean());bss=1-brier/clim
        good=obs.clone();bad=obs.clone();good[:,:,0]=2;bad[:,:,0]=-2;q_good=float(torch.sigmoid(critic(good,alive,wave,h)[:,0]).mean());q_bad=float(torch.sigmoid(critic(bad,alive,wave,h)[:,0]).mean())
    cfg={"enabled":True,"readiness_consecutive_passes":3,"min_validation_boundaries":64,"min_validation_positive":16,"min_validation_negative":16,"min_validation_auroc":.6,"min_validation_brier_skill":0.}
    module=BoundaryRedistributedSegmentCreditModule(cfg);vm={"validation_segments":n,"validation_positive":int(labels.sum()),"validation_negative":int(n-labels.sum()),"validation_auroc":auc,"validation_brier_skill":bss};ready=[]
    for step in (1,2,3):module.apply_validation(W1_BOUNDARY_TO_W2,vm,step);ready.append(module.task_ready[W1_BOUNDARY_TO_W2])
    waves=np.asarray([[1],[1],[1],[1],[1],[1],[2],[2]],int);dones=np.zeros_like(waves,float);flags=np.zeros_like(waves,float);flags[5,0]=1;credit=np.zeros_like(waves,float);credit[5,0]=.003
    adv,active,lengths=redistribute_boundary_credit(waves,dones,flags,credit,.999*.95)
    _,_,cross_lengths=redistribute_boundary_credit(np.asarray([[1],[1],[2],[2]]),np.zeros((4,1)),np.asarray([[0],[1],[0],[1]]),np.asarray([[0],[.2],[0],[.3]]),.999*.95)
    trusted,_,nt,_=caiw_trust_cap([torch.tensor([2.,0.],device=device)],[torch.tensor([4.,0.],device=device)],.25)
    restored=BoundaryRedistributedSegmentCreditModule(cfg);restored.load_state_dict(module.state_dict())
    checks={"bce_initial":initial,"bce_final":final,"auroc":auc,"brier_skill":bss,"ready_sequence":ready,"q_good":q_good,"q_bad":q_bad,
      "redistribution_exact":bool(abs(adv[5,0]-.003)<1e-8 and abs(adv[4,0]-.003*(.999*.95))<1e-8),"segment_length":lengths,"no_cross_wave_lengths":cross_lengths,
      "raw_point003":float(adv[5,0]),"trust_cap_norm":float(torch.linalg.vector_norm(trusted[0])),"trust_limit":float(.25*nt),
      "checkpoint_roundtrip":restored.task_ready==module.task_ready,"global_rng_clean":construction_clean,"finite":bool(np.isfinite([initial,final,auc,bss,q_good,q_bad]).all())}
    passed=(final<initial and auc>.6 and bss>0 and ready==[False,False,True] and q_good>q_bad and checks["redistribution_exact"] and cross_lengths==[2,2] and checks["trust_cap_norm"]<=checks["trust_limit"]+1e-6 and checks["checkpoint_roundtrip"] and construction_clean and checks["finite"])
    print(json.dumps({"status":"BRSC_SYNTHETIC_SMOKE_PASS" if passed else "BRSC_SYNTHETIC_SMOKE_FAIL",**checks},indent=2));raise SystemExit(0 if passed else 2)
if __name__=="__main__":main()
