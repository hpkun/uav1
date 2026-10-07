"""Fail-closed preflight for the clean MAPPO single-wave diagnostic."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.single_wave_stability_common import (CROSS_HORIZON_BASE,DIAGNOSTIC_BASE,checkpoint_specs,load_clean_mappo,strict_checkpoint_identity,validate_seed_bank)

def validate(load_cuda=True):
 if not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory for this diagnostic")
 diagnostic=validate_seed_bank(range(DIAGNOSTIC_BASE,DIAGNOSTIC_BASE+50));cross=validate_seed_bank(range(CROSS_HORIZON_BASE,CROSS_HORIZON_BASE+50))
 if set(diagnostic)&set(cross):raise RuntimeError("diagnostic seed banks overlap")
 identities=[]
 for spec in checkpoint_specs():
  identity=strict_checkpoint_identity(spec)
  if load_cuda:
   trainer,loaded=load_clean_mappo(spec,"cuda");del trainer;torch.cuda.empty_cache()
   if loaded["checkpoint_sha256"]!=identity["checkpoint_sha256"]:raise RuntimeError("identity changed during CUDA load")
  identities.append({k:v for k,v in identity.items() if not k.endswith("config")})
 output=ROOT/"outputs/single_wave_checkpoint_trajectory_audit"
 if output.exists():raise FileExistsError(f"formal diagnostic output already exists: {output}")
 return {"status":"READY_FOR_SINGLE_WAVE_STABILITY_DIAGNOSTIC","cuda_device":torch.cuda.get_device_name(0),"checkpoint_count":len(identities),"checkpoint_identities":identities,"diagnostic_seeds":[diagnostic[0],diagnostic[-1]],"cross_horizon_seeds":[cross[0],cross[-1]],"formal_48m_seeds_accessed":False,"output_absent":True}

def main():
 p=argparse.ArgumentParser();p.add_argument("--print-only",action="store_true");p.add_argument("--skip-cuda-load",action="store_true");a=p.parse_args();print(json.dumps(validate(not a.skip_cuda_load),indent=2))
if __name__=="__main__":main()
