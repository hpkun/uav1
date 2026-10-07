from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
import yaml

from algorithm.modular_mappo.protocol import validate_fbmr_stage2_branch
from algorithm.train_modular_mappo import load_config
from tools.preflight_fbmr_stage2_development import validate

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"outputs/formal_eawb/mappo_seed3101/latest.pt"

@pytest.mark.skipif(not torch.cuda.is_available() or not SOURCE.exists(),reason="CUDA/source checkpoint unavailable")
def test_preflight_and_strict_branch_whitelist():
 result=validate(check_outputs=False,check_cuda=True,check_existing_34m=True)
 assert result["status"]=="READY_FOR_FBMR_STAGE2_DEVELOPMENT" and result["planned_runs"]==6
 assert result["validation_seed_range"]==[34000000,34000019] and result["reserved_33m_executed"] is False
 state=torch.load(SOURCE,map_location="cuda",weights_only=False);env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"));config=load_config(ROOT/"configs/dev_fbmr_mean_residual_1200k.yaml")
 assert validate_fbmr_stage2_branch(state,env,config)["intervention"]=="frozen_base_mean_residual"
 mutations=[lambda c:c["training"].update(gamma=.99),lambda c:c["network"].update(attention_heads=1),lambda c:c["modules"]["wave_balancing"].update(enabled=True),lambda c:c["modules"]["entity_attention"].update(max_mean_correction=.5)]
 for mutate in mutations:
  bad=deepcopy(config);mutate(bad)
  with pytest.raises(RuntimeError):validate_fbmr_stage2_branch(state,env,bad)

def test_manifest_is_exact_paired_stage2_matrix():
 manifest=json.loads((ROOT/"experiments/fbmr_stage2_development_manifest.json").read_text(encoding="utf-8"))
 assert len(manifest["runs"])==6 and manifest["source_training_seeds"]==[3101,3102,3103]
 assert [(r["method"],r["source_training_seed"]) for r in manifest["runs"]]==[(m,s) for s in (3101,3102,3103) for m in ("MAPPO Continuation","FBMR-EA")]
 assert manifest["reserved_untouched_future_final_test"]["executed"] is False

def test_launcher_is_paired_and_never_uses_33m():
 text=(ROOT/"tools/run_fbmr_stage2_development.sh").read_text(encoding="utf-8")
 expected=[f"{method}_seed{seed}" for seed in (3101,3102,3103) for method in ("mappo_cont","fbmr")]
 assert [text.index(x) for x in expected]==sorted(text.index(x) for x in expected)
 assert "34000000" in text and "33000000" not in text and "1200000" in text
