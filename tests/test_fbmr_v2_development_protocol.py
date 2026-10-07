from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
import yaml

from algorithm.modular_mappo.protocol import validate_fbmr_v2_stage2_branch
from algorithm.train_modular_mappo import load_config
from tools.preflight_fbmr_v2_development import validate

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"outputs/formal_eawb/mappo_seed3101/latest.pt"

@pytest.mark.skipif(not torch.cuda.is_available() or not SOURCE.exists(),reason="CUDA/source checkpoint unavailable")
def test_v2_preflight_and_strict_validator():
 result=validate(check_outputs=False,check_cuda=True)
 assert result["status"]=="READY_FOR_FBMR_V2_DEVELOPMENT" and result["planned_runs"]==3
 assert result["comparators_rerun"] is False and result["reserved_33m_executed"] is False
 env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"));state=torch.load(SOURCE,map_location="cuda",weights_only=False);cfg=load_config(ROOT/"configs/dev_fbmr_v2_dual_bound_1200k.yaml")
 mutations=[lambda c:c["training"].update(gamma=.99),lambda c:c["training"].update(actor_learning_rate=3e-4),lambda c:c["modules"]["entity_attention"].update(alpha_rel=.5),lambda c:c["modules"]["wave_balancing"].update(enabled=True),lambda c:c["development_branch"].update(source_sampled_steps=600000)]
 for mutate in mutations:
  bad=deepcopy(cfg);mutate(bad)
  with pytest.raises(RuntimeError):validate_fbmr_v2_stage2_branch(state,env,bad)

def test_v2_manifest_and_launcher_are_exactly_three_runs():
 manifest=json.loads((ROOT/"experiments/fbmr_v2_dual_bound_development_manifest.json").read_text(encoding="utf-8"));runs=manifest["runs"]
 assert len(runs)==3 and [r["source_training_seed"] for r in runs]==[3101,3102,3103]
 assert all(r["method"]=="FBMR-V2 Dual Bound" for r in runs)
 assert manifest["existing_comparators_are_not_rerun"] is True
 assert manifest["validation"]["validation_already_exposed_in_prior_development"] is True
 assert manifest["reserved_untouched_future_final_test"]["executed"] is False
 assert manifest["failure_rule"]["decision"]=="STOP_ENTITY_ACTOR_ARCHITECTURE_SEARCH"
 text=(ROOT/"tools/run_fbmr_v2_development.sh").read_text(encoding="utf-8")
 assert "run_v2_branch 3101" in text and "run_v2_branch 3102" in text and "run_v2_branch 3103" in text
 assert "34000000" in text and "33000000" not in text
 assert "mappo_cont_seed" not in text and "fbmr_seed310" not in text
