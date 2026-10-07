"""Preflight for the matched Plain transition-mechanism diagnostic."""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.modular_mappo.evaluation import per_wave_episode_diagnostics
from env.factory import make_combat_environment
from tools.plain_transition_diagnostic_common import (
    CHECKPOINTS,EVALUATION_SEEDS,FUTURE_FINAL_RANGE,OUTPUT_DIR,POLICY_SEEDS,
    canonical_spawn_seed,checkpoint_contract,future_rng_seed,
)
import tools.run_plain_transition_mechanism_diagnostic as runner


def same_step(left,right,actions):
    a=left.step(actions);b=right.step(actions)
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
    assert a[2:4]==b[2:4]
    for key in ("red_alive_mask","blue_alive_mask"):
        assert np.array_equal(a[4][key],b[4][key])
    assert a[4]["wave_index"]==b[4]["wave_index"]
    assert left.combat_counts==right.combat_counts
    return a


def clone_checks(env_config):
    env=make_combat_environment(env_config);env.reset(88_100_001)
    for _ in range(7):env.step(np.zeros((4,3),np.float32))
    clone=deepcopy(env);rng=np.random.default_rng(88_100_002)
    for _ in range(30):
        result=same_step(env,clone,rng.uniform(-1,1,(4,3)).astype(np.float32))
        if result[2] or result[3]:break
    transition=make_combat_environment(env_config);transition.reset(88_100_003)
    for blue in transition.blue:blue.alive=False
    result=transition.step(np.zeros((4,3),np.float32))
    assert result[4]["spawned_next_wave"] and transition.wave_index==2
    clone=deepcopy(transition)
    for _ in range(20):
        result=same_step(transition,clone,np.zeros((4,3),np.float32))
        if result[2] or result[3]:break
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",default=str(OUTPUT_DIR));p.add_argument("--overwrite",action="store_true");a=p.parse_args()
    output=Path(a.output_dir)
    checks={};contracts={}
    for seed in POLICY_SEEDS:
        assert CHECKPOINTS[seed].exists();contract=checkpoint_contract(CHECKPOINTS[seed],True);contracts[seed]=contract
        assert contract["training_seed"]==seed and contract["sampled_steps"]==3_000_000
        assert contract["actor_context_dim"]==contract["critic_context_dim"]==0
        assert contract["actor_gru_hidden_dim"]==contract["critic_gru_hidden_dim"]==0
        checks[f"checkpoint_{seed}"]="PASS"
    assert tuple(EVALUATION_SEEDS)==tuple(range(44_000_000,44_000_050));checks["44m_exact"]="PASS"
    manifest=json.loads((ROOT/"experiments/plain_transition_mechanism_diagnostic_manifest.json").read_text(encoding="utf-8"))
    assert manifest["future_final"]=={"start":45_000_000,"end":45_000_199,"executed":False}
    assert FUTURE_FINAL_RANGE==(45_000_000,45_000_199);checks["45m_guard"]="PASS"
    rows=[{"wave_index":1,"wave_cleared":True,"red_survivors_end":4,"blue_survivors_end":0,"red_attack_kills":4,"red_boundary_exits":0,"red_ground_losses":0,"start_step":0,"duration_steps":10},
          {"wave_index":2,"wave_cleared":False,"red_survivors_end":1,"blue_survivors_end":2,"red_attack_kills":2,"red_boundary_exits":0,"red_ground_losses":1,"start_step":10,"duration_steps":20}]
    flat=per_wave_episode_diagnostics({"per_wave_metrics":rows},3)
    assert flat["wave_1_recorded"] and flat["wave_1_cleared"] and flat["wave_2_recorded"] and not flat["wave_2_cleared"] and not flat["wave_3_recorded"]
    checks["explicit_wave_semantics"]="PASS"
    assert clone_checks(contracts[5301]["environment_config"]);checks["deepcopy_fidelity"]="PASS"
    assert future_rng_seed(44_000_007,2)==future_rng_seed(44_000_007,2)
    assert canonical_spawn_seed(44_000_007,2)==1_000_000_072
    assert canonical_spawn_seed(44_000_007,2)!=future_rng_seed(44_000_007,2)
    assert "source_policy_seed" not in inspect.signature(future_rng_seed).parameters and "continuation_policy_seed" not in inspect.signature(future_rng_seed).parameters
    checks["matched_future_rng"]="PASS"
    checks["canonical_spawn_rng"]="PASS"
    source=inspect.getsource(runner)
    assert "trainer.update(" not in source and "optimizer.step(" not in source and ".backward(" not in source
    checks["no_training_or_optimizer"]="PASS"
    required=("direct_case_results.csv","transition_states.csv","transition_agents.csv","ground_risk_cases.csv",
              "death_pretrace.jsonl","replay_integrity.json","continuation_results.csv",
              "canonical_transition_states.csv","canonical_continuation_results.csv","run_metadata.json")
    if output.exists() and not a.overwrite and not all((output/name).is_file() for name in required):
        raise RuntimeError("diagnostic output is incomplete; rerun diagnostic runner with --overwrite")
    else: checks["output_fresh_or_overwrite"]="PASS"
    result={"status":"READY_FOR_PLAIN_TRANSITION_MECHANISM_DIAGNOSTIC","checks":checks}
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
