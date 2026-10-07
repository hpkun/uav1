#!/usr/bin/env bash
set -euo pipefail
runtime_sha(){ python -c 'from algorithm.common.protocol import runtime_source_manifest;from pathlib import Path;print(runtime_source_manifest(Path.cwd())["runtime_source_manifest_sha256"])'; }
python -u tools/preflight_w1sg_300k.py
LOCK="$(runtime_sha)"
for seed in 5301 5302 5303; do
  [[ "$(runtime_sha)" == "$LOCK" ]] || { echo "ERROR: runtime source changed" >&2; exit 2; }
  output="outputs/dev_w1sg_current_actor_seed${seed}_300k"
  [[ ! -e "$output" ]] || { echo "Refusing existing output: $output" >&2; exit 2; }
  python -u algorithm/train_modular_mappo.py \
    --env-config configs/persistent_wave_v2_environment.yaml \
    --algorithm-config configs/dev_w1sg_current_actor_300k.yaml \
    --branch-from "outputs/diag_mappo_learnability/l3_seed${seed}/checkpoint_1505280.pt" \
    --output-dir "$output" --device cuda --seed "$seed" --num-envs 24 --total-sampled-steps 1805280
done
