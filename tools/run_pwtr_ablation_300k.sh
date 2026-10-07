#!/usr/bin/env bash
set -euo pipefail

runtime_source_sha() {
  python -c 'from algorithm.common.protocol import runtime_source_manifest; from pathlib import Path; print(runtime_source_manifest(Path.cwd())["runtime_source_manifest_sha256"])'
}

BASELINE_RUNTIME_SOURCE_SHA="$(runtime_source_sha)"
branches=(plain stratified current_extra uniform_recent priority_recent full)

for seed in 5301 5302 5303; do
  source_ckpt="outputs/diag_mappo_learnability/l3_seed${seed}/checkpoint_1505280.pt"
  for branch in "${branches[@]}"; do
    current_sha="$(runtime_source_sha)"
    if [[ "$current_sha" != "$BASELINE_RUNTIME_SOURCE_SHA" ]]; then
      echo "ERROR: runtime source tree changed during matched PWTR ablation" >&2
      exit 2
    fi
    config="configs/dev_pwtr_${branch}_300k.yaml"
    output="outputs/dev_pwtr_${branch}_seed${seed}_300k"
    if [[ -e "$output" ]]; then
      echo "Refusing existing output: $output" >&2
      exit 2
    fi
    python -u algorithm/train_modular_mappo.py \
      --env-config configs/persistent_wave_v2_environment.yaml \
      --algorithm-config "$config" \
      --branch-from "$source_ckpt" \
      --output-dir "$output" \
      --device cuda --seed "$seed" --num-envs 24 --total-sampled-steps 1805280
  done
done
