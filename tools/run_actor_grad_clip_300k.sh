#!/usr/bin/env bash
set -euo pipefail
runtime_sha(){ python -c 'from algorithm.common.protocol import runtime_source_manifest;from pathlib import Path;print(runtime_source_manifest(Path.cwd())["runtime_source_manifest_sha256"])'; }
python -u tools/preflight_actor_grad_clip_300k.py
LOCK="$(runtime_sha)"
for seed in 5301 5302 5303; do
  source="outputs/diag_mappo_learnability/l3_seed${seed}/checkpoint_1505280.pt"
  source_sha="$(sha256sum "$source" | awk '{print $1}')"
  for variant in 05 10; do
    [[ "$(runtime_sha)" == "$LOCK" ]] || { echo "ERROR: runtime source changed" >&2; exit 2; }
    if [[ "$variant" == "05" ]]; then config="configs/dev_actor_grad_clip_05_control_300k.yaml"; output="outputs/dev_actor_clip05_seed${seed}_300k"; else config="configs/dev_actor_grad_clip_10_300k.yaml"; output="outputs/dev_actor_clip10_seed${seed}_300k"; fi
    [[ ! -e "$output" ]] || { echo "Refusing existing output: $output" >&2; exit 2; }
    [[ "$(sha256sum "$source" | awk '{print $1}')" == "$source_sha" ]] || { echo "ERROR: source checkpoint changed" >&2; exit 2; }
    python -u algorithm/train_modular_mappo.py --env-config configs/persistent_wave_v2_environment.yaml --algorithm-config "$config" --branch-from "$source" --output-dir "$output" --device cuda --seed "$seed" --num-envs 24 --total-sampled-steps 1805280
  done
done
