#!/usr/bin/env bash
set -euo pipefail

for seed in 5301 5302 5303; do
  output="outputs/dev_iwsc_mappo_3m/seed${seed}"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing output: $output" >&2
    exit 2
  fi
  python -u algorithm/train_modular_mappo.py \
    --env-config configs/persistent_wave_v2_environment.yaml \
    --algorithm-config configs/dev_iwsc_mappo_3m.yaml \
    --output-dir "$output" \
    --device cuda \
    --seed "$seed" \
    --total-sampled-steps 3000000
done
