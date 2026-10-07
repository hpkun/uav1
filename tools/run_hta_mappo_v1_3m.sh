#!/usr/bin/env bash
set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" == "" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set" >&2
  exit 2
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is mandatory for HTA-MAPPO V1 development")
PY

for seed in 5301 5302 5303; do
  output="outputs/dev_hta_mappo_v1_3m/seed${seed}"
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing output: $output" >&2
    exit 2
  fi
  python -u algorithm/train_modular_mappo.py \
    --env-config configs/persistent_wave_v2_environment.yaml \
    --algorithm-config configs/dev_hta_mappo_v1_3m.yaml \
    --output-dir "$output" \
    --device cuda \
    --seed "$seed" \
    --total-sampled-steps 3000000
done
