#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(5301|5302|5303)$ ]]; then
  echo "Usage: bash tools/run_fire_ready_plain_3m.sh {5301|5302|5303}" >&2
  exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "uav" ]]; then
  echo "ERROR: activate the uav conda environment first" >&2
  exit 2
fi
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is mandatory")
print(torch.cuda.get_device_name(0))
PY

seed="$1"
output="outputs/dev_fire_ready_plain_3m/seed${seed}"
if [[ -e "$output" ]]; then
  echo "ERROR: refuse overwrite existing output: $output" >&2
  exit 2
fi

python -u tools/preflight_fire_ready_plain.py --seed "$seed"
python -u algorithm/train_modular_mappo.py \
  --env-config configs/persistent_wave_v2_fire_ready_environment.yaml \
  --algorithm-config configs/dev_fire_ready_plain_3m.yaml \
  --output-dir "$output" \
  --device cuda \
  --seed "$seed" \
  --num-envs 24 \
  --total-sampled-steps 3000000
