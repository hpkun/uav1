#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == "uav" ]] || { echo "Activate conda environment uav first" >&2; exit 1; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"'
python -u tools/preflight_ws_pbrs_development.py
mkdir -p outputs/dev_ws_pbrs
run_one() {
  local method="$1"
  local seed="$2"
  local config="$3"
  local dir="outputs/dev_ws_pbrs/${method}_seed${seed}"

  [[ ! -e "$dir" ]] || {
    echo "Refusing non-fresh $dir" >&2
    exit 1
  }

  python -u algorithm/train_modular_mappo.py \
    --env-config configs/persistent_wave_v2_environment.yaml \
    --algorithm-config "$config" \
    --output-dir "$dir" \
    --device cuda \
    --seed "$seed" \
    --num-envs 24 \
    --total-sampled-steps 900000 \
    > "${dir}_nohup.log" 2>&1
}

run_seed() {
  local seed="$1"
  run_one baseline "$seed" configs/dev_ws_pbrs_mappo_baseline_900k.yaml
  run_one pbrs "$seed" configs/dev_ws_pbrs_proposed_900k.yaml
}

for seed in 5101 5102 5103; do
  run_seed "$seed"
done
