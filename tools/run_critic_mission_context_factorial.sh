#!/usr/bin/env bash
set -euo pipefail

[[ "${CONDA_DEFAULT_ENV:-}" == "uav" ]] || { echo "Activate conda environment uav first" >&2; exit 1; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"'
python -u tools/preflight_critic_mission_context_factorial.py --launch-check

mkdir -p outputs/dev_critic_mission_context

run_one() {
  local cell
  local seed
  local config
  local dir
  cell="$1"
  seed="$2"
  config="$3"
  dir="outputs/dev_critic_mission_context/${cell}_seed${seed}"

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
  local seed
  seed="$1"
  run_one c0r0 "$seed" configs/dev_c0r0_plain_mappo_900k.yaml
  run_one c0r1 "$seed" configs/dev_c0r1_pbrs_900k.yaml
  run_one c1r0 "$seed" configs/dev_c1r0_critic_context_900k.yaml
  run_one c1r1 "$seed" configs/dev_c1r1_critic_context_pbrs_900k.yaml
}

for seed in 5201 5202 5203; do
  run_seed "$seed"
done
