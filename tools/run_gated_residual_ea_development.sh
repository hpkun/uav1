#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "uav" ]]; then
  echo "Activate the uav Conda environment before starting development training." >&2
  exit 1
fi

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for development training"; print(torch.cuda.get_device_name(0))'
python -u tools/preflight_gated_residual_ea_development.py
mkdir -p outputs/dev_grea

run_one() {
  local name="$1"
  local config="$2"
  local seed="$3"
  local run_dir="outputs/dev_grea/${name}"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite development run directory: $run_dir" >&2
    exit 1
  fi
  python -u algorithm/train_modular_mappo.py \
    --env-config configs/persistent_wave_v2_environment.yaml \
    --algorithm-config "$config" --output-dir "$run_dir" \
    --device cuda --seed "$seed" --num-envs 24 --total-sampled-steps 400000 \
    > "outputs/dev_grea/${name}_nohup.log" 2>&1
}

run_one mappo_seed4101 configs/dev_grea_mappo_400k.yaml 4101
run_one mappo_seed4102 configs/dev_grea_mappo_400k.yaml 4102
run_one mappo_seed4103 configs/dev_grea_mappo_400k.yaml 4103
run_one full_ea_seed4101 configs/dev_grea_full_ea_400k.yaml 4101
run_one full_ea_seed4102 configs/dev_grea_full_ea_400k.yaml 4102
run_one full_ea_seed4103 configs/dev_grea_full_ea_400k.yaml 4103
run_one residual_ea_seed4101 configs/dev_grea_residual_ea_400k.yaml 4101
run_one residual_ea_seed4102 configs/dev_grea_residual_ea_400k.yaml 4102
run_one residual_ea_seed4103 configs/dev_grea_residual_ea_400k.yaml 4103
run_one gated_ea_seed4101 configs/dev_grea_gated_ea_400k.yaml 4101
run_one gated_ea_seed4102 configs/dev_grea_gated_ea_400k.yaml 4102
run_one gated_ea_seed4103 configs/dev_grea_gated_ea_400k.yaml 4103

