#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "uav" ]]; then
  echo "Activate the uav Conda environment before starting formal training." >&2
  exit 1
fi

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for formal training"; print(torch.cuda.get_device_name(0))'
python -u tools/preflight_eawb_formal_multiseed.py
mkdir -p outputs/formal_eawb

run_one() {
  local name="$1"
  local config="$2"
  local seed="$3"
  local run_dir="outputs/formal_eawb/${name}"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite formal run directory: $run_dir" >&2
    exit 1
  fi
  python -u algorithm/train_modular_mappo.py \
    --device cuda --seed "$seed" --num-envs 24 --total-sampled-steps 900000 \
    --env-config configs/persistent_wave_v2_environment.yaml \
    --algorithm-config "$config" --output-dir "$run_dir" \
    > "outputs/formal_eawb/${name}_nohup.log" 2>&1
}

run_one mappo_seed3101 configs/formal_eawb_mappo_900k.yaml 3101
run_one mappo_seed3102 configs/formal_eawb_mappo_900k.yaml 3102
run_one mappo_seed3103 configs/formal_eawb_mappo_900k.yaml 3103
run_one wb_seed3101 configs/formal_eawb_wb_900k.yaml 3101
run_one wb_seed3102 configs/formal_eawb_wb_900k.yaml 3102
run_one wb_seed3103 configs/formal_eawb_wb_900k.yaml 3103
run_one ea_seed3101 configs/formal_eawb_ea_900k.yaml 3101
run_one ea_seed3102 configs/formal_eawb_ea_900k.yaml 3102
run_one ea_seed3103 configs/formal_eawb_ea_900k.yaml 3103
run_one ea_wb_seed3101 configs/formal_eawb_ea_wb_900k.yaml 3101
run_one ea_wb_seed3102 configs/formal_eawb_ea_wb_900k.yaml 3102
run_one ea_wb_seed3103 configs/formal_eawb_ea_wb_900k.yaml 3103
run_one ea_wb_fixed_seed3101 configs/formal_eawb_ea_wb_fixed_900k.yaml 3101
run_one ea_wb_fixed_seed3102 configs/formal_eawb_ea_wb_fixed_900k.yaml 3102
run_one ea_wb_fixed_seed3103 configs/formal_eawb_ea_wb_fixed_900k.yaml 3103

