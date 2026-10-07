#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "uav" ]]; then
  echo "Activate the uav Conda environment before starting FBMR Stage-2 development." >&2
  exit 1
fi

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; print(torch.cuda.get_device_name(0))'
python -u tools/preflight_fbmr_stage2_development.py
mkdir -p outputs/dev_fbmr_stage2/source_reference

source_reference() {
  local seed="$1"
  python -u algorithm/evaluate_modular_mappo.py \
    --checkpoint "outputs/formal_eawb/mappo_seed${seed}/latest.pt" \
    --env-config configs/persistent_wave_v2_environment.yaml \
    --episodes 20 --seed-base 34000000 --device cuda \
    --output "outputs/dev_fbmr_stage2/source_reference/source_reference_seed${seed}.json"
}

run_branch() {
  local name="$1" config="$2" seed="$3"
  local source="outputs/formal_eawb/mappo_seed${seed}/latest.pt"
  local output="outputs/dev_fbmr_stage2/${name}"
  if [[ -e "$output" ]]; then echo "Refusing to overwrite: $output" >&2; exit 1; fi
  python -u algorithm/train_modular_mappo.py \
    --env-config configs/persistent_wave_v2_environment.yaml \
    --algorithm-config "$config" --branch-from "$source" \
    --output-dir "$output" --device cuda --seed "$seed" --num-envs 24 \
    --total-sampled-steps 1200000 \
    > "outputs/dev_fbmr_stage2/${name}_nohup.log" 2>&1
}

source_reference 3101
source_reference 3102
source_reference 3103

run_branch mappo_cont_seed3101 configs/dev_fbmr_mappo_continuation_1200k.yaml 3101
run_branch fbmr_seed3101 configs/dev_fbmr_mean_residual_1200k.yaml 3101
run_branch mappo_cont_seed3102 configs/dev_fbmr_mappo_continuation_1200k.yaml 3102
run_branch fbmr_seed3102 configs/dev_fbmr_mean_residual_1200k.yaml 3102
run_branch mappo_cont_seed3103 configs/dev_fbmr_mappo_continuation_1200k.yaml 3103
run_branch fbmr_seed3103 configs/dev_fbmr_mean_residual_1200k.yaml 3103
