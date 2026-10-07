#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "uav" ]]; then
  echo "Activate the uav Conda environment before starting FBMR V2 development." >&2
  exit 1
fi

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; print(torch.cuda.get_device_name(0))'
python -u tools/preflight_fbmr_v2_development.py
mkdir -p outputs/dev_fbmr_v2

run_v2_branch() {
  local seed="$1"
  local name="fbmr_v2_seed${seed}"
  local source="outputs/formal_eawb/mappo_seed${seed}/latest.pt"
  local output="outputs/dev_fbmr_v2/${name}"
  if [[ -e "$output" ]]; then echo "Refusing to overwrite: $output" >&2; exit 1; fi
  python -u algorithm/train_modular_mappo.py \
    --env-config configs/persistent_wave_v2_environment.yaml \
    --algorithm-config configs/dev_fbmr_v2_dual_bound_1200k.yaml \
    --branch-from "$source" --output-dir "$output" \
    --device cuda --seed "$seed" --num-envs 24 \
    --total-sampled-steps 1200000 \
    > "outputs/dev_fbmr_v2/${name}_nohup.log" 2>&1
}

# Existing MAPPO-continuation and FBMR-V1 comparators are intentionally not rerun.
# The configured deterministic development validation range is 34000000..34000019.
run_v2_branch 3101
run_v2_branch 3102
run_v2_branch 3103
