#!/usr/bin/env bash
set -euo pipefail

[[ "${CONDA_DEFAULT_ENV:-}" == "uav" ]] || { echo "Activate conda environment uav first" >&2; exit 1; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"'
python -u tools/preflight_actor_mission_context.py --summary

mkdir -p outputs/dev_actor_mission_context_3m

run_one() {
  local seed="$1"
  local output="outputs/dev_actor_mission_context_3m/seed${seed}"
  [[ ! -e "$output" ]] || { echo "Refusing non-fresh formal method output: $output" >&2; exit 1; }
  echo "[RUN_START] method=actor_mission_context seed=$seed actor_input_dim=57 critic_context_dim=0 waves=3 max_steps=3000 total_sampled_steps=3000000 output=$output"
  python -u algorithm/train_modular_mappo.py \
    --env-config configs/persistent_wave_v2_environment.yaml \
    --algorithm-config configs/dev_actor_mission_context_3m.yaml \
    --output-dir "$output" \
    --device cuda --seed "$seed" --num-envs 24 --total-sampled-steps 3000000 \
    2>&1 | tee "${output}_nohup.log"
  echo "[RUN_DONE] method=actor_mission_context seed=$seed output=$output"
}

for seed in 5301 5302 5303; do
  run_one "$seed"
done
