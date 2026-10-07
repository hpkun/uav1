#!/usr/bin/env bash
set -euo pipefail

[[ "${CONDA_DEFAULT_ENV:-}" == "uav" ]] || { echo "Activate conda environment uav first" >&2; exit 1; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"'
python -u tools/preflight_mappo_baseline_learnability.py --launch-check

mkdir -p outputs/diag_mappo_learnability_corrected

run_one() {
  local condition="$1" seed="$2" env_config="$3"
  local output="outputs/diag_mappo_learnability_corrected/${condition}_seed${seed}"
  [[ ! -e "$output" ]] || { echo "Refusing non-fresh $output" >&2; exit 1; }
  local identity
  identity="$(python -c 'import sys,yaml; from algorithm.train_modular_mappo import load_config; from algorithm.modules.curriculum import CurriculumController; e=yaml.safe_load(open(sys.argv[1])); c=load_config("configs/diag_mappo_learnability_common_3m.yaml"); r=CurriculumController(c["modules"]["curriculum"]).runtime_config(e,0); print(e["persistent_waves"]["total_waves"],r["persistent_waves"]["total_waves"],e["simulation"]["max_steps"],r["simulation"]["max_steps"])' "$env_config")"
  local declared_waves effective_waves declared_steps effective_steps
  read -r declared_waves effective_waves declared_steps effective_steps <<< "$identity"
  echo "[RUN_START] condition=$condition seed=$seed declared_waves=$declared_waves effective_waves=$effective_waves declared_max_steps=$declared_steps effective_max_steps=$effective_steps output=$output"
  python -u algorithm/train_modular_mappo.py \
    --env-config "$env_config" \
    --algorithm-config configs/diag_mappo_learnability_common_3m.yaml \
    --output-dir "$output" \
    --device cuda --seed "$seed" --num-envs 24 --total-sampled-steps 3000000 \
    2>&1 | tee "${output}_nohup.log"
  echo "[RUN_DONE] condition=$condition seed=$seed output=$output"
}

for seed in 5301 5302 5303; do
  run_one l1 "$seed" configs/diag_learnability_1wave_environment.yaml
  run_one l2 "$seed" configs/diag_learnability_2wave_environment.yaml
  run_one l3 "$seed" configs/persistent_wave_v2_environment.yaml
done
