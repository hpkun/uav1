#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -u tools/preflight_mission_aware_film.py --summary
mkdir -p outputs/dev_mission_aware_film_3m

for seed in 5301 5302 5303; do
  output="outputs/dev_mission_aware_film_3m/seed${seed}"
  echo "[RUN_START] method=mission_aware_film seed=${seed} output=${output}"
  python -u algorithm/train_modular_mappo.py \
    --env-config configs/persistent_wave_v2_environment.yaml \
    --algorithm-config configs/dev_mission_aware_film_3m.yaml \
    --output-dir "${output}" --device cuda --seed "${seed}" \
    --num-envs 24 --total-sampled-steps 3000000 2>&1 | tee "outputs/dev_mission_aware_film_3m/seed${seed}_launcher.log"
  echo "[RUN_DONE] method=mission_aware_film seed=${seed} output=${output}"
done
