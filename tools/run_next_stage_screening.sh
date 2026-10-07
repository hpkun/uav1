#!/usr/bin/env bash
set -euo pipefail

ENV_CONFIG="configs/persistent_wave_v2_environment.yaml"
DIRECT_BEST="${DIRECT_BEST:-outputs/d999_seed2023/best_eval.pt}"

RUN_NAMES=(
  pw_alloff_matched_1p5m_seed2023
  pw_m6_screen_control_300k_seed2023
  pw_m6_m8_anchor_c0001_300k_seed2023
  pw_m6_m8_anchor_c0003_300k_seed2023
  pw_m6_m8_anchor_c001_300k_seed2023
  pw_m6_m8_anchor_c003_300k_seed2023
  pw_m6_m8_anchor_c01_300k_seed2023
)

if [[ ! -f "$DIRECT_BEST" ]]; then
  echo "Direct source checkpoint is missing: $DIRECT_BEST" >&2
  exit 1
fi

# Complete every collision check before the first training process starts.
for name in "${RUN_NAMES[@]}"; do
  run_dir="outputs/${name}"
  log_path="outputs/${name}_nohup.log"
  if [[ -d "$run_dir" ]] && [[ -n "$(find "$run_dir" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to start: non-empty run directory: $run_dir" >&2
    exit 1
  fi
  if [[ -e "$run_dir" ]] && [[ ! -d "$run_dir" ]]; then
    echo "Refusing to start: run target is not a directory: $run_dir" >&2
    exit 1
  fi
  if [[ -e "$log_path" ]]; then
    echo "Refusing to overwrite existing nohup log: $log_path" >&2
    exit 1
  fi
done

direct_sha256="$(sha256sum "$DIRECT_BEST" | awk '{print $1}')"
echo "Direct source: $DIRECT_BEST"
echo "Direct source SHA-256: $direct_sha256"
python -u tools/analyze_next_stage_screening.py \
  --preflight-sources "$DIRECT_BEST" "$DIRECT_BEST"

mkdir -p outputs

run_one() {
  local name="$1"
  local config="$2"
  local budget="$3"
  shift 3
  python -u algorithm/train_modular_mappo.py \
    --device cuda \
    --seed 2023 \
    --num-envs 24 \
    --total-sampled-steps "$budget" \
    --env-config "$ENV_CONFIG" \
    --algorithm-config "$config" \
    --output-dir "outputs/${name}" \
    "$@" \
    > "outputs/${name}_nohup.log" 2>&1
  [[ -f "outputs/${name}/latest.pt" ]] || { echo "Missing latest.pt for $name" >&2; exit 1; }
  [[ -f "outputs/${name}/run_summary.json" ]] || { echo "Missing run_summary.json for $name" >&2; exit 1; }
}

run_one pw_alloff_matched_1p5m_seed2023 \
  configs/pw_alloff_matched_1p5m.yaml 1500000

run_one pw_m6_screen_control_300k_seed2023 \
  configs/pw_m6_screen_control_300k.yaml 300000 \
  --warm-start-checkpoint "$DIRECT_BEST"

run_one pw_m6_m8_anchor_c0001_300k_seed2023 \
  configs/pw_m6_m8_anchor_c0001_300k.yaml 300000 \
  --warm-start-checkpoint "$DIRECT_BEST" --reference-checkpoint "$DIRECT_BEST"
run_one pw_m6_m8_anchor_c0003_300k_seed2023 \
  configs/pw_m6_m8_anchor_c0003_300k.yaml 300000 \
  --warm-start-checkpoint "$DIRECT_BEST" --reference-checkpoint "$DIRECT_BEST"
run_one pw_m6_m8_anchor_c001_300k_seed2023 \
  configs/pw_m6_m8_anchor_c001_300k.yaml 300000 \
  --warm-start-checkpoint "$DIRECT_BEST" --reference-checkpoint "$DIRECT_BEST"
run_one pw_m6_m8_anchor_c003_300k_seed2023 \
  configs/pw_m6_m8_anchor_c003_300k.yaml 300000 \
  --warm-start-checkpoint "$DIRECT_BEST" --reference-checkpoint "$DIRECT_BEST"
run_one pw_m6_m8_anchor_c01_300k_seed2023 \
  configs/pw_m6_m8_anchor_c01_300k.yaml 300000 \
  --warm-start-checkpoint "$DIRECT_BEST" --reference-checkpoint "$DIRECT_BEST"

echo "All next-stage screening runs completed successfully."
