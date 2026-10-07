#!/usr/bin/env bash
set -euo pipefail

ENV_CONFIG="configs/persistent_wave_v2_environment.yaml"

if [[ -z "${DIRECT_2024:-}" ]]; then
  DIRECT_2024="$(python -u tools/analyze_multiseed_confirmation.py --print-direct-source 2024)"
fi
if [[ -z "${DIRECT_2025:-}" ]]; then
  DIRECT_2025="$(python -u tools/analyze_multiseed_confirmation.py --print-direct-source 2025)"
fi

RUN_NAMES=(
  pw_alloff_matched_1p5m_seed2024
  pw_m5_wave_balance_1p5m_seed2024
  pw_alloff_matched_1p5m_seed2025
  pw_m5_wave_balance_1p5m_seed2025
  pw_m6_screen_control_300k_seed2024
  pw_m6_m8_anchor_c003_300k_seed2024
  pw_m6_screen_control_300k_seed2025
  pw_m6_m8_anchor_c003_300k_seed2025
)

# Refuse every collision before the first training process starts.
for name in "${RUN_NAMES[@]}"; do
  run_dir="outputs/${name}"
  log_path="outputs/${name}_nohup.log"
  if [[ -d "$run_dir" ]] && [[ -n "$(find "$run_dir" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to start: non-empty run directory: $run_dir" >&2; exit 1
  fi
  if [[ -e "$run_dir" ]] && [[ ! -d "$run_dir" ]]; then
    echo "Refusing to start: run target is not a directory: $run_dir" >&2; exit 1
  fi
  if [[ -e "$log_path" ]]; then
    echo "Refusing to overwrite existing nohup log: $log_path" >&2; exit 1
  fi
done

python -u tools/analyze_multiseed_confirmation.py --preflight \
  --direct-2024 "$DIRECT_2024" --direct-2025 "$DIRECT_2025"

mkdir -p outputs

run_one() {
  local name="$1" config="$2" seed="$3" budget="$4"
  shift 4
  python -u algorithm/train_modular_mappo.py \
    --device cuda --seed "$seed" --num-envs 24 --total-sampled-steps "$budget" \
    --env-config "$ENV_CONFIG" --algorithm-config "$config" \
    --output-dir "outputs/${name}" "$@" \
    > "outputs/${name}_nohup.log" 2>&1
  for artifact in latest.pt best_eval.pt run_summary.json run_config.json; do
    [[ -f "outputs/${name}/${artifact}" ]] || {
      echo "Missing ${artifact} for ${name}" >&2; exit 1;
    }
  done
}

run_one pw_alloff_matched_1p5m_seed2024 configs/pw_alloff_matched_1p5m.yaml 2024 1500000
run_one pw_m5_wave_balance_1p5m_seed2024 configs/pw_m5_wave_balance.yaml 2024 1500000
run_one pw_alloff_matched_1p5m_seed2025 configs/pw_alloff_matched_1p5m.yaml 2025 1500000
run_one pw_m5_wave_balance_1p5m_seed2025 configs/pw_m5_wave_balance.yaml 2025 1500000

run_one pw_m6_screen_control_300k_seed2024 configs/pw_m6_screen_control_300k.yaml 2024 300000 \
  --warm-start-checkpoint "$DIRECT_2024"
run_one pw_m6_m8_anchor_c003_300k_seed2024 configs/pw_m6_m8_anchor_c003_300k.yaml 2024 300000 \
  --warm-start-checkpoint "$DIRECT_2024" --reference-checkpoint "$DIRECT_2024"
run_one pw_m6_screen_control_300k_seed2025 configs/pw_m6_screen_control_300k.yaml 2025 300000 \
  --warm-start-checkpoint "$DIRECT_2025"
run_one pw_m6_m8_anchor_c003_300k_seed2025 configs/pw_m6_m8_anchor_c003_300k.yaml 2025 300000 \
  --warm-start-checkpoint "$DIRECT_2025" --reference-checkpoint "$DIRECT_2025"

echo "All 3-training-seed confirmation runs completed successfully."
