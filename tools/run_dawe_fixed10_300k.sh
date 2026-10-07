#!/usr/bin/env bash
set -euo pipefail

[[ "${CONDA_DEFAULT_ENV:-}" == "uav" ]] || { echo "ERROR: activate conda environment uav first" >&2; exit 2; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
python -u tools/preflight_dawe_fixed10_300k.py --print-only

lock_snapshot() {
  python - <<'PY'
from pathlib import Path
from algorithm.common.protocol import config_sha256, runtime_source_manifest
from algorithm.train_modular_mappo import load_config
root = Path.cwd()
runtime = runtime_source_manifest(root)["runtime_source_manifest_sha256"]
environment = config_sha256(load_config(root / "configs/persistent_wave_v2_environment.yaml"))
control = config_sha256(load_config(root / "configs/dev_dawe_fixed10_control_300k.yaml"))
treatment = config_sha256(load_config(root / "configs/dev_dawe_fixed10_v1_300k.yaml"))
print(runtime, environment, control, treatment)
PY
}

read -r LOCK_RUNTIME_SOURCE_SHA LOCK_ENVIRONMENT_CONFIG_SHA LOCK_CONTROL_CONFIG_SHA LOCK_TREATMENT_CONFIG_SHA < <(lock_snapshot)
echo "[LOCK] runtime_source_sha=${LOCK_RUNTIME_SOURCE_SHA}"
echo "[LOCK] env_config_sha=${LOCK_ENVIRONMENT_CONFIG_SHA}"
echo "[LOCK] control_config_sha=${LOCK_CONTROL_CONFIG_SHA}"
echo "[LOCK] treatment_config_sha=${LOCK_TREATMENT_CONFIG_SHA}"

check_all_locks() {
  local seed="$1" variant="$2" source="$3" expected_source_sha="$4"
  local current_runtime current_environment current_control current_treatment current_source
  read -r current_runtime current_environment current_control current_treatment < <(lock_snapshot)
  [[ "$current_runtime" == "$LOCK_RUNTIME_SOURCE_SHA" ]] || { echo "ERROR: runtime source lock changed" >&2; exit 2; }
  [[ "$current_environment" == "$LOCK_ENVIRONMENT_CONFIG_SHA" ]] || { echo "ERROR: environment config lock changed" >&2; exit 2; }
  [[ "$current_control" == "$LOCK_CONTROL_CONFIG_SHA" ]] || { echo "ERROR: Control config lock changed" >&2; exit 2; }
  [[ "$current_treatment" == "$LOCK_TREATMENT_CONFIG_SHA" ]] || { echo "ERROR: Treatment config lock changed" >&2; exit 2; }
  current_source="$(sha256sum "$source" | awk '{print $1}')"
  [[ "$current_source" == "$expected_source_sha" ]] || { echo "ERROR: source checkpoint changed" >&2; exit 2; }
  echo "[LOCK_CHECK] PASS seed=${seed} variant=${variant}"
}

for seed in 5301 5302 5303; do
  source="outputs/diag_mappo_learnability/l3_seed${seed}/checkpoint_1505280.pt"
  source_sha="$(sha256sum "$source" | awk '{print $1}')"
  for variant in control v1; do
    if [[ "$variant" == "control" ]]; then
      config="configs/dev_dawe_fixed10_control_300k.yaml"
      output="outputs/dev_dawe_control_seed${seed}_300k"
    else
      config="configs/dev_dawe_fixed10_v1_300k.yaml"
      output="outputs/dev_dawe_v1_seed${seed}_300k"
    fi
    check_all_locks "$seed" "$variant" "$source" "$source_sha"
    [[ ! -e "$output" ]] || { echo "ERROR: refusing existing output $output" >&2; exit 2; }
    echo "===== DAWE SCREEN START seed=${seed} variant=${variant} output=${output} ====="
    python -u algorithm/train_modular_mappo.py \
      --env-config configs/persistent_wave_v2_environment.yaml \
      --algorithm-config "$config" \
      --branch-from "$source" \
      --output-dir "$output" \
      --device cuda \
      --seed "$seed" \
      --num-envs 24 \
      --total-sampled-steps 1805280
    check_all_locks "$seed" "$variant" "$source" "$source_sha"
    echo "===== DAWE SCREEN DONE seed=${seed} variant=${variant} output=${output} ====="
  done
done
