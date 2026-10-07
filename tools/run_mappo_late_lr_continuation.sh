#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == "uav" ]] || { echo "ERROR: activate conda uav first" >&2; exit 2; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
python -u tools/preflight_mappo_late_lr_continuation.py --print-only

lock_snapshot(){ python - <<'PY'
from pathlib import Path
from algorithm.common.protocol import config_sha256,runtime_source_manifest
import hashlib,yaml
r=Path.cwd();load=lambda p:yaml.safe_load(p.read_text(encoding="utf-8"))
source=r/"outputs/mappo_attention_seed5303_1p5m/checkpoint_1001472.pt"
print(config_sha256(load(r/"configs/persistent_wave_v2_environment.yaml")),config_sha256(load(r/"configs/mappo_attn_lr_control_3e4_cont_1p5m.yaml")),config_sha256(load(r/"configs/mappo_attn_lr_treatment_1e4_cont_1p5m.yaml")),hashlib.sha256(source.read_bytes()).hexdigest(),runtime_source_manifest(r)["runtime_source_manifest_sha256"])
PY
}
read -r LOCK_ENV LOCK_CONTROL LOCK_TREATMENT LOCK_CHECKPOINT LOCK_SOURCE < <(lock_snapshot)
check_locks(){ read -r env control treatment checkpoint source < <(lock_snapshot); [[ "$env" == "$LOCK_ENV" && "$control" == "$LOCK_CONTROL" && "$treatment" == "$LOCK_TREATMENT" && "$checkpoint" == "$LOCK_CHECKPOINT" && "$source" == "$LOCK_SOURCE" ]] || { echo "ERROR: protocol/source lock changed" >&2; exit 2; }; }

declare -a CONFIGS=("configs/mappo_attn_lr_control_3e4_cont_1p5m.yaml" "configs/mappo_attn_lr_treatment_1e4_cont_1p5m.yaml")
declare -a OUTPUTS=("outputs/mappo_attn_lr3e4_cont_seed5303_1p5m" "outputs/mappo_attn_lr1e4_cont_seed5303_1p5m")
for index in 0 1; do
  check_locks
  [[ ! -e "${OUTPUTS[$index]}" ]] || { echo "ERROR: refusing existing output ${OUTPUTS[$index]}" >&2; exit 2; }
  python -u tools/train_mappo_late_lr_branch.py --algorithm-config "${CONFIGS[$index]}" --output-dir "${OUTPUTS[$index]}"
  check_locks
done
