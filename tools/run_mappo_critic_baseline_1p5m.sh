#!/usr/bin/env bash
set -euo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == "uav" ]] || { echo "ERROR: activate conda uav first" >&2; exit 2; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
python -u tools/preflight_mappo_critic_baseline_1p5m.py --print-only
lock_snapshot(){ python - <<'PY'
from pathlib import Path
from algorithm.common.protocol import config_sha256,runtime_source_manifest
import yaml
r=Path.cwd();load=lambda p:yaml.safe_load(p.read_text(encoding="utf-8"))
print(config_sha256(load(r/"configs/persistent_wave_v2_environment.yaml")),config_sha256(load(r/"configs/mappo_mlp_baseline_1p5m.yaml")),config_sha256(load(r/"configs/mappo_attention_baseline_1p5m.yaml")),runtime_source_manifest(r)["runtime_source_manifest_sha256"])
PY
}
read -r LOCK_ENV LOCK_MLP LOCK_ATTN LOCK_SOURCE < <(lock_snapshot)
check_locks(){ read -r env mlp attn source < <(lock_snapshot); [[ "$env" == "$LOCK_ENV" && "$mlp" == "$LOCK_MLP" && "$attn" == "$LOCK_ATTN" && "$source" == "$LOCK_SOURCE" ]] || { echo "ERROR: protocol/source lock changed" >&2; exit 2; }; }
for arm in mlp attention; do
 check_locks
 config="configs/mappo_${arm}_baseline_1p5m.yaml";output="outputs/mappo_${arm}_seed5303_1p5m"
 [[ ! -e "$output" ]] || { echo "ERROR: refusing existing output $output" >&2; exit 2; }
 python -u algorithm/train_mappo.py --env-config configs/persistent_wave_v2_environment.yaml --algorithm-config "$config" --output-dir "$output" --device cuda --seed 5303 --num-envs 24 --total-sampled-steps 1500000
 check_locks
done
