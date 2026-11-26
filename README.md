# Airbase interception research

This project trains five defender UAVs to intercept five scripted attacker UAVs before they enter a protected area. Attackers spawn at the top of a 200 m square. Defenders spawn at the bottom; the protected circle is centered at `(0, -80)` m. Train `direct_rush` and `split_attack` in separate runs. Each attacker is resolved by capture or penetration; a new episode ends only after all five are resolved. A single penetration still makes the episode a defensive failure.

The environment uses two-dimensional point-mass dynamics, continuous acceleration actions, local defender observations, and a centralized training state. Capture, penetration, boundary and timeout events are recorded explicitly. Reward coefficients in the environment configurations are initial project choices; the cited paper does not publish their numerical values.

`configs/interception_assignment.yaml` enables the `Interception/2` distance-matching interface. Each defender observes its own world position and velocity, the relative positions and relative velocities of visible teammates and attackers, and the relative position and velocity of its assigned visible target. The base is absent from the actor input. The allocator finds a maximum-cardinality, minimum-total-distance one-to-one matching on currently visible edges. It replans at reset, every 20 steps, when the defense system first detects an attacker, or after a capture or penetration; otherwise it retains assignments. `configs/interception_assignment_nearest.yaml` keeps independent nearest selection as a weaker comparison with the same triggers. The centralized critic receives physical state for all ten UAVs and the 5×5 assignment matrix. `configs/interception.yaml` retains the `Interception/1` absolute observation interface for historical runs.

New training runs use one-to-one capture: the matched defender and attacker both leave the episode immediately. A defender can capture at most one attacker. Every defender receives the same task reward: `+20` for each attacker captured and `-100` once, at the first penetration in the episode. Later penetrations add no further penalty. Individual shaping is `-2` per defender boundary violation plus `+0.5` per metre of actual distance reduction in `Interception/2`; historical `Interception/1` configurations and runs retain `+0.05`. In `Interception/2`, distance is measured to the assigned target held before the step; `Interception/1` measures it to the nearest attacker visible before the step. Both UAVs' positions are compared before and after movement. Positions advance with the previous velocity and acceleration updates the next velocity, so the current action influences this distance reward from the following step. The MAPPO team reward is the shared task reward plus the mean of the five shaping rewards; each MADDPG critic uses the shared task reward plus its defender's shaping reward. Ignoring shaping and discounting, five captures yield `+100`, while four captures and one penetration yield `-20`; zero captures and five penetrations yield `-100`. Attackers unresolved at the 5000-step limit are counted as penetrations. Historical checkpoints keep their saved episode protocols, including the earlier `resolve_all_one_penalty` rule where defenders remained after capture, `resolve_all` with per-attacker penalties, and older first-breach termination.

## Commands

Run commands in PowerShell after activating the existing `uav` Conda environment. Every output directory must be new. `preflight` does not train.

```powershell
conda activate uav
python -m experiment.cli preflight `
  --env-config configs/interception_assignment.yaml `
  --algorithm-config configs/mappo_interception.yaml `
  --algorithm mappo --attack-mode direct_rush `
  --total-sampled-steps 10000 --num-envs 4 `
  --output-dir runs/interception_plan

python -m experiment.cli train `
  --env-config configs/interception_assignment.yaml `
  --algorithm-config configs/mappo_interception.yaml `
  --algorithm mappo --attack-mode direct_rush `
  --total-sampled-steps 10000 --num-envs 4 `
  --eval-interval 500 --eval-episodes 20 `
  --log-every-steps 5000 `
  --output-dir runs/interception_example

python -m experiment.cli evaluate `
  --checkpoint runs/interception_example --label Best `
  --attack-mode direct_rush --episodes-per-mode 200 `
  --output-dir outputs/interception_example_eval

python -m tools.replay `
  --checkpoint runs/interception_example --label Best `
  --attack-mode direct_rush --seed 900001 `
  --html --output replays/interception_example_direct

python -m experiment.cli report `
  --run runs/interception_example `
  --evaluation outputs/interception_example_eval `
  --plot outputs/interception_example_curve.svg
```

MAPPO, MADDPG and isolated MATD3 use `configs/mappo_interception.yaml`, `configs/maddpg_interception.yaml` and `configs/matd3_interception.yaml` (for `Interception/13` radial speed-heading, use `configs/maddpg_jia_heading_interval10.yaml` or `configs/matd3_jia_heading_interval10.yaml`). MATD3 keeps its own algorithm name (`matd3`) and checkpoint schema (`interception_matd3_1`) isolated from MADDPG (`interception_maddpg_2`). The dual-critic baseline currently supports `Interception/1` only. `configs/interception_smoke.yaml` limits episodes to 40 steps for interface checks only.

Run each algorithm once with `--attack-mode direct_rush` and again with `--attack-mode split_attack`, using a new output directory each time. Periodic and independent evaluation, as well as replay, use the checkpoint's training mode. A mismatched or unsupported mode is rejected before any output is written.

The exact MAPPO, MADDPG and dual-critic observation, action, critic and reward interfaces are documented in `docs/ALGORITHM_IO.md`.

## Outputs

Training writes `settings.json`, `episodes.jsonl`, `updates.jsonl.gz`, `updates_summary.jsonl`, `evaluations.jsonl`, `progress.jsonl`, `summary.json`, and a SHA-256 checkpoint catalog under `checkpoints/`.

- `updates.jsonl.gz`: complete gzip-compressed raw per-update training metrics (streamed directly during training; new runs no longer write uncompressed `updates.jsonl`, while existing historical runs keep their original files).
- `updates_summary.jsonl`: fixed-window aggregated update statistics written every `--update-log-interval` policy updates (default `100`, including the final partial window). Each line records `step_start`, `step_end`, `update_start`, `update_end`, `count`, per-metric `mean`/`std` (population std)/`min`/`max`/`count` under `metrics`, and window-end cumulative values under `counters`.
- `progress.jsonl`: periodic training progress and recent-window status snapshots.

By default, each periodic evaluation also saves `checkpoints/step_<step>.pt` and updates `checkpoints/index.json` atomically under label `step_<step>` (disable with `--no-save-eval-checkpoints`). Evaluation, replay, and diagnostics accept either `--checkpoint <run_dir> --label <Best|Last|step_<step>>` or a direct cataloged `.pt` file path `--checkpoint <run_dir>/checkpoints/step_<step>.pt`. Each training episode row includes summed reward components; training no longer writes per-step reward traces. Independent evaluation writes `evaluation.json` and `episodes.jsonl`. Replay writes a complete compressed trace, an offline HTML player, and a summary. In the new protocol, unresolved attackers at the time limit count as penetrations, so completed episodes end as all-captured or penetrated. Historical first-breach runs can still have a separate time-limit outcome.

Reading compressed raw update logs and plotting training curves from `updates_summary.jsonl`:

```python
import gzip
import json
import matplotlib.pyplot as plt

# 1. Stream-read complete raw updates from updates.jsonl.gz
with gzip.open("runs/interception_example/updates.jsonl.gz", "rt", encoding="utf-8") as f:
    raw_updates = [json.loads(line) for line in f if line.strip()]

# 2. Load aggregated summaries and plot critic_loss mean ± std
with open("runs/interception_example/updates_summary.jsonl", "rt", encoding="utf-8") as f:
    summaries = [json.loads(line) for line in f if line.strip()]

steps = [row["step_end"] for row in summaries]
mean = [row["metrics"]["critic_loss"]["mean"] for row in summaries]
std = [row["metrics"]["critic_loss"]["std"] for row in summaries]

plt.plot(steps, mean, label="critic_loss mean")
plt.fill_between(steps, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], alpha=0.2)
plt.xlabel("Sampled steps")
plt.ylabel("Critic loss")
plt.legend()
plt.show()
```

Or use `experiment.update_log.read_update_log(run_dir)` to transparently read either new `updates.jsonl.gz` or historical `updates.jsonl`.

Existing run and replay artifacts are not modified by the new commands.

The offline `replay.html` shows defenders in red and attackers in blue. Each defender has optional sensing and capture rings, using the saved environment's `sensing_radius` and `capture_radius`. Select a defender to highlight the friendly and attacker UAVs it can currently sense. Recreate a replay in a new output directory after updating the script; existing HTML files are static snapshots.

Training prints one compact status block every `--log-every-steps` sampled steps (default 10,000), at the next rollout boundary after 30 seconds without a status block, and at completion. It includes recent outcomes for the run's single attack mode, capture efficiency, return, learner losses, throughput and ETA. The same snapshots are saved to `progress.jsonl`. By default, periodic evaluation runs every 5% of the sampled-step budget, at the sampled step reaching each threshold, on that same attack mode only; the final evaluation follows the last learner update. Each `[eval]` block prints mean undiscounted reward per episode for capture, penetration, approach, boundary and total team return. The same breakdown is saved in `evaluations.jsonl`; independent evaluation also saves each episode's reward components. Use `--eval-interval 0` only when validation is intentionally disabled.

`summary.json` records `best_selection` and `best_selection_mode`; the `Best` checkpoint catalog entry records `selection` and `selection_mode`. `periodic_evaluation` means validation on the training mode selected that checkpoint; `final_unvalidated` means validation was disabled and `Best` is a copy of the final model. Historical result files remain on disk, but checkpoints with unsupported training modes cannot be loaded. Checkpoints support inference and learner-state loading, but the training command does not restore replay buffers, worker states or unfinished episodes, so it cannot resume an interrupted run exactly.
