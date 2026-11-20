# Airbase interception research

This project trains five defender UAVs to intercept five scripted attacker UAVs before an attacker enters a protected area. Attackers spawn at the top of a 200 m square. Defenders spawn at the bottom; the protected circle is centered at `(0, -80)` m. The available attack modes are `direct_rush`, `split_attack`, and a balanced `mixed` training schedule.

The environment uses two-dimensional point-mass dynamics, continuous acceleration actions, local defender observations, and a centralized state assembled from those observations. Capture, penetration, boundary and timeout events are recorded explicitly. Reward coefficients in `configs/interception.yaml` are initial project choices; the cited paper does not publish their numerical values.

New actor observations include all four teammates and all five active attackers within the same 80 m sensing radius, with fixed zero-filled slots for out-of-range agents. Every observed UAV uses absolute world position and velocity; the base is not an actor input. The paper uses separate 60 m communication and 80 m sensing ranges and only the nearest three teammates and three attackers. The unified sensing radius and larger observation layout are project choices.

New training runs use capture (`+20` for the capturing defender), boundary (`-2` per offending defender), penetration (`-100` for each defender when any attacker breaches), and approach (`+0.05` per metre of defender motion toward the nearest attacker visible before the step). Approach is zero when no attacker is within the 80 m sensing radius; attacker motion alone does not earn approach reward. The team reward is the mean of five defender rewards. Historical runs with a nonzero `danger_weight` retain their original danger and approach calculations when loaded from their saved configuration.

## Commands

Run commands in PowerShell after activating the existing `uav` Conda environment. Every output directory must be new. `preflight` does not train.

```powershell
conda activate uav
python -m experiment.cli preflight `
  --env-config configs/interception.yaml `
  --algorithm-config configs/dual_critic_interception.yaml `
  --algorithm dual_critic --attack-mode mixed `
  --total-sampled-steps 10000 --num-envs 4 `
  --output-dir runs/interception_plan

python -m experiment.cli train `
  --env-config configs/interception.yaml `
  --algorithm-config configs/dual_critic_interception.yaml `
  --algorithm dual_critic --attack-mode mixed `
  --total-sampled-steps 10000 --num-envs 4 `
  --eval-interval 5000 --eval-episodes 20 `
  --log-every-steps 5000 `
  --output-dir runs/interception_example

python -m experiment.cli evaluate `
  --checkpoint runs/interception_example --label Best `
  --attack-mode both --episodes-per-mode 200 `
  --output-dir outputs/interception_example_eval

python -m tools.replay `
  --checkpoint runs/interception_example --label Best `
  --attack-mode split_attack --seed 900001 `
  --html --output replays/interception_example_split

python -m experiment.cli report `
  --run runs/interception_example `
  --evaluation outputs/interception_example_eval `
  --plot outputs/interception_example_curve.svg
```

MAPPO and MADDPG are also available using `configs/mappo_interception.yaml` and `configs/maddpg_interception.yaml`. `configs/interception_smoke.yaml` limits episodes to 40 steps for interface checks only. Its checkpoint contract differs from the full scenario.

The exact MAPPO, MADDPG and dual-critic observation, action, critic and reward interfaces are documented in `docs/ALGORITHM_IO.md`.

## Outputs

Training writes `settings.json`, `episodes.jsonl`, `updates.jsonl`, `evaluations.jsonl`, `progress.jsonl`, `summary.json`, and a SHA-256 checkpoint catalog under `checkpoints/`. Each training episode row includes summed reward components; training no longer writes per-step reward traces. Independent evaluation writes `evaluation.json` and `episodes.jsonl`. Replay writes a complete compressed trace, an offline HTML player, and a summary. A time limit is a separate outcome; all-capture and penetration rates do not have to sum to one.

Existing run and replay artifacts are not modified by the new commands.

The offline `replay.html` shows defenders in red and attackers in blue. Each defender has optional sensing and capture rings, using the saved environment's `sensing_radius` and `capture_radius`. Select a defender to highlight the friendly and attacker UAVs it can currently sense. Recreate a replay in a new output directory after updating the script; existing HTML files are static snapshots.

Training prints one compact status block every `--log-every-steps` sampled steps (default 10,000), at the next rollout boundary after 30 seconds without a status block, and at completion. It includes recent direct-rush and split-attack outcomes, capture efficiency, return, learner losses, throughput and ETA. The same snapshots are saved to `progress.jsonl`.

`summary.json` records `best_selection`, and the `Best` checkpoint catalog entry records `selection`. `periodic_evaluation` means validation selected that checkpoint; `final_unvalidated` means validation was disabled and `Best` is a copy of the final model. Older runs without these fields can be identified by a non-null `best_score` and non-empty `evaluations.jsonl`. Checkpoints support inference and learner-state loading, but the training command does not restore replay buffers, worker states or unfinished episodes, so it cannot resume an interrupted run exactly.
