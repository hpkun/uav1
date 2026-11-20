# Airbase interception research

This project trains five defender UAVs to intercept five scripted attacker UAVs before an attacker enters a protected area. Attackers spawn at the top of a 200 m square. Defenders spawn at the bottom; the protected circle is centered at `(0, -80)` m. The available attack modes are `direct_rush`, `split_attack`, and a balanced `mixed` training schedule.

The environment uses two-dimensional point-mass dynamics, continuous acceleration actions, local defender observations, and a centralized state assembled from those observations. Capture, penetration, boundary and timeout events are recorded explicitly. Reward coefficients in `configs/interception.yaml` are initial project choices; the cited paper does not publish their numerical values.

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

Training writes `settings.json`, `episodes.jsonl`, `updates.jsonl`, `evaluations.jsonl`, `progress.jsonl`, `reward_steps.jsonl.gz`, `summary.json`, and a SHA-256 checkpoint catalog under `checkpoints/`. Independent evaluation writes `evaluation.json` and `episodes.jsonl`. Replay writes a complete compressed trace, an offline HTML player, and a summary. A time limit is a separate outcome; all-capture and penetration rates do not have to sum to one.

Existing run and replay artifacts are not modified by the new commands.

Training prints one compact status block every `--log-every-steps` sampled steps (default 10,000), at the next rollout boundary after 30 seconds without a status block, and at completion. It includes recent direct-rush and split-attack outcomes, capture efficiency, return, learner losses, throughput and ETA. The same snapshots are saved to `progress.jsonl`.
