# Wave-Survival PBRS Development

WS-PBRS is a development-only, training-reward intervention for persistent-wave MAPPO. It does not modify the environment reward, observations, actor, critic, PPO, GAE, dynamics, weapons, or evaluation metrics.

For each transition, cumulative progress is `P = cumulative_blue_removed / total_blue`, survival is `S = red_alive / initial_red_count`, and agent potential is `Phi_i = alive_i * P * S`. Training uses `R_train_i = R_env_i + gamma * Phi_i(next) - Phi_i(pre)` with coefficient `1.0` and the training gamma `0.999`. All true terminal and truncated transitions use absorbing zero next potential. Evaluation always reports raw environment reward.

The frozen development matrix is MAPPO Baseline and MAPPO + WS-PBRS, paired on training seeds `5101`, `5102`, and `5103`, trained from scratch to `900000` sampled steps. Both methods use the same 50 deterministic development scenarios `39000000..39000049`; `latest@900k` is primary and best checkpoints are diagnostic only. The future-final `33M` range remains untouched.

Run preflight before starting the six long runs:

```bash
conda activate uav
python -u tools/preflight_ws_pbrs_development.py
bash tools/run_ws_pbrs_development.sh
```
