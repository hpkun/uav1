# On-policy training metrics storage

Future MAPPO, RMAPPO, EA-MAPPO and STEA-MAPPO runs write exactly one
`training_metrics.jsonl` record after each completed rollout/update cycle.
Collection performs no training-metrics file writes. All four algorithms use
the MAPPO base accumulator and writer; recurrent rollout/reset logic is unchanged.

Each record has `record_type="rollout_summary"`, cumulative `sampled_steps`,
`rollout_update`, `rollout_vector_steps`, `rollout_sampled_steps`, and
`completed_episodes_in_rollout`.

Step aggregates retain `mean_step_reward`, `mean_r1_reward` through
`mean_r4_reward`, red/blue fire-window pairs, step fire attempts and weapon hits.
Rewards are summed in float64 and divided by the number of scalar agent samples;
pair/attempt/hit means use environment-transition samples. Each field has an
explicit sum/count, so uneven batch sizes remain correctly weighted.

Completed episodes from every vector step in the rollout are pooled. Return,
win/loss/draw/timeout, losses, kills, attempts, hits, boundaries, ground losses,
episode lengths, R1–R4 totals and other `aggregate_combat_records` diagnostics
are averaged per completed episode. Each episode has equal weight. If none
completes, the original explicit episode fields are null and episode count is
zero; completion-only aggregate fields remain absent, as before. No observations,
actions, hidden states, trajectories, info arrays or per-agent episode-return
arrays are serialized.

The summary also contains only five current optimizer scalars: actor_loss,
value_loss, entropy, approx_kl, clip_fraction. Complete optimizer diagnostics,
including dimensional log_std/sigma, gradients, epochs, learning rates and KL
protocol fields, remain in `optimization_metrics.jsonl`, once per PPO update.
No `ppo_update` duplicate is written to new training logs.

With 3,000,000 transitions, 16 environments and 256 vector steps per rollout:

- Legacy: 187,500 step records, plus about 733 duplicate optimizer records.
- Compact: ceil(3,000,000 / 4096) = 733 rollout summaries.
- The final rollout is partial: 1,728 transitions / 108 vector steps.

The reduction is in write frequency and optimizer duplication. The exact byte
size depends on environment diagnostic fields and completed episodes.

No algorithm-config field is added or edited, so configuration SHA contracts
are unchanged. Sampling, RNG, models, PPO update frequency, console scheduling
and its recent-100-episode window, checkpoint/evaluation scheduling, optimization
logs, evaluation CSV and run-summary semantics are unchanged.

Resume rollback continues to use cumulative `sampled_steps` for training logs,
optimization logs and evaluation CSV. It requires no new checkpoint fields or
checkpoint format. Resuming a legacy directory can produce a mixed-format log.

`algorithm.common.training_log` distinguishes records with no record_type
(legacy step), `ppo_update` (legacy optimizer duplicate), and `rollout_summary`.
The STEA representation diagnostic reads PPO updates solely from optimization
logs. Existing legacy duplicates are optionally checked against the matching
sampled_steps record, allowing an old prefix and new compact suffix. A compact
log with no duplicates is valid. The initial-policy audit likewise excludes
optimizer duplicates when counting completed training episodes.

Historical output files are never migrated, compressed or rewritten by this
change. Validation uses isolated temporary/new directories only.
