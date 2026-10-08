# Formal 5v5 control algorithms

RMAPPO and EA-MAPPO complete the representation/memory comparison alongside
the existing MAPPO and STEA-MAPPO. They use v2.5, 65 observation features,
three continuous actions and five Red agents. Existing algorithms and the
environment are unchanged.

| Algorithm | Actor representation | Temporal memory | Critic |
|---|---|---|---|
| MAPPO | Existing flat MLP | None | CentralizedMLPCritic |
| RMAPPO | Linear(65,128), ReLU | GRU128, one layer | CentralizedMLPCritic |
| EA-MAPPO | STEA entity attention and spatial fusion | None | CentralizedMLPCritic |
| STEA-MAPPO | Existing entity attention and spatial fusion | Existing GRU128 | CentralizedMLPCritic |

## Exact actors

RMAPPO: `Linear(65,128) → ReLU → GRU(128,128,1,batch_first=True)
→ Linear(128,128) → ReLU → Linear(128,3)`.
Its actor has **124,422 parameters** including the three log standard
deviations. There are no entity encoders, attention modules, manually added
agent IDs or extra flat encoder layers. Hidden shape is `[E,5,128]` with
independent environment/agent entries. Sequence execution uses `[B,L,5,65]`.

EA-MAPPO: independent `Linear(7,64)+ReLU` self and ally encoders and
`Linear(6,64)+ReLU` enemy encoder; independent two-head ally/enemy attention,
with self embedding as query; concatenate `[self64,ally64,enemy64]`, then
`Linear(192,128)+ReLU → Linear(128,128)+ReLU → Linear(128,3)`.
Its actor has **68,038 parameters**. It has no GRU/LSTM, recurrent interface,
hidden buffer, episode-start recurrent masks, sequence packing or TBPTT.

EA directly reuses STEA's pure entity encoding, masked attention and spatial
fusion methods. With copied frontend weights, fused features and diagnostic
tensors are exactly equal for the same observation/alive masks. Invalid
entity weights and empty contexts are exactly zero. Slot permutations preserve
the fused features and permute attention weights correspondingly.

The seven self features include pitch, not a self-alive flag. EA's stateless
distribution uses entity alive flags from observation slots; formal dead
observations are all zero. The inherited MAPPO trainer masks dead actions and
log probabilities using the explicit alive mask. No observation semantics
were added or changed.

## Shared critic and Gaussian/PPO protocol

All four formal algorithms use the exact existing `CentralizedMLPCritic`
class: `390 → 256 → 256 → 1`, **166,145 parameters**. It concatenates the
masked 325D joint observation with the focal 65D observation. The two new
trainers preserve the old baseline construction order before replacing only
the temporary actor, so same-seed critic tensors, keys, shapes and counts
match MAPPO and STEA exactly. The critic has no GRU or attention.

Both new configurations default to **3,000,000 sampled environment steps**,
16 training environments, rollout 256, minibatch 512 and at most 10 PPO epochs.
Adam actor/critic learning rates are both 3e-4; gamma=.99, lambda=.95,
clip=.2, value coefficient=.5, entropy coefficient=.001, target KL=.015,
max gradient norm=.5, advantage normalization and clipped value loss enabled.

The state-independent Gaussian has `log_std_parameter[3]`, initialized at
-.5 and clamped to [-5,.5]. The mean head uses orthogonal gain .01 and zero
bias. Training samples `raw ~ Normal(mu,sigma)` and applies `tanh(raw)`;
evaluation uses `tanh(mu)`. Formal rollouts retain raw actions. Both new
actors reuse the existing exact, saturation-safe tanh Jacobian calculation.

RMAPPO uses the existing verified recurrent PPO design, independently
implemented in its package: contiguous length32 chunks per environment,
128 chunks for a 256×16 rollout, 16 chunks per 512-transition minibatch.
Only chunks are shuffled. Each chunk starts from its actual recorded hidden;
partial chunks are padded and masked. Death resets the corresponding hidden,
episode completion resets the whole environment hidden, and episode-start
masks cut history and cross-episode gradients. No future observation leaks
into earlier actions.

Before every RMAPPO update, all live rollout log probabilities are replayed
using saved raw actions, hidden and reset masks. A non-finite ratio or
`max_abs_error >= 1e-4` raises RuntimeError. Corrupted old log probabilities
are rejected before optimizer updates.

EA inherits ordinary MAPPO transition-shuffled rollout/update. Its additional
attention statistics run once after each update, under no-grad, over the
rollout observations and explicit live masks. These report **post-update
weights**, unlike STEA's within-update diagnostic averages. They consume no
random numbers and do not change training forward, weights or gradients;
an exact update/RNG parity test verifies this. Entropy/top1 are raw existing
diagnostic definitions, not K-normalized concentration measures.

## Checkpoints and commands

Each package has its own factory, protocol, runner and evaluation module.
Checkpoint algorithm identities are exactly `RMAPPO` and `EA-MAPPO`.
Each uses implementation version1 with an algorithm-specific version field
and independent architecture metadata. Strict runner/evaluation validation
checks environment version, dimensions, config SHA, architecture, Gaussian
protocol and parameter counts before loading. Cross-algorithm loads in every
direction are rejected. Resume uses immutable config snapshots and the same
existing rollback/seed-provenance utilities.

Training commands for future manual use; these were **not run at 3M**:

```bash
conda activate uav
python algorithm/train_rmappo.py --device cuda --seed 1 \
  --output-dir outputs/rmappo_v25_5v5_seed1_3m
python algorithm/train_ea_mappo.py --device cuda --seed 1 \
  --output-dir outputs/ea_mappo_v25_5v5_seed1_3m
```

Defaults select `configs/combat_environment_v25.yaml` and the corresponding
`configs/rmappo_5v5.yaml` or `configs/ea_mappo_5v5.yaml`. WSL Ubuntu with
`conda activate uav` and CUDA is required by repository runtime policy.
Training/evaluation runtime fails explicitly when CUDA is unavailable.

For a short check, use `--smoke --total-sampled-steps 512` and a fresh output
directory. Smoke preserves formal actor/critic widths and every PPO
hyperparameter; it only uses rollout32, two evaluation episodes and reporting
intervals at the short target. With 16 environments this is one 512-transition
rollout/update. Config snapshots continue to record the default 3M protocol,
while runtime metadata explicitly records the smoke target and mode.

```bash
python algorithm/evaluate_rmappo.py \
  --checkpoint outputs/rmappo_v25_5v5_seed1_3m/checkpoint_3000000.pt \
  --seed-base 70000000 --episodes 50 --device cuda --output outputs/rmappo_holdout.json
python algorithm/evaluate_ea_mappo.py \
  --checkpoint outputs/ea_mappo_v25_5v5_seed1_3m/checkpoint_3000000.pt \
  --seed-base 70000000 --episodes 50 --device cuda --output outputs/ea_mappo_holdout.json
```

For checkpoints trained from another configuration, pass the saved run's
`--env-config` and `--algorithm-config`. Evaluation seeds must be contiguous
and outside the conservative training seed range.

The acceptance artifacts live in `outputs/new_baseline_validation/`,
including smoke logs, saved checkpoints, reloaded two-episode evaluations,
`smoke_report.json`, full pytest output and source/config SHA checks.
Smoke checks software execution and finite metrics; its two games and
untrained policies do not establish comparative algorithm performance.
