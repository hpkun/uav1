# Formal v2.4 8v8 comparison

The four formal configurations are:

| Algorithm | Config | Actor | Actor parameters |
|---|---|---|---:|
| MAPPO | configs/mappo_8v8_formal.yaml | 104→256→256→3 | 93446 |
| RMAPPO | configs/rmappo_8v8.yaml | 104→128→GRU128→128→3 | 129414 |
| EA-MAPPO | configs/ea_mappo_8v8.yaml | Entity encoders/attention→spatial128→128→3 | 68038 |
| STEA-MAPPO v1 | configs/stea_mappo_8v8_formal.yaml | Entity encoders/attention→spatial128→GRU128→128→3 | 167110 |

All use the existing, unmodified `configs/combat_environment_v24.yaml`:
observation/action/agents = **104/3/8**. Reward, weapon, Blue policy,
termination and scenario are unchanged. Historical `mappo_8v8.yaml` and
`stea_mappo_8v8.yaml` remain unchanged for historical compatibility; they are
not the formal protocol introduced here.

The new configurations are exact copies of their current formal 5v5
counterparts except for observation_dim=104 and num_agents=8. All default to
3,000,000 sampled environment steps and share Adam, actor/critic lr3e-4,
gamma.99, lambda.95, PPO clip.2, value coefficient.5, entropy coefficient.001,
target KL.015, gradient clip.5, rollout256, epochs≤10, minibatch512,
16 environments, advantage normalization and clipped value loss.
Evaluation is deterministic, 20 episodes every100,000 steps; checkpoint
interval is500,000 steps.

The state-independent Gaussian is initialized with log_std=-.5 and bounds
[-5,.5], and uses an orthogonal mean head with gain.01 and zero bias.
Raw latent actions and the exact tanh Jacobian protocol are unchanged.
None of the new configs uses attention critic, legacy state-dependent
standard deviation, entropy.01 or log_std_max2.

## Dimensions and compatibility

The existing `CentralizedMLPCritic` already derives input width from
`(num_agents+1)*observation_dim`. All four 8v8 algorithms therefore share
**936→256→256→1**, **305,921 parameters**. No critic code was changed.
Same-seed class, keys, shapes, counts and initial tensors are exactly equal;
unit tests cover seeds0/3 and the CUDA smoke covers seed31.

EA and STEA reuse the unchanged 13*N entity decomposition:
self7 + 7 allies×7 + 8 enemies×6 =104. Embeddings remain64, attention heads2,
fusion192→128. Dead weights and empty contexts remain exactly zero, with
slot permutation equivariance. EA has no temporal state; STEA remains v1,
with its existing GRU128. There is no residual or gated STEA-v2 change.

RMAPPO accepts configured observation dimensions65 or104, with all temporal
core sizes fixed at128/128/one layer. Its sequence shape validation/error
uses the configured observation dimension. RMAPPO and EA runners and strict
checkpoint validators permit environment versions2.4 and2.5, while requiring
exact environment/config dimension agreement. A5v5 config cannot initialize
the8v8 environment, nor can an8v8 config initialize5v5.

Checkpoint identities and implementation versions are unchanged. Existing
5v5 network_architecture metadata is preserved, so old5v5 checkpoints remain
compatible. Dimension and environment/config SHA validation still reject
cross-environment loading. MAPPO and STEA source code was not modified.

A pre-change fixture, `tests/fixtures/formal_5v5_before_8v8.pt`, records all
four5v5 actors and critics, fixed inputs, nonzero hidden/reset cases, means,
scales, forward results and masked deterministic actions. The corresponding
regression tests require tensor equality, not approximate closeness, after
the compatibility change. It is a new reference captured before editing;
existing environment reference fixtures were not updated.

## Commands for future manual experiments

These3M commands are examples and were not executed during acceptance.
Use WSL Ubuntu and the `uav` Conda environment with CUDA.

```bash
conda activate uav
python algorithm/train_mappo.py --device cuda --seed 1 \
  --env-config configs/combat_environment_v24.yaml \
  --algorithm-config configs/mappo_8v8_formal.yaml \
  --output-dir outputs/mappo_v24_8v8_seed1_3m_formal

python algorithm/train_rmappo.py --device cuda --seed 1 \
  --env-config configs/combat_environment_v24.yaml \
  --algorithm-config configs/rmappo_8v8.yaml \
  --output-dir outputs/rmappo_v24_8v8_seed1_3m

python algorithm/train_ea_mappo.py --device cuda --seed 1 \
  --env-config configs/combat_environment_v24.yaml \
  --algorithm-config configs/ea_mappo_8v8.yaml \
  --output-dir outputs/ea_mappo_v24_8v8_seed1_3m

python algorithm/train_stea_mappo.py --device cuda --seed 1 \
  --env-config configs/combat_environment_v24.yaml \
  --algorithm-config configs/stea_mappo_8v8_formal.yaml \
  --output-dir outputs/stea_mappo_v24_8v8_seed1_3m_formal
```

The RMAPPO/EA default entry configs are5v5 and MAPPO/STEA default entries are
legacy4v4. Always pass both8v8 config arguments explicitly. For evaluation,
use the matching `algorithm/evaluate_*.py` with the run's saved env/config
snapshots, explicit CUDA, checkpoint, seed base, episode count and output.

## Bounded CUDA acceptance

```bash
python tools/smoke_formal_8v8.py --output outputs/fresh_formal_8v8_smokes
```

This tool refuses populated run directories and caps each algorithm at
**512 sampled steps**, with16 real multiprocess environments. It uses the
formal-width runners rather than legacy `--smoke`, which would reduce
MAPPO/STEA dimensions. Configs remain3M with rollout256; the512-step cap
collects a32-step final partial rollout. Actual sampled-step counters and
runtime provenance record512. No3M job is started.

Each run executes GAE, actor/critic PPO updates and target-KL checks, saves
and strictly reloads its checkpoint, then evaluates exactly two deterministic
episodes. RMAPPO/STEA perform their runtime sequence ratio audits. Initial
critics are compared exactly before updates. Finite gradients/weights/values,
parameter counts, Gaussian metrics, dimensions and checkpoint metadata are
checked. No best-checkpoint selection rule or official evaluation default
was changed. Smoke results are execution checks, not trained performance
comparisons.

Acceptance outputs are in `outputs/formal_8v8_validation/`, including
pre-change protected hashes, targeted/full pytest logs and the four smoke
directories under `smokes/` with saved snapshots/checkpoints, optimization
metrics and reloaded two-episode evaluations.
