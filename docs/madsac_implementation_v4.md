# MADSAC implementation-v5 (delayed-policy clock correction)

V5 retains the v4 algorithm structure and corrects the delayed-policy clock.
The production implementation aligns the MADSAC critic and explicit training cycle with Li et al.
(2023), *Multi-UAV Cooperative Air Combat Decision-Making Based on Multi-Agent
Double-Soft Actor-Critic*, [DOI](https://doi.org/10.3390/aerospace10070574).
No combat environment, reward, weapon, observation, Blue policy, MAPPO, or MAAC
implementation changes are part of this revision. No formal long training was run.

## Critic and policy

The shared local actor remains a two-hidden-layer 256-unit tanh diagonal Gaussian
policy with reparameterization and the existing stable tanh Jacobian correction.
Alpha stays fixed at 0.1. The two centralized critics remain independent, with
their own Adam optimizers and target networks. Their existing structure is:

```text
[o_i, a_i] -> two-hidden-layer embedding MLP
q = Wq e; k = Wk e; v = LeakyReLU(Wv e, negative_slope=0.01)
attention = softmax(q k^T / sqrt(head_dim)), including self among live keys
concatenate weighted-value heads -> context-only Q MLP
```

Production computes `math.sqrt(self.head_dim)`; formal width 256 / 2 heads gives
head_dim 128. Eq.(17) does not explicitly print the divisor, but Figure 6(c)
explicitly labels **Scaled Dot-Product**. Section 3.2.2 refers to MAAC, whose
[original critic](https://github.com/shariqiqbal2810/MAAC/blob/master/utils/critics.py)
scales logits by the square root of key width. These support this production
choice; they do not establish what the MADSAC authors' unpublished code did.
Eq.(16) explicitly includes the own-agent contribution. Therefore MADSAC keeps
self-inclusive keys/values and does not adopt MAAC's exclude-self/raw-self shortcut.
Projection biases and the existing MLP architecture are unchanged.

The target remains Eq.(18): target_actor samples next actions, and bootstrap uses
`min(target_Q1, target_Q2) - alpha * next_log_prob`. Terminal/next-dead masks stop
bootstrap. The actor minimizes agent-summed, replay-batch-mean
`alpha * log_prob - min(Q1, Q2)`, using only the own action gradient for each Q_i.
Peer actions still enter the centralized critic numerically, but their actor
gradient is detached for that agent's term, consistent with Eq.(21).

No attention regularization, auxiliary objective, entropy-temperature optimizer,
normalization, dropout, gradient clipping, reward rescaling, prioritized replay,
or n-step return was added. Attention entropy and max weight are measured on the
existing critic forward every 100 Q rounds; these detached diagnostics never enter
the loss.

## Algorithm-1 update cycle

```text
T = transitions_since_update = 0
for each vector step:
    collect M joint environment transitions and insert into replay
    sampled_steps += M; vector_steps += 1; T += M
    if replay_size >= minibatch_size and T >= steps_per_update:
        update_cycle_count += 1
        repeat num_updates:
            batch_q = replay.sample(minibatch_size)
            trainer.update_critics(batch_q)  # Q1 and Q2 each optimize once
        if vector_steps % policy_delay == 0:
            repeat num_updates:
                batch_pi = replay.sample(minibatch_size)  # a new sample each time
                trainer.update_actor(batch_pi)
            trainer.update_targets()  # once for all three targets, after the loop
        T = 0
```

There is no extra 10,000-transition warmup. Readiness means replay can supply a
full minibatch and the cycle threshold is satisfied. T accumulates during replay
warmup, then resets to zero after each cycle; overshoot is discarded, not carried.
Algorithm 1 writes the delayed-policy condition in terms of the environment-step
index t. In the parallel implementation, one synchronized vector environment step
is used as the implementation counterpart of t: each vector.step_batch increments
vector_steps by one, while sampled_steps increases by M. The condition is exactly
`vector_steps % policy_delay == 0`, evaluated only inside a ready update cycle.
update_cycle_count remains accounting only. It does not determine policy delay;
sampled_steps is not used as the delay clock. The runner retains its global vector
step counter across independently resetting environments; no per-agent clock is
introduced. The numerical d=2 itself is still paper-unspecified/project reconstruction.

Each critic and actor iteration calls replay.sample separately. Independent draws
can contain overlapping transitions because sampling is with replacement; the
guarantee is separate draws, not disjoint indices. A delayed cycle with n=4 has
four actor optimizer steps but only one Polyak update of each target.

## Paper specifications and reconstruction choices

| Item | Setting / provenance |
|---|---|
| Shared actor, independent twin centralized attention critics | Paper, Sec.3.2 |
| Self contribution, embedding/attention/Q structure | Paper, Eq.(16), Figure 6 |
| Scaled dot-product | Figure 6(c), supported by referenced MAAC; Eq.(17) omits explicit scaling |
| Target actor / critics, double-min Q, reparameterization | Paper, Eq.(18)-(21) |
| Actor hidden 256/256, critic hidden 256, 2 heads | Paper, Sec.4.1; exact per-head split is implementation detail |
| Adam actor/critic lr=1e-4, replay=1e6, batch=1024 | Paper, Sec.4.1 |
| Fixed alpha=.1, gamma=.99, tau=.001 | Paper, Sec.4.1 |
| Delayed policy branch and update-cycle structure | Paper, Algorithm 1 |
| 24 parallel training environments, 20 evaluation seeds | Paper, Sec.3.3.1 / Sec.4.1 |
| Experimental sampling >8M | Paper, Sec.4.1 |
| steps_per_update=100, num_updates=4 | **PAPER-UNSPECIFIED; MAAC-reference reconstruction** |
| policy_delay=2; clock=vector_steps | **PAPER-UNSPECIFIED; project reconstruction** |
| Formal budget 8,400,000 | Project choice exceeding the paper's >8M; not an exact paper value |
| Evaluation interval 100,000 transitions | Project mapping; paper says every two training cycles without unique step conversion |
| Exact tanh Gaussian, log_std clamp [-5,2], ReLU | Paper-unspecified continuous-SAC reconstruction |
| Wq/Wk/Wv bias, LReLU slope .01, alive masks | Project implementation details / combat adaptation |

The [original MAAC training defaults](https://github.com/shariqiqbal2810/MAAC/blob/master/main.py)
are steps_per_update=100 and num_updates=4. **Li et al. do not report those numerical
MADSAC values, nor policy_delay=2.** Only the defaults are referenced: MAAC's exact
modulo trigger, actor-batch reuse, target frequency, and regularizers are not copied.
V5 follows the requested Algorithm-1 loops, separate actor samples, T=0, and delayed
target placement. Config metadata separates `paper_reported`, `paper_unspecified`
with reconstruction sources, and `project_implementation_choice`.

All checked-in `configs/madsac*.yaml` now describe v5 for their respective existing
environments. Their saved historical counterparts in `outputs/` are unchanged.
Historical v3 logs/audits remain legacy-v3 evidence, and v4 checkpoints/logs
remain legacy-v4 evidence using cycle-count delay. Neither is reinterpreted as v5.
Old v3/v4 checkpoints are incompatible; no weights are migrated. This document
retains its existing filename to preserve links.

## Accounting at 24 environments

Sampled steps count joint environment transitions, never agent slots. The first
full minibatch is available at `ceil(1024/24)*24 = 1032` transitions; T is already
above 100, so this is the first update. Subsequent cycles occur every five vector
steps, i.e. 120 transitions (the threshold overshoot is discarded).

Steady state: 4 paired-Q rounds per 120 transitions = 1/30 per-critic UTD;
two Q optimizers combined = 1/15; actor = 4/(2*120) = 1/60. This is close to,
but not identical to, legacy v3's per-critic 1/24. The effective spacing is 120,
not 100, at 24 environments and T=0; reporting 4/100 would be incorrect here.

Counters distinguish `critic_update_count` (paired-Q rounds) from
`critic_optimizer_steps` (Q1+Q2 steps). `target_update_count` counts grouped
three-network update events, not three times that number. The runner also logs
`update_cycle_count`, `actor_optimizer_steps`, both replay-sample counts and T.
The final cycle_complete log captures the target event and reset T.

## Checkpoints and validation

MADSAC_IMPL_VERSION=5. Metadata records `attention_logit_scaling=sqrt_head_dim`,
`attention_head_dim`, self inclusion, unchanged value activation/shortcut contract,
all three cycle parameters, `update_cycle_clock=vector_environment_steps_mod_policy_delay`,
`policy_delay_clock=vector_steps`,
`delayed_policy_condition=vector_steps % policy_delay == 0`, and
`target_update_schedule=once_per_delayed_update_cycle`. Both protocol validation
and direct `load_for_evaluation` reject v3/v4 and forged scaling/clock metadata. Save/load
restores all six network states for evaluation; exact training resume remains
unsupported because replay is not checkpointed.

`--smoke` no longer silently shrinks architecture, replay, batch, or changes update
schedule. Its default duration is short and it can schedule evaluation at the end.
The strict validation below uses the original 24-environment formal config and
only overrides total_sampled_steps=4800; all other hyperparameters remain intact.

```powershell
conda activate uav
python -u tools/validate_madsac_v4.py --output-dir outputs/madsac_v5_validation/new_run
python -m pytest -q tests/test_madsac_v4.py tests/test_madsac.py tests/test_madsac_v326.py tests/test_madsac_v3.py tests/test_madsac_algorithm_audit.py
```

The 4,800-transition run must produce 200 vector steps, 32 update cycles, 128
paired-Q rounds / 256 Q optimizer steps, 64 actor optimizer steps, 16 grouped
target updates (48 individual network Polyak calls), 128 critic and 64 actor
replay samples, and final T=48. Validation compares counters to actual optimizer
hooks, replay calls, target calls, checkpoint roundtrip, actual legacy-v3/v4 rejection,
20-seed evaluation, and the protected environment/MAPPO/MAAC file hash manifest.
This is schedule/correctness validation, not a trained-policy performance claim.


## Clock-specific regression

The formal update steps 43,48,53,58,... happen to give identical actor triggers
under cycle parity and vector-step parity for d=2. The smoke counts alone cannot
prove the new clock. CUDA unit tests therefore hold vector_steps fixed at 100
or 101 and vary the pre-cycle update_cycle_count between 0 and 1. Even vector100
always triggers four actor updates and one target event; odd vector101 never
triggers either, regardless of cycle parity. sampled_steps=24*vector_steps is
even in all cases, also ruling out an accidental sampled-step condition.
A separate count-reference test uses update spacing4 and odd vector steps
43,47,51, where three critic cycles must result in zero actor/target updates.
These are unit-test counterexamples only; formal 100/4/2 values are unchanged.
