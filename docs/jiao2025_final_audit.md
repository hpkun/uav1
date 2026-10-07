# Jiao et al. (2025) reproduction final audit

## Experimental role

- **PRIMARY FAIR BASELINE — Jiao-Core:** scalar round `F` + actor GRU + critic GRU + PopArt + paper-reported PPO hyperparameters, while retaining the frozen `persistent_wave_v2` raw reward.
- **SUPPLEMENTARY PAPER TRANSFER — Jiao-Full:** Jiao-Core plus Eq. (12) as a replacement training reward. It is excluded from the primary All-Off/Jiao-Core/WB-MAPPO ranking because the reward differs.

The source was checked directly rather than through the earlier project note. The implementation remains a transfer to the frozen 4v4, 3-D, nearest-target-Blue benchmark, not a reproduction of the paper's 3v3 planar simulator.

## Source-to-implementation audit

| Paper statement | Location | Current implementation | Classification | Audit result |
|---|---|---|---|---|
| State includes the scalar round count `F`. | Sec. 2.3, Eq. (7) | Append one unnormalised scalar to every actor and centralized-critic agent input: W1/W2/W3 = 1/2/3; 52D becomes 53D. | PAPER_SPECIFIED | PASS |
| The swarm is fully cooperative and shares an objective/reward. | Sec. 2.5, Eq. (8) | Full sums all new Red/Blue death terms into one team R2 and copies it to Red agents alive at transition start. | PAPER_SPECIFIED | PASS |
| Actor and value networks use GRU recurrence; trajectories store hidden state and are split into chunks. | Sec. 3.1, Algorithm 1/2 | Actor backbone→GRU→Gaussian heads and project critic backbone→GRU→value head; saved pre-step hidden initializes contiguous BPTT chunks. | PAPER_SPECIFIED | PASS |
| Red UAVs share actor and critic parameters. | Sec. 3.1 | One shared actor and one shared critic process all four homogeneous Red agents. | PAPER_SPECIFIED | PASS |
| Critic input is described both as not using centralized observation and as taking global observation. | Sec. 3.1, adjacent sentences | Retain the project's centralized attention critic. | SOURCE_NOTATION_AMBIGUITY / COMMON_PROJECT_BACKBONE_ADAPTATION | DISCLOSED |
| PPO actor uses clipping and entropy. | Eq. (9) | Existing stable squashed-Gaussian PPO ratio, clipping and entropy path. | PAPER_SPECIFIED / COMMON_PROJECT_BACKBONE_ADAPTATION | PASS |
| Value objective is clipped. | Eq. (10) | Existing clipped value loss. | PAPER_SPECIFIED | PASS |
| Algorithm 1 computes GAE and also labels the PopArt target reward-to-go. | Algorithm 1 | Advantage is GAE; target is `GAE + value` (lambda return), then PopArt-normalized. It is not claimed to be pure Monte Carlo return. | PAPER_AMBIGUOUS / COMMON_PROJECT_BACKBONE_ADAPTATION | DISCLOSED |
| PopArt normalizes value targets. | Algorithm 1 | Running mean/variance/count plus output-preserving rescaling of recurrent critic `value_head`; checkpoint/resume includes state. | PAPER_SPECIFIED | PASS |
| Blue death reward is `(j+1)(L+1)` and Red death penalty is `-(i+1)`. | Eq. (11), Eq. (12) | Paper `L+1` is mapped to one-based current `wave_index`; Red loss is not wave-scaled. | PAPER_SPECIFIED / SOURCE_NOTATION_AMBIGUITY | PASS; symbol conflict disclosed |
| `L` is used as a round-related multiplier while `L^r/L^b` denote survival states. | Eq. (11), Eq. (12) | Mapping is explicit and isolated in `jiao_r2_replacement`. | SOURCE_NOTATION_AMBIGUITY | DISCLOSED |
| Death is represented through survival state becoming zero; the paper simulator does not separate boundary/ground causes. | Eq. (11), Eq. (12), scenario description | Any reliable alive 1→0 event maps to R2, including boundary/ground loss; cause totals are logged separately. | PAPER_AMBIGUOUS / NONCOMBAT_DEATH_ADAPTATION | DISCLOSED |
| Equal-rank cooperative agents nevertheless receive index-dependent death coefficients. | Sec. 2.5 versus Eq. (12) | Preserve coefficients 1/2/3 and naturally extend to 4 for the fourth project UAV. | SOURCE_DESIGN_TENSION / PAPER_UNSPECIFIED_ADAPTATION | PASS; no reweighting |
| Actor/critic LR .0005, epochs 10, clip .1, entropy .01, lambda .95, gamma .99. | Table 2 | Exact numerical values in Core and Full. | PAPER_SPECIFIED | PASS |
| Episode length is 150. | Table 2 | Map 150 to PPO rollout horizon while frozen environment `max_steps=3000` remains unchanged. | PAPER_UNSPECIFIED_ADAPTATION | DISCLOSED; not called a paper rollout length |
| Paper action is `[v, omega]`; scenario is 3v3 planar and new Blue groups are dispatched between rounds. | Sec. 2.3–2.4 | Frozen 3-D three-action, homogeneous 4v4, three-wave benchmark and nearest-target Blue. | COMMON_PROJECT_BACKBONE_ADAPTATION | DISCLOSED |
| Network widths, GRU width, chunk length, parallel env count, minibatch and 1.5M budget are not fully specified. | Paper/Table 2 omissions | 256 backbone, GRU 128, chunk 32, 24 envs, minibatch 512, seed 2023, 1.5M sampled steps. | PAPER_UNSPECIFIED_ADAPTATION | DISCLOSED |

## Recurrent and return semantics

The paper specifies recurrent trajectories and chunk restoration but does not define hidden-state handling at a persistent wave spawn, rollout boundary, agent death, or environment truncation. The project adaptation is: zero at episode start; persist over ordinary steps, W1→W2, W2→W3 and rollout boundaries; immediately zero a dead Red agent; reset all agent hidden state only on true termination/truncation. Training evaluation and post-training analysis call the same canonical episode evaluator.

Wave clear is not `done`, so GAE bootstraps and recurses through W1→W2 and W2→W3. True termination/truncation stops recursion; an individual dead agent is also masked because it has no subsequent policy sample.

## Eq. (12) event semantics and diagnostics

Full is strict replacement: `training_reward = team_R2`, never `raw + R2`. A death is a one-time transition-start alive 1→0 event. The final death of a cleared wave is recovered before the environment's immediate Blue respawn mask can hide it. Simultaneous and mutual deaths are summed. Agents already dead at transition start receive zero and are absent from PPO alive loss.

The runner records raw and Jiao returns, Blue/Red R2 components, wave-specific R2, death counts for indices 0–3, and aggregate weapon/boundary/ground causes. Cause identity is not guessed beyond reliable environment counters.

## Fairness, leakage, and frozen-state audit

All-Off and M5/WB-MAPPO snapshots both use seed 2023, 1.5M sampled steps, 20 deterministic validation episodes starting at 10,000,000, evaluation every 100k sampled steps, and the same persistent-v2 lexicographic best key `(W3, average waves, raw return, -Red loss)`. Core and Full now match this information budget.

The originally reserved 37,000,000–37,000,099 range cannot truthfully remain fresh: retained local smoke artifacts prove that Core and Full each evaluated two episodes starting at 37,000,000 before this audit. The evidence was not deleted. That entire range is permanently rejected, and the untouched replacement is 38,000,000–38,000,099. It remains disjoint from training validation and the 20M formal holdout. Focused smoke uses only 99M+.

New modular evaluation-history rows persist both `evaluation_seed_base` and `evaluation_seed_end`. Post-training analysis requires those fields for Jiao runs and rejects missing, mismatched, contaminated-37M or premature-38M provenance before creating any comparison cache.

Frozen SHA-256 guards cover the All-Off config, M5 config, environment config semantics and the complete `env/*.py` source tree. No environment file or frozen control config is modified by this work.

The source-tree guard is computed deterministically from each repository-relative POSIX path, a NUL separator, file bytes and a trailing NUL. Its locked value is `9f5726802979ec42394761515c5da2d4a832f2b9f6138b5611eaf4c1bd599c15`; this replaces an earlier stale preflight constant without changing any `env/` file.

## Final configuration and gate

Core enabled modules: `wave_context`, `recurrent_memory`, `popart`; reward adapter, wave balance, warm start, curriculum and anchor are off. Full adds only `multi_wave_reward: jiao_r2_replacement`.

## Verification result

- `python -m compileall algorithm tools tests`: PASS.
- Full test suite: **475 passed**; the focused Jiao file contributed 23 passing tests.
- CUDA Core smoke: 4,096 then resume to 8,192 sampled steps; 512 optimization rows finite, actor/critic GRU gradients nonzero, PopArt finite, raw and training reward sums identical.
- CUDA Full smoke: 4,096 then resume to 8,192 sampled steps; 512 optimization rows finite, actor/critic GRU gradients nonzero, PopArt finite, and R2/raw channels distinct.
- Both smoke runs used 99M+ only and completed checkpoint, resume and two-episode final evaluation on CUDA.
- Static preflight generated `outputs/jiao2025_screening_protocol.json`; it contains no performance result and marks fresh comparison unexecuted.

Final gate: **READY_FOR_1P5M_SCREENING**. Smoke metrics are not research results. No formal 1.5M training, 38M comparison or 20M holdout was run during this audit.
