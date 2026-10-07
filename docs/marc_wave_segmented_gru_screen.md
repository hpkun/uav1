# MARC Credit+Balance phase-GRU development screen

This is a new development method, not MARC V3 or a change to formal MARC V2.

| Item | Control | Treatment |
| --- | --- | --- |
| Method | marc_credit_balance_ablation | marc_mappo_wsgru_v1 |
| Actor | existing feed-forward MLP | 52 → 256 → 256 → GRU(128) → Gaussian heads |
| Modules | actor_lr_decay, milestone_aware_retention_credit | same + recurrent_memory |
| Retention | coefficient 0 | coefficient 0 |
| Critic | existing feed-forward centralized attention | identical architecture/initialization |

Both inherit the existing 1M MARC/common training parameters: seed5301, 24
environments, rollout256, 10 PPO epochs, minibatch512. The frozen environment is
persistent_wave_v2, Red4, Blue[4,3,3], max_steps3000, observation52, action3.
Validation is deterministic development-only 44000000..44000049, not final
holdout. The reserved 45M bank is not executed.

## Changed files

- algorithm/modules/recurrent_memory.py
- algorithm/modules/milestone_aware_retention_credit.py
- algorithm/modular_mappo/networks.py
- algorithm/modular_mappo/buffer.py
- algorithm/modular_mappo/factory.py
- algorithm/modular_mappo/protocol.py
- algorithm/modular_mappo/trainer.py
- algorithm/modular_mappo/runner.py
- algorithm/modular_mappo/evaluation.py
- tools/record_combat_episode.py
- tools/preflight_actor_only_gru_history.py (scope the existing source audit to
  the legacy carry path; no old-GRU runtime change)
- configs/dev_marc_credit_balance_1m.yaml
- configs/dev_marc_wsgru_v1_1m.yaml
- tools/preflight_marc_wsgru_screen.py
- tools/smoke_marc_wsgru_screen.py
- tests/test_marc_wsgru_screen.py
- docs/marc_wave_segmented_gru_screen.md

Final verification: py_compile PASS, 223 related tests passed / 0 failed,
CUDA preflight PASS, tiny CUDA Control and Treatment updates/resume PASS.
Each method took8 actual vector-environment sampled steps and a six-step
controlled canonical evaluation. Constructed phase data exercised all three
phase gradients and10 epochs. Both formal output directories were absent.

## Lifecycle and BPTT

The Actor owns a zero-initialized, trainable [3,128] phase parameter. Each alive
agent starts an episode with e1 and gets e2/e3 **before** its first decision on
the newly spawned wave. Wave-clearing actions belong to the previous wave.
Dead-agent hidden is zero. Sampling/rollout boundaries alone do not reset it.

Rollouts explicitly store pre-action actor_recurrent_phase_reset_flags [T,E].
The new chunk planner validates this provenance and covers every transition
once, without crossing episode or wave boundaries, at most32 steps per chunk.
Phase-start chunks use the current live phase parameter, with a gradient path;
ordinary truncated chunks use detached saved pre-action hidden. The old
actor_gru mode still uses its old planner and cross-wave carry semantics.

The combined path preserves the existing MARC global/local/continuation
advantages, their normalization order, and tempered actor wave weights. The
critic still learns full-horizon returns using its existing flat PPO update.
Entropy is unweighted as before. Both optimizer counts and all10 epochs are
checked; mismatches raise RuntimeError. Zero-coefficient V2 skips ingestion,
sampling and state-only elite losses; positive-coefficient formal V2 is
unchanged and continues to reject recurrent memory.

## Verification entry points

- tools/preflight_marc_wsgru_screen.py: read-only CUDA, matched-config and
  initialization checks, refuses occupied formal outputs.
- tools/smoke_marc_wsgru_screen.py: CUDA tiny actual rollout/update, controlled
  three-phase PPO update, checkpoint/resume, six-step controlled canonical
  evaluation; never train(), never formal50-episode evaluation.
- tests/test_marc_wsgru_screen.py plus existing MARC/old-GRU/regression suites.

Periodic, best/final and standalone modular evaluation use the same canonical
evaluator. The recording tool uses the same pre-action phase flags. Unsupported
historical callers that omit wave/reset provenance fail explicitly rather than
silently deploying the new Actor with legacy hidden semantics.

## Human screen gate (not implemented as automatic PASS/FAIL)

Compare AW, W1/W2/W3, Q2/Q3, Return, Boundary and Ground. A useful first-stage
signal is AW gain≥0.20 plus W3 gain≥0.08 or Q3 gain≥0.10, without substantial W1
or safety degradation. This single-seed development screen is not a statistical
confirmation or final-test result.
