# Jiao-2025 matched 3M transfer protocol

Source: Yongkang Jiao et al., *Collaborative decision-making for UAV swarm
confrontation based on reinforcement learning*, IET Control Theory &
Applications 19, e12781 (2025), DOI 10.1049/cth2.12781.
Local original: `papers/Jiao 等 - 2025 - Collaborative decision‐making for UAV swarm confrontation based on reinforcement learning.pdf`.
Sec.3.1/Algorithm1 and Table2 were checked directly. This is a paper-aligned
framework transfer, NOT an exact reproduction of the original simulator.

## Two variants

| Property | Jiao-Matched Plain | Jiao-Core |
| --- | --- | --- |
| Raw observation/action/Red agents | 52/3/4 | 52/3/4 |
| Actor | 52 -> MLP256-256 -> Gaussian3 -> tanh | 53 -> MLP256-256 -> GRU128 -> Gaussian3 -> tanh |
| Critic | Project centralized attention2 backbone -> value | Same project encoder + scalar F -> GRU128 -> value |
| Context | None | Unnormalized scalar F=1/2/3 to BOTH networks |
| PopArt | OFF | beta=.999, epsilon=1e-5 |
| Active modules | None | wave_context, recurrent_memory, popart ONLY |

Shared: seed5301, exact3M, Adam, constant Actor/Critic LR5e-4, gamma.99,
lambda.95, clip.1, entropy.01, value coefficient.5, gradient clip.5,
rollout256, 24 environments, minibatch512, 10 actual PPO epochs, ReLU,
existing log_std[-5,2], normalized advantages, clipped value loss.
Frozen environment: persistent_wave_v2, 4 Red vs Blue[4,3,3], H3000.
No environment file or raw observation is modified.

Development validation: deterministic tanh(mean), 50 episodes,
44000000..44000049, every100k; exact3M final is primary, best diagnostic only.
Checkpoint interval remains300k. 45000000..45000199 is reserved, unexecuted.
Eq.12 reward replacement, MARC, balancing, PBRS, curriculum, warm start,
anchor, entity attention, mission-context additions, LR decay and KL guards
are all OFF. Training reward is exactly the frozen raw environment reward.

## Source classification

PAPER-SPECIFIED: scalar round F, recurrent Actor/value networks, trajectory
chunks, PopArt, LR5e-4, epochs10, clip.1, entropy.01, lambda.95, gamma.99.

PROJECT ADAPTATIONS: 433 3-D combat, continuous three-action tanh-Gaussian,
width256, GRU128, BPTT32, rollout256, 24 environments, minibatch512,
centralized attention Critic, PopArt beta/epsilon, 3M and development bank.
Table2's episode length150 is NOT a PPO rollout specification. New rollout256
does not reinterpret150. Historical 1.5M files and historical notes remain
preserved as historical provenance, not authority for this new protocol.

Ambiguities: the paper's centralized/global value input wording is inconsistent;
the attention encoder is project infrastructure, not a Jiao mechanism.
Algorithm1 names reward-to-go while the project retains GAE+value lambda-return.
PopArt normalizes that target and preserves raw value outputs by rescaling the
recurrent value head. Raw rewards are never normalized/replaced.

## Reused infrastructure and narrowly scoped correction

Existing scalar_round, Gaussian Actor/attention Critic GRUs, canonical
evaluation, saved pre-action hidden states, detached chunk starts, dead masks,
PopArt rescaling/checkpoint and RNG/optimizer resume are reused.
Core hidden starts atzero, carries across ordinary steps, rollout boundaries
and Blue respawns; only F changes after the clearing action. Death immediately
masks that Red's Actor/Critic hidden; true termination/truncation resets the
next episode. No wave reset or phase initialization is enabled.

The existing ordinary recurrent path could include an episode reset INSIDE a
fixed32-step chunk. The NEW Core path uses episode_contiguous_chunks to split
at true episode starts and32-step limits, never at wave changes. Historical
chunk selection is unchanged. All joint recurrent paths gain counter assertions
and diagnostic epoch/optimizer counts without changing losses or optimizer math.
Every rollout step is covered once per epoch, sequences remain ordered, and
Actor/Critic use their detached saved chunk-start hidden.

A/B use the same training seed and environment initialization/evaluation protocol.
They do NOT have matched full parameter tensors: input sizes and recurrent
architecture differ. No network construction was redesigned to force matching.
The package experiment does not individually identify F vs GRU vs PopArt effects.

## Verification tools

`tools/preflight_jiao2025_core_3m.py`: read-only exact-pair/config/env-SHA,
architecture, constant LR and fresh-output checks.
`tools/smoke_jiao2025_core_3m.py`: each variant12 actual environment samples,
controlled wave-index signals, PPO10 epochs, save/resume and canonical six-step
deterministic evaluation; only99M diagnostic seeds. Controlled signals are
lifecycle checks, not evidence of actual combat clears.

No 3M training is launched by these tools. Historical regression dependencies
must not be faked or weakened to produce a readiness label.

## Verification and remaining historical prerequisites

New focused tests:24 passed. Extended related regression:365 passed,2 failed.
The two failures are unchanged historical checks: missing
`outputs/pw_alloff_matched_1p5m_seed2023/algorithm_config.yaml`, and historical
environment-source lock `0aacc84d...` differs from current source `721baf2a...`.
No historical fixture was fabricated, no old lock was updated, no test edited
or skipped in source. A subsequent explicit run excluding those two checks
reported365 passed,2 deselected; that is not a claim that the full gate passed.

The new current-protocol preflight pins the frozen433 config SHA
`50d42e96...` and current environment source SHA `721baf2a...`, independently
of the preserved1.5M lock. New preflight PASS; both CUDA tiny smokes PASS,
including Actor/Critic optimizers, RNG, parameters and PopArt resume. No44M
or45M scenario is executed by these smokes. Both formal output directories
are absent. The overall all-tests-passed readiness label is withheld pending
the historical evidence prerequisites.
