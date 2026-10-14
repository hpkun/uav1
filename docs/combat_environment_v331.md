# Combat Environment v3.31: R4 coefficient scale ablation

CombatEnvironmentV331 directly subclasses CombatEnvironmentV330. Its only
scientific change relative to v3.30 is multiplying all six R4 offense/threat
coefficients by two. The class overrides only environment_version; observation,
transition, reward calculation, weapon, controller and Blue policy are inherited.
v3.30 remains the original reward baseline, including mutual destruction -> draw.

The experiment asks whether near-range tactical reward is too weak in the
current 4v4 / 1km weapon setting. The proposed distance-scale rationale compares
an approximately 4km reward support with the present w4(d) =
clip((3000-d)/2000, 0, 1). The latter integrates to approximately 2km: a 1km
plateau plus the area of a 2km triangle. The initial heuristic calibration is
4/2 = 2. This does not imply that actual cumulative episode reward doubles:
trajectories, geometry gates, aggregation and unchanged R1-R3 still matter.

## Complete environment configuration differences

| Field | v3.30 | v3.31 |
|---|---:|---:|
| environment_version | 3.30 | 3.31 |
| reward.offense_reward_30deg | 0.01 | 0.02 |
| reward.offense_reward_15deg | 0.02 | 0.04 |
| reward.offense_reward_5deg | 0.10 | 0.20 |
| reward.threat_penalty_30deg | -0.015 | -0.030 |
| reward.threat_penalty_15deg | -0.025 | -0.050 |
| reward.threat_penalty_5deg | -0.15 | -0.30 |

No other environment configuration field changes. validate_v331 checks these
six exact doubled values, maps a deep copy back to v3.30, and invokes its
validator. Original coefficient ratios and the stronger threat magnitudes
remain unchanged.

## Frozen mechanics and reward conditions

R1 (+10 kill / -10 loss), R2 (-10 boundary) and R3 (0.001 guide) are unchanged.
R3 retains its body azimuth/elevation gates, w3 and max aggregation. R4 retains
its 3000m distance cutoff, continuous w4, 30-degree AA gate, 30/15/5-degree body
azimuth/elevation levels and all reverse-geometry threat conditions. Offense
uses max and defense uses min over the same live pairs. No gate is removed or
relaxed, no angle shaping or new reward is introduced.

Real 4v4 / 45D observations, [delta heading, delta pitch, delta speed] actions,
the shared 10G controller, Blue 225m/s step-start nearest-live pursuit/fire
coupling, Red eligible-target selection, 1km +/-30-degree weapon, six rounds,
1s cooldown, hit model, simultaneous resolution, initial formation/distance,
dt, horizon, bounds and outcome semantics remain unchanged.

## Algorithms and diagnostics

New MAPPO/MADSAC/MAAC configurations retain every v3.30 network and training
parameter. Only existing environment identity metadata changes to 3.31;
the MAPPO config has no environment identity field and is identical to its
v3.30 counterpart. Explicit version support is added to environment factories,
algorithm protocols and diagnostic version lists. Checkpoint identity remains
strict, and v3.30 checkpoints are incompatible with the v3.31 environment.

Existing episode reward diagnostics already expose cumulative R1, R2, R3,
tactical_offense and tactical_defense. Evaluations retain these fields;
no diagnostic is inserted into observations, actions, losses or checkpoint
ranking. No new training statistics mechanism is needed.

## Validation

tests/test_combat_v331.py covers exact coefficient scaling, all body-angle
levels, forward/reverse AA and angle gates, zero-distance-weight cases,
nonzero R3, max/min aggregation, event rewards, inherited mutual draw,
dimensions, frozen methods/reset and unchanged algorithm configurations.

tools/audit_combat_v331_reward.py is network-free and runs seeds 0-199 with
the same nearest-pursuit mathematics used by the v3.30 fairness audit. Both
versions execute identical actions and are compared at every transition:
states, observations, RNG, ammo/cooldown/targets, weapon counts and outcomes
must match exactly; R1-R3 must match and per-step R4 components must be exactly
twice the baseline. It records real per-agent and team cumulative components,
mean/median/p90/p95/max absolute, and actual static pair geometry. Ratios for
zero reward terms are undefined rather than treated as successful scaling.

Artifacts are under outputs/v331_validation. CUDA smoke uses the existing
reduced smoke mode; formal configuration hyperparameters remain unchanged.
No formal training is started. Large dense reward totals are reported as
observations of this fixed-policy audit, not used to automatically retune R4.

## Completed 200-seed audit

Both versions produced Red win=27, Blue win=26, draw=147, timeout=0, mutual
destruction=147, Red/Blue attack kills=774/773 and noncombat losses=0/0.
Every simulated transition passed the identical-physics/event/outcome checks.
All 200 episodes had zero R4 offense and zero R4 defense. Consequently their
actual cumulative reward is identical between versions. Nonzero x2 scaling
is established by the deterministic geometry tests, not by these zero-R4
episodes. This mirror audit cannot establish the practical effect of R4
rescaling on reward strength or learning when R4 is active.

The following cumulative statistics apply to both versions. Per-agent rows
pool four Red aircraft across 200 episodes (800 samples); team rows sum the
four Red aircraft within each episode (200 samples). P95 is a signed quantile;
the last column is the maximum absolute value.

| Scope / component | Mean | Median | P90 | P95 | Max absolute |
|---|---:|---:|---:|---:|---:|
| Agent R1 | 0.012500 | 0 | 0 | 0 | 15 |
| Agent R2 | 0 | 0 | 0 | 0 | 0 |
| Agent R3 | 0.134368 | 0.134372 | 0.134918 | 0.134918 | 0.134927 |
| Agent R4 offense | 0 | 0 | 0 | 0 | 0 |
| Agent R4 defense | 0 | 0 | 0 | 0 | 0 |
| Agent dense (R3+R4) | 0.134368 | 0.134372 | 0.134918 | 0.134918 | 0.134927 |
| Agent total | 0.146868 | 0.134368 | 0.134918 | 0.134918 | 15.134927 |
| Team R1 | 0.050000 | 0 | 10 | 10 | 10 |
| Team R2 | 0 | 0 | 0 | 0 | 0 |
| Team R3 | 0.537473 | 0.537473 | 0.537473 | 0.537473 | 0.537482 |
| Team R4 offense | 0 | 0 | 0 | 0 | 0 |
| Team R4 defense | 0 | 0 | 0 | 0 | 0 |
| Team dense (R3+R4) | 0.537473 | 0.537473 | 0.537473 | 0.537473 | 0.537482 |
| Team total | 0.587473 | 0.537473 | 10.537473 | 10.537473 | 10.537482 |

No dense-reward accumulation remotely exceeding the +/-10 event scale appears
in these trajectories. The larger per-agent total maximum comes from event
reward, not dense shaping. No coefficients or gates were retuned after the audit.

CUDA MAPPO/MADSAC/MAAC smoke sampled 32 transitions each, performed updates,
saved/loaded checkpoints and completed evaluation. Formal configs are frozen.
There are 36 v3.31-specific tests and 502 historical/environment/algorithm
tests, for 538 passing related tests in total.
