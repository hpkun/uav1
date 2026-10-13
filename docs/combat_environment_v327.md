# Combat Environment v3.27

Direct baseline: CombatEnvironmentV326. The only behavior change is persistent
Blue pursuit targeting, with the weapon still coupled to the same target.

## CALIBRATION / paper-consistent reconstruction

The paper's public description supports nearest-target pursuit but does not
specify whether targets can change every simulation step or remain locked until
invalid. Persistent lock until target death is therefore a calibration /
paper-consistent reconstruction, not a PAPER exact rule.

Each Blue acquires the current nearest live Red in 3D using the unchanged
authoritative nearest_target_index selector (index tie-break). It keeps that Red
while it is alive, even when another Red becomes closer. Multiple Blue may lock
the same Red: there is no assignment/balancing algorithm.

## One authoritative lock and step timing

The environment owns locked_target_indices: int64[5], initially/reset -1.
At step start, surviving Blue keep live targets and reacquire only absent/dead
targets. Dead Blue clear to -1. A successful reacquisition after the previous
target's death increments blue_target_switches; initial acquisition does not.

The environment copies locks into blue_step_target_indices before motion. The
thin V327LockedTargetBluePolicy holds a reference to that snapshot, not a second
lock state. team_actions uses it to compute the same desired heading/elevation/
speed and inherited v3.21 action encoder. v3.26's _entry_attempts uses the same
snapshot. No target selection occurs after dynamics or death resolution.

The wrapper skips only v3.26's stateless step-start reselection, calling the
fully inherited physical step implementation. Reward, dynamics, observation,
noncombat/combat resolution and Red firing remain the same methods.

If a Red target dies during the step, the snapshot and persistent lock retain
that dead index until the next step. The dead pair cannot fire. At the next step
Blue reacquires the nearest survivor. Blue dying during a step clears its
persistent lock at step end; the snapshot still describes that step. External
blue_actions override movement only; lock update and weapon binding still apply.

step info contains copies of blue_pursuit_target_indices (frozen step snapshot)
and blue_locked_target_indices (end-of-step persistent state). Policy counters
are diagnostics only; they do not select targets. Reset clears locks, snapshots
and policy counters, so no target state leaks between episodes.

## Firing and frozen mechanisms

Weapon firing window is fully retained: alive pair, ammo, range=1000m, body
azimuth/elevation ±30 degrees; actual firing additionally requires cooldown=0.
An ineligible locked target suppresses firing even when another target qualifies.
No temporary off-target shot is allowed. A continuously qualifying locked target
can fire repeatedly at S,S+10,S+20,... without exiting/reentering the window.
Ammo remains six and target switching neither resets cooldown nor restores ammo.

No ground avoidance or recovery logic is added. Symmetric ground/horizontal/
ceiling deaths, timeout and outcome are unchanged, even if Blue still crash.
Signed 56D observations, feasible controller, RK4/dynamics, Blue speed=250,
body rewards, continuous w3/w4, zero Safety/terminal/team reward, hit model and
Red weapon target selection are all inherited. No reward parameter changes.

Environment YAML differs only by version. MAPPO YAML is byte-identical to v3.26;
MADSAC YAML differs only in its environment identity. MADSAC implementation stays
3, including its self-inclusive attention critic. Networks and schedules are
unchanged. Both algorithm checkpoint protocols reject cross-environment loads.
Recording uses existing dynamic dispatch.

## Validation

v3.27 tests: 16 passed. Related environment/protocol regressions: 196 passed
(v3.26 18, v3.25 75, v3.24 37, v3.23 64, factory/MAPPO checkpoint 2).
MADSAC-related tests: 66 passed (includes the v3.27 checkpoint test).

Tests cover initial locks, closer alternatives, locked-target window suppression,
ground/horizontal/ceiling deaths, other-Blue kills, next-step switches, no
same-step fallback, diagnostic copies, external Blue actions, six-round cadence,
cooldown across switches, reset/dead Blue/3D ties, Red kill regression, fixed
actions without lock-switch differences, config and checkpoint isolation.

CPU environment smoke: four seeds, 100 steps each, finite states/observations/
rewards. CUDA short training smokes: MAPPO and MADSAC each complete 1200 environment
transitions with existing smoke widths/settings. These are smoke-only overrides;
formal configs are unchanged. Detailed outputs are under outputs/v327_validation.
No formal 300k/3M training runs.
