# Persistent-Wave / Multi-Round environment audit

This document describes the implemented persistent-wave variants. Version 1 is a
minimal wrapper around the frozen V2.3 direct-combat environment, not the
previous V3 redesign proposal. Version 2 is the current primary environment.

## Frozen contracts

- Four homogeneous learned Red UAVs fight four Blue UAV slots.
- Actor observation remains 52 dimensions and action remains 3 dimensions.
- V2.3 point-mass dynamics, controller, probabilistic entry-triggered weapon,
  simultaneous-hit resolution, and R1-R4 rewards are unchanged.
- Blue uses only `NearestTargetPursuitPolicy`.
- There is no ammunition count, cooldown, Hungarian assignment, communication
  model, inter-wave countdown, explicit missile, or extra boundary reward.

## Mission configuration

`configs/persistent_wave_environment.yaml` selects
`environment_variant: persistent_wave_v1`. The default mission contains three
waves and has one global horizon of 3,000 physics steps (`dt=0.1`, 300 seconds).
Every replacement Blue wave has four live aircraft. A candidate formation is
centered at radius 4,400 m. The environment enumerates 72 perimeter directions
at five-degree intervals, generates the complete perturbed four-aircraft
formation for each direction, and selects the smallest-index candidate whose
minimum 3D distance to every surviving Red aircraft is greatest. Configuration
validation proves all 72 formations fit inside the 5,000 m arena.

There is no rejection loop, hard minimum-distance threshold, fallback, or
spawn-time fire-window exclusion. Actual minimum spawn distance and the selected
candidate index are diagnostics. A fresh wave may start in a normal V2.3 weapon
window; it appears in the returned next observation and can only participate in
weapon resolution on the following call to `step`.

## Exact boundary order

1. `MultiUAVCombatEnv.step` advances both teams, applies boundary/ground loss,
   snapshots post-motion state, computes R1-R4, evaluates both entry-triggered
   attacks, resolves simultaneous hits, and creates the old-wave reward/info.
2. If Red survives and every Blue slot is dead, the wrapper closes an immutable
   per-wave metric record and increments `waves_cleared`.
3. On a non-final wave before the global time limit, it enumerates and selects a
   fresh Blue formation, resets both Red and Blue `FireState` objects, and zeros only the
   new Blue wave's last executed bank values.
4. Red physical state, live/dead slots, and last executed bank values persist.
   `wave_index` increments, the returned observation is rebuilt from the new
   Blue wave, and both terminal flags are false.
5. The clearing-step reward is returned unchanged. Final-wave clearance uses
   the original V2.3 success terminal. Global timeout prevents replacement.

The returned `red_alive_mask`, `blue_alive_mask`, survivor counts, loss counts,
observation, and wave fields are all rebuilt or overwritten after replacement
so they describe the same post-transition state.

## Construction and checkpoint contract

All formal vector training and deterministic evaluation create environments via
`make_combat_environment`. Each subprocess reports its actual class and variant
at startup for auditability. Training checkpoints store both
`environment_version` and `environment_variant`; Direct and Persistent
checkpoints are rejected on resume in either direction. Explicit diagnostic
weight loading remains possible in standalone evaluation tools.

The Direct MAPPO algorithm file remains unchanged at `gamma: 0.99`. Persistent
runs use `configs/mappo_persistent_wave.yaml`, which uses `gamma: 0.999`. With 0.1-second
steps, preserving a conventional per-second discount gives
`0.99 ** 0.1 = 0.998995...`, hence the rounded value.

Example:

```powershell
python algorithm/train_mappo.py --device cuda --seed 2023 --num-envs 24 `
  --total-sampled-steps 8000000 `
  --env-config configs/persistent_wave_v2_environment.yaml `
  --algorithm-config configs/mappo_persistent_wave.yaml `
  --output-dir outputs/mappo_persistent_wave_v2
```

## Per-wave and mission metrics

Every completed wave records start/end/duration steps, Red survivors at both
ends, Blue survivors, each side's attempts/hits/combat kills/boundary exits/
ground losses, R1-R4 totals, team return, `wave_cleared`, and the wave's terminal
reason. A final partial wave is also closed on Red elimination, mutual
destruction, or mission timeout, so terminal episodes never lose their last
wave record. Evaluation and final training
summaries add mean waves cleared, unconditional probability of clearing each
wave, conditional mean Red survivors after each cleared wave, total Blue/Red
losses, and a loss-denominator-clipped kill/loss ratio.

Direct checkpoints retain the lexicographic selection key
`(win_rate, average_return, -average_red_loss)`. Persistent checkpoints use
`(clear_wave_3_probability, average_waves_cleared, average_return,
-average_red_loss)` so progress before the first full three-wave success can
still update `best_eval.pt`.

`tools/validate_persistent_wave_environment.py` performs the pure-environment
100,000-case replacement stress audit across 1-4 Red survivors and center,
boundary, spread, altitude, heading, and speed layouts. It records selected
sectors, distance percentiles, immediate weapon windows, and the following one
second of normal dynamics without treating a weapon window as a spawn failure.

## Read-only Markov audit

The 52-dimensional actor observation does not contain `wave_index` or remaining
wave count. For a fixed physical 4v4 state, within-wave dynamics, Blue policy,
weapon behavior, and reward are independent of the index. The hidden variable
matters only when a wave is cleared: the same observed clearance state either
spawns another wave or terminates on the final wave. Therefore the actor-facing
process is formally non-Markov at mission-boundary/value-estimation level, even
though local flight and engagement control remain Markov with respect to the
unchanged physical observation. A feed-forward critic cannot distinguish these
two continuation values. This is a real experimental limitation to disclose,
but resolving it would require the explicitly forbidden 53D/mission-context
observation change and is outside `persistent_wave_v1`.

## Persistent wave v2 ground avoidance

`persistent_wave_v2` preserves every v1 mission, spawn, weapon, reward,
observation, and Red-side transition rule. Its only semantic change is the
fixed Blue policy. Direct V2.3 and `persistent_wave_v1` continue to construct
the original `NearestTargetPursuitPolicy`; v2 constructs
`GroundAwareNearestTargetPursuitPolicy`.

The v2 policy first computes the same nearest-target heading, executable LOS
pitch command, and desired speed as v1. Let altitude be `h`, speed be `v`,
current flight-path pitch be `theta`, and the nearest-target LOS elevation be
`theta_los`. It computes

```
v_down = max(-v sin(theta), -v sin(theta_los), 0)
t_ground = h / v_down
```

when `v_down > 1e-6 m/s`. If `t_ground <= 2 * pitch_time_constant`, only the
pitch target is replaced with `theta_max`; heading and speed remain the normal
nearest-target commands. The check is stateless and is recomputed each step.
There is no fixed safe altitude or protected band.

Because this changes the environment-owned Blue transition semantics, v1 and
v2 checkpoint identities are intentionally incompatible for resume. Explicit
cross-variant diagnostic evaluation must name the checkpoint's source variant.
