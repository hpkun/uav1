# Combat Environment v3.36

`CombatEnvironmentV336(CombatEnvironmentV335)` changes only the weapon's angular
geometry. **v3.36 restores the paper ATA/HA angular firing and hit geometry while
retaining the project's 1 km engagement range and fire-control extensions.**

The project's Li et al. (2023) PDF, Section 2.4, page 7, Eq.(7) uses distance,
ATA and HA for firing qualification. Eq.(8) uses those same quantities in the
shared-Gaussian hit test. This is not an exact reproduction of the full paper
weapon model.

## Geometry and hit test

Both eligibility and hit sampling use the existing `engagement_geometry`:

```
ATA = wrap(line_of_sight - attacker.psi)
HA  = atan2(-dz, horizontal_distance)
eligible = attacker.alive and target.alive and ammo > 0
           and 0 <= distance <= 1000
           and abs(ATA) <= pi/6 and abs(HA) <= pi/6
epsilon ~ N(0, 1)                         # one draw per attempt
threshold = pi * exp(-distance / effective_hit_distance)
hit = abs(ATA + c4*epsilon) <= threshold
      and abs(HA + c5*epsilon) <= threshold
```

ATA is the horizontal LOS bearing relative to heading. HA is the LOS elevation
relative to the world horizontal plane in NED coordinates. Neither subtracts
attacker pitch. A target at HA=35 degrees with attacker pitch=20 degrees still
fails the HA gate, even though its body elevation is about 15 degrees.
AA and body-frame angles do not enter weapon qualification or hit sampling.
Angular boundaries retain the project's existing `angle_within` tolerance of
1e-12 radians; range boundaries retain the strict inclusive comparison.

`env/v336_weapon.py` contains the independent `PaperATAHAWeapon`.
`BodyFrameGaussianWeapon` is unchanged. Diagnostic `expected_hit_probability`
accepts `(distance, ata, ha)`, in radians, and intersects the two intervals for
the same Gaussian sample.

## Frozen project calibration and extensions

| Parameter or mechanism | Retained value |
|---|---|
| Range | 0–1000 m; the paper's 4000 m maximum is not restored |
| Ammo / cooldown | 6 per aircraft / 1 s |
| Effective hit distance | 558.1106265512473 m |
| c4 / c5 | 0.9984711359155588 / 0.9984711359155588 |
| Red automatic firing | Nearest eligible target; existing distance/index tie-breaking |
| Blue firing | Only the current step's nearest-surviving-Red pursuit target |
| Motion / observation | v3.32 simplified kinematics / 4 agents × 45D |
| Initialization | v3.34 staggered offsets, center radius 4000 m, unchanged reset semantics |
| Reward / outcome | Entire v3.35 implementation, including body geometry and continuous w3/w4 |

The numerical hit calibration, 1 km range, ammo, cooldown and Red tie-breaking
are project choices, not parameters established by the paper. Fire-state,
resolution order and all outcome semantics remain inherited. A changed
eligibility window can naturally change the selected eligible Red firing target,
shot timing, ammo consumption, RNG history, casualties and subsequent trajectory;
their controlling mechanisms are unchanged.

The environment config changes only the version and renames the equal-valued
angle limits from `boresight_azimuth_max/boresight_elevation_max` to
`ata_max/ha_max`. Validation reverses those names and invokes `validate_v335`,
rejecting any numerical/configuration drift. Algorithm configs change only
environment identity; checkpoint version and hash checks remain strict.

## Read-only audit

`tools/audit_combat_v336_weapon_geometry.py` compares both weapons on each
identical pre-fire snapshot and ammo array, without RNG draws or environment
mutation. It reports agreement, v335-only and v336-only eligible pair-steps,
and writes discrepancy rows with distance, ATA, HA, body angles and pitch.
Both teams' range, range+ATA, range+ATA+HA, attempts, hits and kills are reported.
The window denominator is live opposing pair-steps. Range/angle funnel counts
ignore ammo/cooldown; eligibility includes ammo and leaves cooldown to inherited
fire control. Historical Red agent-step diagnostic keys remain compatible:
`range_az` now counts ATA and `geometry` counts ATA+HA.

The audit uses seeds 0–19 for fixed-pursuit and an explicitly labeled Red pitch
probe per version, plus a controlled angle grid. Blue always uses its inherited
automatic policy. Probe actions are diagnostic inputs, not a policy/config
change. Independent version rollouts may diverge; comparisons are on identical
snapshots within each rollout. Grid counts describe the chosen grid, not the
frequency of opportunities in learned policies. Observer tests verify identical
states, actions, rewards, ammo, cooldown and RNG history with and without auditing.

Only unit/regression tests, environment smoke, geometry audits and 32-sample
CUDA algorithm smokes are run. No formal training is started.
