# Combat Environment v3.37

**v3.37 restores the paper ATA/HA/AA reward angular geometry while retaining the
project's current distance shaping, 1 km weapon range, aggregation and
fire-control calibration.** It does not reproduce Eq.(25) exactly.

`CombatEnvironmentV337(CombatEnvironmentV336)` overrides only
`dense_combat_reward()`, using `env/v337_reward.py`. Weapon, transition order,
kinematics, observations, initialization, Blue policy and outcome are inherited.

## Authoritative angular geometry

Reward and the inherited v3.36 Weapon use `geometry.engagement_geometry`:

```
ATA = wrap(line_of_sight - attacker.psi)
HA  = atan2(-dz, horizontal_distance)
AA  = wrap(target.psi - line_of_sight)
```

All reward comparisons use absolute angles with the existing 1e-12 radian
roundoff tolerance. HA is relative to world horizontal and does not subtract
attacker pitch. **Do not use `v36_geometry.combat_geometry().ha` for this
version:** that historical helper subtracts attacker pitch from LOS elevation.
Historical helpers and versions are unchanged.

## Paper angular relationships and retained project shaping

Let `L(ATA,HA)` be the strictest satisfied 30/15/5 degree tier, with the
unchanged offense coefficients +0.01/+0.02/+0.1 and threat coefficients
−0.015/−0.025/−0.15. Unsatisfied tiers give zero. Both aircraft must be alive.

```
w3 = clip((d - 1000) / 2000, 0, 1)
w4 = clip((3000 - d) / 2000, 0, 1)
guide_pair = 0.001*w3 if abs(ATA_r), abs(HA_r) <= 30 deg else 0
offense_pair = L_offense(ATA_r,HA_r)*w4
               if d<=3000 and abs(AA_r)<=30 deg else 0
threat_pair = L_threat(ATA_b,HA_b)*reverse_w4
              if reverse_d<=3000 and abs(AA_b)<=30 deg else 0
R3_i = max_j guide_pair[i,j]
R4_offense_i = max_j offense_pair[i,j]
R4_defense_i = min_j threat_pair[j,i]
tactical_i = R3_i + R4_offense_i + R4_defense_i
```

The reverse geometry is explicitly recomputed with Blue as attacker and Red
as target; it is not the negation of a Red-centered pair. R3 has no AA gate.
R4's aspect gate remains 30 degrees. At 3000 m w4 is zero; at 3500 m aligned
R3 is still +0.001. No 4000 m hard segmentation is introduced.

Paper: Red-centered ATA/HA for R3; Red-centered AA gate plus ATA/HA tiers for
offense; Blue-centered AA gate plus ATA/HA tiers for defense.

Project calibration/extensions: 1 km Weapon, 1–3 km continuous w3/w4 handoff,
3 km tactical maximum, multi-target max/max/min aggregation, six rounds,
one-second cooldown, Dhit=558.1106265512473 and c4=c5=0.9984711359155588.
These numerical hit parameters are not claimed as published paper values.

R1 kill/loss +10/−10, R2 boundary −10, disabled terminal/safety/potential
rewards and all other coefficients are frozen. `guide_distance_min=3000`
remains in the configuration; it is not converted into a hard gate or cleaned up.

## Configuration, audits and validation

`combat_environment_v337.yaml` differs from v3.36 only in the version string.
Its validator maps that identity back and calls `validate_v336`. The three
algorithm configurations retain every formal hyperparameter and change only
environment identity where present. Historical checkpoints remain rejected by
strict environment version/hash checks.

`tools/audit_combat_v337_reward_geometry.py` computes both rewards on identical
authoritative pre-fire states. It separately reports pair and agent agreement,
changed counts, signed/absolute deltas and total tactical differences. CSV
discrepancies include distance, ATA/HA/AA, body angles, attacker pitch and each
version's pair reward; defense rows use Blue-centered geometry. Pair rewards
are distinct from the resulting max/min agent aggregation.

The audit uses fixed-pursuit and a deterministic Red pitch probe, seeds 0–9,
plus a 1512-case controlled geometry grid. The probe is an audit action input;
Blue's automatic policy is unchanged. Paired environments receive the same
actions and RNG seed. Every step asserts identical states, observations,
termination, ammo, cooldown, weapon RNG, combat counters and R1/R2/disabled
reward components. Only dense reward values and associated diagnostics differ.
The grid describes chosen geometries, not learned-policy opportunity frequency.

Validation includes 200 identical resets, boundary/pitch/geometry tests,
historical selected regressions, read-only A/B and weapon zero-difference
audits, CPU environment reset/step and 32-sample CUDA smokes for all algorithms.
No formal training is launched. Detailed results are in
`outputs/v337_validation/report.md`.
