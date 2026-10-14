# Combat Environment v3.35

v3.35 directly inherits `CombatEnvironmentV334` and changes only the configured
hit-distance scale. The environment class overrides no method.

## Complete environment configuration difference

| Field | v3.34 | v3.35 |
|---|---:|---:|
| environment_version | 3.34 | 3.35 |
| weapon.effective_hit_distance (m) | 2232.442506204989 | 558.1106265512473 |

The new value is exactly the Python float `1000 / ln(6)`. The validator checks
this exact value, reverses these two differences in a deepcopy, and calls the
unchanged `validate_v334`. All other fields are identical.

## Recalibration meaning

The retained value equals `4000 / ln(6)`, associated with the previous 4 km
scale. The weapon envelope is currently 1 km. v3.35 preserves the normalized
ratio `D_hit / D_fire_max = 1 / ln(6)` by scaling D_hit by one quarter.
This is a project distance-scale recalibration, not a new paper-derived hidden
parameter or a simultaneous angular-noise calibration.

The unchanged `BodyFrameGaussianWeapon` uses
`threshold(d)=pi*exp(-d/D_hit)` and one shared Gaussian epsilon for the two
absolute-value body azimuth/elevation tests. Both noise scales remain
0.9984711359155588. No probability lookup or alternate hit formula is introduced.
At normalized distances rho=0/.25/.5/.75/1, the old 4 km scale and new 1 km scale
have exactly equal thresholds and expected probabilities. This threshold
comparison does not make the v3.34 weapon eligible at 4 km.

## Frozen semantics

The v3.34 point-symmetric staggered formation, center radius 4000 m, 45D
observations, sensor, outcome and nearest-surviving-target Blue policy remain
inherited. Motion remains the v3.32 direct per-step yaw/pitch/speed changes.
Reward remains the v3.32 continuous w3/w4 implementation and AA=30 degrees.
Weapon eligibility is unchanged: 0-1000 m, body azimuth/elevation within 30
degrees, without an AA gate. Ammo 6, cooldown 1 s, target selection, repeated
firing and simultaneous resolution remain unchanged. No actor or learning
hyperparameter is changed; algorithm configurations update only environment
identity. Old checkpoint version/config hashes remain incompatible.

## Diagnostic probabilities and tests

| Perfect alignment distance | v3.34 expected hit probability | v3.35 expected hit probability |
|---|---:|---:|
| 0 m | 0.998347079640 | 0.998347079640 |
| 250 m | 0.995092840535 | 0.955608226365 |
| 500 m | 0.988098220579 | 0.801037718828 |
| 750 m | 0.975461645366 | 0.588200580081 |
| 1000 m | 0.955608226365 | 0.400000000000 |

Values are computed by the existing `expected_hit_probability` diagnostic,
not hard-coded into environment logic. Unit tests cover these values, normalized
threshold/probability invariance (including nonzero angles), one shared epsilon,
unchanged eligibility, strict config isolation, identical 200-seed initialization,
frozen reward/motion without hit draws, and checkpoint version isolation.

`tools/audit_combat_v335_hit_probability.py` calls the existing `attempt_hit`
for every Monte Carlo sample. Sample count and seed are CLI parameters; default
sample count is 1,000,000 per case, seed 335, with body-angle pairs (0,0),
(15,15), (30,30), (30,-30) degrees and distances 0/250/500/750/1000 m in both
versions. These are diagnostic sampling settings, not environment parameters.
The opposite-sign angle case checks the shared-epsilon interval intersection.
The six-standard-error field is an audit flag, not a reward or weapon threshold.

The three existing algorithm runners receive minimal 32-transition CUDA smokes,
followed by official checkpoint reload and evaluation. All commands use the
PowerShell uav Conda environment. No formal training is run. Measured Monte
Carlo, smoke and regression results are in `outputs/v335_validation/report.md`.
