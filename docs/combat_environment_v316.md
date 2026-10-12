# Combat Environment v3.16

v3.16 inherits v3.15 and changes only weapon qualification and Equation (8).
It does not introduce locking delay, missile flight, guidance, cooldown,
action smoothing or a new controller.

Qualification retains the paper conditions: both aircraft alive, ammo > 0,
distance in [0, 4000] m, horizontal ATA <= 30 degrees and paper HA <= 30 degrees.
Paper HA remains absolute LOS elevation; own pitch is not subtracted. The
additional condition is true three-dimensional off-boresight <= 30 degrees,
using the existing `engagement_geometry` and its current-pitch/current-heading
forward vector. `weapon.boresight_max = pi/6` is the only new config field.
Both teams use the same weapon. Nearest-qualified-target selection, ammo=6,
one attempt per aircraft per step and repeat fire every eligible step remain.

The v3.15 case own pitch +60 degrees / LOS elevation -22 degrees has true
off-boresight about 82 degrees. It remains legal in v3.15 and is rejected in
v3.16 only by the new boresight condition. Own pitch +18 / LOS elevation +22
has off-boresight about 4 degrees and remains legal. Paper gates still apply
even when the true nose is aligned.

Equation (8) uses one shared epsilon ~ N(0, 1) per attempt:

```text
T = pi * exp(-distance / effective_hit_distance)
hit = abs(ATA + ata_noise_scale * epsilon) <= T
      and abs(HA + ha_noise_scale * epsilon) <= T
```

Its inputs remain paper ATA/HA/distance; off-boresight is not an Eq.(8) input.
The analytic diagnostic intersects the two valid epsilon intervals and
returns Phi(upper)-Phi(lower), or zero for an empty intersection. It is never
used in sampling. Effective hit distance and both noise scales are unchanged.
Correcting the formula changes statistical hit probabilities; no parameters
have been recalibrated to preserve the old erroneous probabilities.

Observation remains exactly v3.15: 56D, self index4=0 and paper HA slots.
R1-R4/Safety and reward parameters, scenarios, Blue pursuit policy, feasible
targets, the 2s/2s/2s/8g controller, PointMassDynamics and RK4 are unchanged.
Changed weapons can produce different kills and thus different realized
episode rewards; this does not change the reward formulas.

CombatEnvironmentV316 only overrides the version and build_weapon. The
inherited attempt_hit_for_pair already passes authoritative paper geometry
to self.weapon, so no duplicate hit-input path is necessary.
Historical v3.13-v3.15 weapons and trajectories remain isolated. Checkpoints
must match environment version even when network shapes are equal.

Run validation inside the uav Conda environment with CUDA:

```text
python -m pytest tests/test_combat_v316.py -q
python tools/audit_combat_v316_weapon.py
python tools/smoke_formal_8v8.py --env-config configs/combat_environment_v316.yaml --output outputs/v316_validation/cuda_smoke
python tools/validate_combat_v316.py
python -m pytest -q
python -m compileall -q env algorithm tools tests
```

The four formal-width smoke runs cap sampling at 512 transitions each and
save/reload before two deterministic episodes. They do not start formal
500k/3M training. Results, static Monte Carlo diagnostics and existing test
failures are recorded in outputs/v316_validation/report.md.
