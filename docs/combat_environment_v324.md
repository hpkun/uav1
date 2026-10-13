# Combat Environment v3.24

v3.24 is an observation geometry CORRECTION / paper-consistent reconstruction.
It directly inherits v3.23 and overrides only _observations besides the version.
The change restores signed relative direction information to Red actor inputs.

The observation remains float64 with shape (5,56); the network's existing tensor
conversion is unchanged. Layout:

|Indices|Content|Change|
|---|---|---|
|0:7|self x,y,altitude,speed,reserved zero,psi,theta|none|
|7:31|four allies, six features each|none|
|31:56|five enemies, five features each|signed geometry correction|

Self normalization and ally flight_path_frame/_relative_position, speed, heading,
pitch and fixed aircraft ordering are preserved exactly. Dead self rows and dead
ally slots remain zero. Blue j stays in enemy slot j even when other Blue die.

For each live enemy, engagement_geometry(own,enemy) supplies all geometry:

```text
[distance / relative_position_scale,
 (enemy.v-speed_center) / speed_scale,
 signed AA / pi,
 signed body azimuth / pi,
 signed body elevation / (pi/2)]
```

The raw fields are distance, aa, boresight_azimuth_error and
boresight_elevation_error. There is no abs, angular_difference, old paper ATA/HA
conversion or sign flip in the enemy encoding. The signs follow the existing
engagement_geometry conventions. Each dead enemy's entire five-feature slot is
zero. Global enemy information remains available without sensor masking.

At own heading/pitch zero, target horizontal positions (1000,+1000) and
(1000,-1000) produce body azimuth +45/-45 degrees, normalized +.25/-.25. v3.23
produces identical whole observation rows for this fixture; v3.24 distinguishes
them. Targets at x=1000 and altitude offsets +500/-500 produce elevation
+26.565051/-26.565051 degrees, normalized +.295167235/-.295167235. v3.23 again
aliases the mirror fixtures. A target heading at LOS +/-20 degrees produces
signed AA normalized +.111111111/-.111111111.

Observation preserves direction for decisions. Unchanged v3.23 reward evaluates
absolute body-angle magnitudes and horizontal AA. The unchanged weapon uses the
same signed raw body angles for its threshold/hit calculation. No ammo, cooldown,
fire-ready, history, radar or visibility features are added.

Equation (24)'s relative/body-reference observation idea motivates this
reconstruction. The specific signed fields and normalization here are this
implementation's correction; they are not asserted to be the unique encoding
mandated by published paper code.

Environment YAML differs from v3.23 only by version, including the unchanged
1.0-second firing cooldown. MAPPO YAML is byte-identical. Controller, Blue policy,
body rewards, weapons, Blue ground death, R1/R2/Safety/zero terminal reward,
physics, RK4, termination and all training hyperparameters remain unchanged.
With identical external Red actions and RNG, transitions are identical; actor
decisions may of course differ because observations now retain direction.

Although dimensions stay 56/3/5, v3.23 checkpoint evaluation/resume against v3.24
is rejected by the existing environment-version/config identity contract.
Recording continues to use existing dynamic dispatch.

Validation in the uav Conda environment:

```text
python -m pytest tests/test_combat_v324.py tests/test_combat_v323.py tests/test_combat_protocol.py::test_factory_and_public_api_are_combat_only -q
```

Tests include old-alias reproduction, signed mirror separation, exact general 3D
geometry, absolute reward symmetry, weapon agreement, unchanged self/ally blocks,
fixed dead slots, no masking or ammo/cooldown features, natural fixed-action
trajectories, firing/cooldown and ground/boundary termination regressions, config
freezing and checkpoint isolation. No formal training is run.
