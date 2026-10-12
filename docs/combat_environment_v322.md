# Combat Environment v3.22

v3.22 directly inherits CombatEnvironmentV321 and only overrides
dense_combat_reward (besides the version). v3.20 feasible control and v3.21
Blue encoding, reset, motion, observations, weapons, combat and termination
remain inherited. No training is run.

Distance scales are read from existing formal configuration:

```text
D_weapon = weapon.range_max
D_tactical = reward.tactical_distance_max
span = D_tactical - D_weapon
w3(d) = clip((d-D_weapon)/span, 0, 1)
w4(d) = clip((D_tactical-d)/span, 0, 1)
```

The pure function rejects D_tactical <= D_weapon. There are no hardcoded
distance scales in the new reward implementation and no new formal fields.
The formal values remain 1000m and 3000m. guide_distance_min remains frozen at
3000m in the schema but is no longer a hard R3 distance gate.

For each live Red-Blue pair, R3 eligibility remains |ATA|<=pi/6 and |HA|<=pi/6.
There is no AA, visibility, ammo or weapon-eligibility gate. Eligible pairs get
guide_reward*w3(distance); each Red receives the maximum across Blue targets,
never a sum. R3 therefore remains within [0,0.001] per Red per step.

R4 retains all original forward/reverse geometry gates, angle levels,
coefficients and signs. Each pair is multiplied by w4 before offense max and
defense min aggregation. At formal scales R4 is numerically identical to v3.21.
The result structure remains guide, tactical_offense, tactical_defense,
tactical=guide+offense+defense. Total reward remains R1+R2+R3+R4+Safety.

|Distance m|w3|w4|Eligible R3|
|---:|---:|---:|---:|
|1000|0|1|0|
|1500|0.25|0.75|0.00025|
|2000|0.5|0.5|0.0005|
|2500|0.75|0.25|0.00075|
|2999|0.9995|0.0005|0.0009995|
|3000|1|0|0.001|
|3001|1|0|0.001|

For the best offense angle level at 2999m, offense is 0.00005; at 3000m it is
zero. The handoff is continuous in distance for fixed angle/live eligibility.
Existing angle/live gates retain their original semantics.

Environment YAML differs from v3.21 only by version; MAPPO YAML is byte-identical.
v3.21 checkpoint evaluation/resume against v3.22 is rejected by the existing
version contract. Recording uses the existing dynamic dispatch.

Validation in the uav Conda environment:

```text
python -m pytest tests/test_combat_v322.py tests/test_combat_v321.py tests/test_combat_v319.py tests/test_combat_protocol.py::test_factory_and_public_api_are_combat_only -q
```

245 passed: v3.22 103, v3.21 56, v3.19 85, registration 1. Validation includes
artificial distance scales, formal R4 exact regression, live/angle and max/min
aggregation, same-seed motion, reward integration, weapon sampling/repeat fire,
termination and checkpoint isolation. Details are in outputs/v322_validation.
