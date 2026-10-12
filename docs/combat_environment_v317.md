# Combat Environment v3.17

v3.17 isolates weapon geometry from paper geometry. It inherits v3.16 and
overrides the version, build_weapon, attempt_hit_for_pair, and a thin reset
wrapper that reports nominal anchors using the configured center radius.

Weapon qualification requires both aircraft alive, ammo > 0, distance in
[0, 800] m, abs(body azimuth) <= pi/6 and abs(body elevation) <= pi/6.
Both signed axes come from existing engagement_geometry. Paper ATA/HA are
not weapon gates; off_boresight remains diagnostic and is not a cone gate.
The rectangular window allows body azimuth=25 degrees/elevation=25 degrees
even when off_boresight exceeds 30 degrees. Pitch=45 degrees with a target
on the true forward ray is legal; the historical paper HA=45 degrees remains.

Formal weapon fields are boresight_azimuth_max, boresight_elevation_max,
azimuth_noise_scale and elevation_noise_scale. Axis limits remain pi/6,
noise scales remain 0.9984711359155588, effective_hit_distance remains
2232.442506204989, ammo remains 6. Validators internally map to the frozen
v3.16 numeric contract except range_max, which is strictly 800 m in v3.17/v3.18.
The historical v3.16 range remains strictly 4000 m; range_min stays 0 m.
Old field names are rejected in the public schema. Reward distance parameters
remain unchanged at their historical 4000 m values.

The actual step path calls engagement_geometry and passes its distance,
signed boresight_azimuth_error and signed boresight_elevation_error to:

```text
epsilon = rng.normal(0, 1)   # exactly once per attempt
T = pi * exp(-distance / effective_hit_distance)
abs(body_azimuth + azimuth_noise_scale * epsilon) <= T
and abs(body_elevation + elevation_noise_scale * epsilon) <= T
```

The analytic diagnostic intersects the two shared-epsilon intervals; it is
never used for sampling. No hit parameter is recalibrated.

Paper ATA/AA/HA geometry remains authoritative for Observation and Reward.
56D ordering and slot4=0, R1-R4/Safety, Blue policy, dynamics, RK4,
desired target clipping, controller, action space and termination are inherited
unchanged. Realized combat outcomes may change without changing reward formulas.
Initial formation center radius is 4000 m (nominal separation 8000 m), with
unchanged offsets [-400, -200, 0, 200, 400]. Arena radius remains 5000 m;
initial altitude and speed are unchanged. v3.16 retains radius 4800 m and
nominal separation 9600 m. Only this scenario field differs from v3.16.
Repeat fire, nearest eligible target and simultaneous resolution remain v3.16.
Historical v3.16 and earlier implementations/configurations remain isolated.

Validate in the uav environment:

```text
python -m pytest tests/test_combat_v317.py -q
python tools/audit_combat_v317_v318.py
```

No formal training is started by these checks. Full results are in
outputs/v317_v318_validation/report.md.
