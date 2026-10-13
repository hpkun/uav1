# Combat Environment v3.23

Direct baseline: CombatEnvironmentV322. Only Blue ground contact, dense reward
direction geometry and automatic firing cadence change. v3.22 and earlier
version files remain unchanged. No formal training is run.

## Blue ground contact

A live Blue with altitude <= arena.altitude_min dies before any return correction.
Its pose is not changed: there is no altitude reset or pitch reflection. The
death increments blue ground_losses, never Red attack_kills or kill reward.
The last Blue dying still yields the inherited red_win_elimination outcome,
with win_reward=0. Dead aircraft are excluded from subsequent movement, policy
targeting and weapons through the existing alive masks.

The inherited boundary resolver handles all Red behavior and surviving Blue
horizontal/ceiling returns exactly as before. Ground death takes precedence over
horizontal return. blue_returns.ground stays zero; blue_ground_losses is the
authoritative death diagnostic. Horizontal, ceiling and boundary return counters
remain available.

## Body-frame dense rewards

Reward uses the existing engagement_geometry for every forward and reverse pair:
distance, horizontal aa, boresight_azimuth_error, boresight_elevation_error.
Public geometry and weapon implementations are unchanged. All angle comparisons
use absolute signed angles and the existing inclusive boundary tolerance.

R3 requires live aircraft and both body angles <=30 degrees, weights the pair by
guide_reward*w3, then takes the maximum over Blue. It has no AA, sensor, ammo,
cooldown or weapon-eligibility gate. Each Red still gets at most 0.001 per step.

R4 retains horizontal |AA|<=tactical_aspect_max and tactical range. Direction
levels now require both body angles <=30/15/5 degrees, with the stricter level
overwriting the broader level. Coefficients are unchanged: offense .01/.02/.1;
threat -.015/-.025/-.15. Each pair is weighted before offense max and reverse
defense min aggregation.

v3.22 distance_weights is reused without modification:

```text
D_weapon = weapon.range_max
D_tactical = reward.tactical_distance_max
span = D_tactical-D_weapon
w3 = clip((distance-D_weapon)/span, 0, 1)
w4 = clip((D_tactical-distance)/span, 0, 1)
```

Weapon uses distance and body boresight. R3 is body boresight shaping. R4 adds
horizontal aspect advantage and distance weight; a legal weapon shot can still
have zero R4 when AA fails. Neither reward is cooldown/ammo gated.

The 3D regression uses attacker pitch=58 degrees, world LOS elevation=75 degrees,
body elevation=17 degrees, body azimuth=0, AA=0, distance=2000m. v3.22 guide and
offense are zero; v3.23 guide=.0005 and offense=.005. At 900m the same geometry
passes the actual weapon gate and gives offense=.01.

## Attacker-wide firing cooldown — CALIBRATION

The 1.0-second cooldown is a calibration choice, not a paper-specified parameter.
The only new formal config section is:

```yaml
fire_control:
  cooldown_seconds: 1.0
```

Weapon dataclass/config is unchanged. dt=.1 implies cooldown_steps=10; validation
requires exactly this field and formal value plus a positive finite dt and an
integral interval/dt ratio. All older fields delegate to frozen v3.22 validation.

Each team has a separate int64 vector of five remaining counters:
red_fire_cooldown_steps and blue_fire_cooldown_steps, reset to zero. At each side's
attempt hook the counters decrement to a minimum of zero. An actual attempt
consumes one round and sets that attacker's counter to ten, for both hit and miss.
After firing at S, S+1..S+9 cannot fire, S+10 can. Switching targets does not reset
the timer. Other attackers and the opposite team remain independent. There is no
reload or new firing action; ammo remains six per aircraft.

Weapon eligibility and fire-window metrics are computed independently before the
attempt hook and do not include cooldown. During cooldown an aircraft can have
fire_window=true but attempt=false. Available attackers still choose the nearest
eligible target by 3D distance, with the historical index tie-break. Gaussian hit
sampling, shared epsilon, scales and effective_hit_distance are unchanged.

## Frozen behavior and validation

Controller, Blue pursuit/encoder, observations, initialization, point-mass
dynamics, RK4, R1/R2, Safety and outcome criteria remain inherited. All terminal
reward fields and team casualty penalty stay zero. MAPPO YAML is byte-identical
to v3.22. Checkpoints are separated by the existing version/config contract.
Recording uses the existing dynamic dispatch.

Run in the uav Conda environment:

```text
python -m pytest tests/test_combat_v323.py tests/test_combat_v322.py tests/test_combat_v321.py tests/test_combat_v320.py tests/test_combat_protocol.py::test_factory_and_public_api_are_combat_only -q
```

Validation artifacts are saved under outputs/v323_validation. No formal training
or trained-model evaluation is required for these deterministic contract tests.
