# Combat Environment v3.19

v3.19 directly inherits the current v3.17, retaining repeat fire. It does not
inherit v3.18 attacker-level armed/entry cadence. Only three formal distance
values and pair-level R4 weighting change:

| Parameter | v3.17/v3.18 | v3.19 |
|---|---:|---:|
| weapon.range_max | 800 m | 1000 m |
| reward.guide_distance_min | 4000 m | 3000 m |
| reward.tactical_distance_max | 4000 m | 3000 m |

R3 uses unchanged paper geometry and any-target aggregation: each live Red
receives at most +0.001 when any live Blue is at least 3000 m away with
paper ATA/HA <= 30 degrees. There is no sensor visibility gate.

R4 retains AA qualification, the three ATA/HA levels, and all coefficients.
Each offense and reverse-geometry defense pair is multiplied by:

```text
w(d) = 1                     if d <= 1000
       (3000-d)/2000         if 1000 < d < 3000
       0                     if d >= 3000
```

Only after weighting each pair is offense aggregated by max and defense by
min. R4 is their sum; dense tactical reward remains R3+R4. Signed negative
threat penalties decrease in magnitude with distance. Targets are not
preselected for reward, and rewards are never summed over enemies.

The total returned reward remains R1+R2+R3+weighted R4+Safety. R1/R2, terminal
coefficients and Safety (safe_distance=200, penalty_scale=0.2) are unchanged.
Weapon body azimuth/elevation gates, Eq.(8), one shared Gaussian, noise scales,
effective hit distance, ammo=6, nearest eligible target and simultaneous
resolution are inherited unchanged. Weapon range_min remains 0.

The arena radius is 5000 m; initial center radius is 4000 m and nominal
separation is 8000 m, with offsets [-400,-200,0,200,400]. Initial altitude,
speed, Blue pursuit/desired speed=250, 5v5, dt=0.1, max_steps=1000, 56D
Observation, action, target clipping, controller, point-mass dynamics, RK4,
termination and timeout semantics are unchanged. Paper geometry continues
to supply Observation and Reward; weapon axes remain body-frame quantities.

The v3.19 validator checks the three new values strictly, then maps only a
deepcopy to v3.17 for all other frozen checks. Historical validators and
implementations remain unchanged. The MAPPO config is byte-identical to
v3.17, including its formal 3M target and evaluation/checkpoint protocols.
Checkpoints must match their environment version.

Validation (inside uav):

```text
python -m pytest tests/test_combat_v319.py tests/test_combat_v317.py tests/test_combat_v318.py -q
```

No formal training is run as part of implementation or validation.
