# Combat Environment v3.18

v3.18 inherits v3.17. Only fire cadence changes, via _new_fire_states and
_entry_attempts; step and reset remain inherited. Weapon/config/flight/paper
Observation/Reward/scenario/Blue/action/termination semantics remain v3.17.
Environment configuration differs only by environment_version.
Weapon range is [0, 800] m, matching v3.17; reward distance parameters are unchanged.
Initial center radius is 4000 m and nominal anchor separation is 8000 m,
matching v3.17. Offsets remain [-400, -200, 0, 200, 400]; arena radius remains 5000 m.

Each team has five independent existing FireState objects, one per attacker.
Reset creates all ten with armed=True. No pair-level state or timers exist.

For each attacker every step:

```text
no eligible target: armed=True, no attempt
eligible targets and armed and ammo>0:
    select nearest (distance, target index)
    attempt once; ammo-=1; armed=False
eligible targets and not armed:
    no attempt, no ammo use, no RNG draw
```

A kill or a different nearest target does not rearm while another eligible
target exists. Only a step with an entirely empty eligible set rearms. Leaving
the window for one step and reentering on the next may fire immediately.
There is no cooldown, reload, hysteresis, lock delay or missile-flight model.
Both teams use exactly the same cadence; simultaneous kill resolution remains.

Core regression: twenty continuous eligible steps produce one attempt, use one
round and draw one shared Gaussian. v3.17 still repeats until its six rounds
are exhausted. Version-tagged checkpoints are never interchangeable.

Validation order (uav environment, CUDA for checkpoint audits):

```text
python -m pytest tests/test_combat_v317.py -q
python -m pytest tests/test_combat_v318.py -q
python -m pytest -q
python tools/audit_combat_v317_v318.py
python tools/validate_combat_v317_v318.py
python -m compileall -q env algorithm tools tests
```

The interface validator initializes formal-width CUDA runners and saves/loads
untrained checkpoints at sampled_steps=0; it never calls train or update. It
also verifies real vector auto-reset. No 500k/3M training or old-policy replay
is performed. Detailed results are in outputs/v317_v318_validation/report.md.
