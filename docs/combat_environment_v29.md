# Combat environment v2.9: delayed Stern entry

v2.9 is v2.7's single-variable delayed-Stern experiment. The only substantive
change is `blue_policy.turn_range`: 6500 m to 5500 m. The other config change
is `environment_version`: 2.7 to 2.9. It is built from v2.7, not v2.8;
conversion_range remains 2500 m. The 5v5 formation and 65/3/5 dimensions,
15-degree entry turn, 600 m lateral displacement, 250 m/s, simulation, actions,
aircraft, arena, weapon, rewards and observations are identical to v2.7.

Blue must be closer before switching from PURE_PURSUIT to RELATIVE_BEARING,
reducing the time and space margin for Stern conversion. The experiment is used
to evaluate task difficulty after delaying the fixed Stern opponent's maneuver
start, not to target a MAPPO win rate. Later entry does not guarantee that every
subsequent phase or fire window is later; the unchanged dynamics determine that.

SternConversionPolicy and reward source are unchanged. No phase, tactic,
randomness, speed matching, cooldown or adaptive parameter is introduced.
Target locks, reacquisition, outward turn-side comparison, parity ties, fixed
entry bearing, reciprocal heading, Convert LOS, altitude controller, reset,
external Blue override and AutoFire remain as before. Old configs and the
default v2.3 environment are unchanged.

v2.9 shares Stern schema validation, RearAspectWeaponEnvelope and 5x5
PairFireState. Existing four formal 5v5 algorithm YAMLs are reused unchanged.
RMAPPO/EA-MAPPO version allow-lists add 2.9; MAPPO/STEA retain their existing
dimension and checkpoint fingerprint checks. 8v8 configs remain incompatible.
MADSAC and MADDPG environment support does not expand.

## Files

New: `configs/combat_environment_v29.yaml`, `tests/test_combat_v29.py`,
`docs/combat_environment_v29.md`.

Modified: `env/config.py`, `env/combat_env.py`,
`algorithm/rmappo/runner.py`, `algorithm/rmappo/protocol.py`,
`algorithm/ea_mappo/runner.py`, `algorithm/ea_mappo/protocol.py`,
`tests/test_combat_protocol.py`, `tests/test_formal_8v8.py`,
`tools/validate_stern_conversion.py`.

The geometry tool only adds version support and mean/median RELATIVE_BEARING
entry statistics from already recorded transitions. The trial algorithm and
existing smoke tool are unchanged.

## Acceptance

Run in WSL Ubuntu after `conda activate uav`:

```bash
python tools/validate_stern_conversion.py \
  --env-config configs/combat_environment_v29.yaml --trials 200 --seed-base 260000 \
  --output outputs/v29_validation/geometry_v29.json
python tools/smoke_formal_8v8.py \
  --env-config configs/combat_environment_v29.yaml \
  --output outputs/v29_validation/cuda_smoke
python -m pytest -q
python -m compileall -q algorithm env tools tests
```

The synthetic test distinguishes 6000 m entry and verifies the exact 5500 m
boundary using true 3-D distance, not horizontal range. A subsequent transition
test verifies unchanged 600 m displacement, reciprocal heading and conversion
at a rear-hemisphere range between 2000 and 2500 m. Tests also preserve reset,
outward motion, explicit override, rewards/termination and algorithm compatibility.

200-trial v2.9 results: success and phase completion 100%; first-window
median/p90 269/273 steps; RB-entry mean/median 54.155/54 steps; CONVERT-entry
mean/median 170.515/171 steps. Blue exits, ground losses and boundary-before-window
failures are zero. All 200 straight targets exit after successful conversion.

The same-seed comparison reuses the saved v2.7 report from unchanged source and
config. All 200 RB entries occur 20..22 steps later (mean 21.2, median 21).
CONVERT paired gap mean is +0.065 steps, median 0: 15 later, 183 unchanged,
2 earlier. First-window paired gap mean is +1.08, median +1: 125 later,
61 unchanged, 14 earlier. Stern entry is delayed, while final conversion
capability changes little in this 1v1 block. This is not an RL performance audit.

Reports, per-trial comparisons and frozen-source SHA checks are in
`outputs/v29_validation/`. CUDA smoke is limited to 512 sampled steps per
algorithm and two evaluation episodes after checkpoint reload. No formal 3M
training, parameter tuning or parameter sweep is performed.

Final acceptance: 104 targeted tests passed (15 new v2.9 cases); full pytest
719 passed in 120.54 s; compileall passed. All four CUDA smokes passed at
512 sampled steps with 16 spawn workers, one real PPO update and two episodes
after strict checkpoint reload. Gradients/parameters/values were finite and
recurrent pre-update ratios passed. SHA checks confirmed policy and reward
source, all v2.3-v2.8 environment YAMLs and four formal 5v5 YAMLs unchanged.
