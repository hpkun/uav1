# Combat environment v2.6

v2.6 is an isolated 8v8 environment (104 observations, 3 actions per aircraft).
Its scenario, aircraft, dynamics, timestep, arena, observations, rewards,
weapons, PairFireState, simultaneous resolution and termination are identical
to v2.4. The default remains v2.3; v2.3/v2.4/v2.5 keep nearest-target pursuit.

Only the fixed Blue heading strategy changes. Each Blue independently acquires
the nearest living Red by 3-D range and locks it until death. Reacquisition
clears phase, stored bearing and turn side; the acquisition decision uses pure
pursuit, and subsequent decisions evaluate one phase transition each. Episode
reset clears all states. Explicit external Blue actions bypass the policy.

1. PURE_PURSUIT: target LOS heading; enter RELATIVE_BEARING at range <=6500 m.
2. RELATIVE_BEARING: store entry heading plus a deterministic signed 15 degrees
   once; maintain this bearing until absolute target-frame lateral displacement
   reaches 600 m. For nonzero lateral, compare only the two entry-heading
   candidates using sign(lateral) * sin(candidate - target heading), and choose
   the larger outward-motion score. Within 1e-6 m of zero, or on an exact score
   tie, even/odd Blue index selects +1/-1.
3. OFFSET_RECIPROCAL: command current target heading plus pi, until target-frame
   longitudinal <=0 and 3-D range <=2500 m.
4. CONVERT: target LOS pursuit, retained for the life of this locked target.

All phases command 250 m/s and the existing altitude controller. No threat
awareness, evasion, coordination, speed matching or policy RNG is introduced.
Weapon qualification remains exclusively in the existing weapon system:
range bounds, true 3-D off-boresight <=30 degrees and target aspect <=45 degrees.

The four formal PPO algorithms reuse their existing 8v8 configs. RMAPPO/EA-MAPPO
version allow-lists also accept 2.6; MAPPO/STEA already use validated instance
dimensions. Checkpoint environment version and config SHA must still match:
a v2.4 checkpoint is not relabeled as a v2.6 checkpoint. MADSAC and MADDPG
environment support are unchanged.

## Bounded acceptance

Run all commands in WSL Ubuntu after `conda activate uav`:

```bash
python tools/validate_stern_conversion.py --trials 200 --seed-base 260000 \
  --output outputs/v26_validation/geometry.json
python tools/smoke_formal_8v8.py --env-config configs/combat_environment_v26.yaml \
  --output outputs/v26_validation/cuda_smoke
python -m pytest -q
python -m compileall -q algorithm env tools tests
```

Geometric validation uses the real point-mass dynamics, RK4, action controller
and weapon envelope, with a straight constant-speed target at the existing
8000 m opposing-center distance and existing altitude/speed/heading
perturbations. It loads no checkpoints and stops at the first arena/ground
loss or the unchanged 1000-step limit. The report records both loss ownership
and whether loss preceded the first legal fire window. Straight targets can
leave the arena after a successful conversion; this is reported separately
from conversion failure. First-window percentiles include successful trials.

The CUDA smoke caps each algorithm at 512 sampled steps while preserving the
formal networks, PPO settings, environment and 3M config defaults, then checks
two deterministic episodes after strict checkpoint reload. It is not a formal
training run or a performance comparison. Fixed Stern parameters are not tuned.

## Initial acceptance results (before turn-side correction)

- Full pytest: 654 passed in 156.12 s; dedicated v2.6 tests: 39 passed.
- Final compileall: passed.
- 200 trials, seeds 260000..260199: fire-window success 100%, four-phase
  completion 100%, median first window step 277, p90 291.
- Blue arena exits 0; ground losses 0; losses before first fire window 0.
  All 200 straight targets eventually exited after successful conversion.
- Synthetic 1v1: relative bearing step 33; reciprocal step 146 (605.38 m
  lateral); convert step 172 (longitudinal -42.16 m); first legal window
  step 270. Blue did not exit; target exited at step 401.
- MAPPO, RMAPPO, EA-MAPPO and STEA-MAPPO: each 512 sampled steps, 16 spawn
  workers, one real PPO update, finite gradients/parameters/values, two
  deterministic episodes after checkpoint reload. RTX 5060 Laptop GPU.
- No formal 3M training was started. No fixed parameters were tuned.

Reports: `outputs/v26_validation/geometry.json`,
`outputs/v26_validation/cuda_smoke/summary.json`,
`outputs/v26_validation/pytest.log`, and
`outputs/v26_validation/compileall.log`.

## Changed files

New: `configs/combat_environment_v26.yaml`, `tests/test_combat_v26.py`,
`tools/validate_stern_conversion.py`, `docs/combat_environment_v26.md`.

Modified: `env/fixed_policy.py`, `env/config.py`, `env/combat_env.py`,
`algorithm/rmappo/runner.py`, `algorithm/rmappo/protocol.py`,
`algorithm/ea_mappo/runner.py`, `algorithm/ea_mappo/protocol.py`,
`tests/test_combat_protocol.py`, `tests/test_formal_8v8.py`,
`tools/smoke_formal_8v8.py`.

The existing v2.3/v2.4/v2.5 YAML files and all formal algorithm YAML files were
unchanged. The two existing tests only update config inventory and the supported
version error message. The smoke tool adds an optional environment config;
its default remains v2.4. Old fixed-seed regression tests passed in full pytest.

## Minimal turn-side correction validation

Only the relative-bearing entry candidate comparison changed; no state or
parameter was added. Tests cover rotated and nearly head-on geometry, outward
motion under the real dynamics, near-zero lateral parity and exact score ties.
Dedicated tests: 61 passed; full pytest: 676 passed in 132.08 s; compileall passed.
The same 200 seeds retain 100% fire-window success and phase completion, with
median/p90 first window steps 268/271. Blue arena exits, ground losses and
boundary-before-window failures are all zero. Parameters are unchanged; no
training was started. Reports are under `outputs/v26_turn_side_validation/`.
