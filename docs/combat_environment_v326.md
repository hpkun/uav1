# Combat Environment v3.26

Direct baseline: CombatEnvironmentV325. The only behavioral change is:

**Blue weapon target is coupled to the per-step nearest pursuit target.**

```text
v3.25:
  movement_target = nearest live Red
  fire_target = nearest weapon-eligible Red

v3.26:
  movement_target = nearest live Red at step start
  fire_target = movement_target
  if that target is not eligible: no fire, including no off-target fire
```

## Step-start target identity

Before inherited step and dynamics, blue_step_target_indices stores int64[5].
Each live Blue uses fixed_policy.nearest_target_index on living Red, i.e. the
unchanged authoritative 3D nearest selector. Distance ties use the smaller index.
Dead Blue or no living target is represented by -1. reset clears all five slots.

The inherited automatic policy computes movement actions with the same selector
on the same unchanged pre-dynamics states. Its strategy, action encoder, speed
and target-switch diagnostics remain untouched. The wrapper then calls inherited
step and adds a copied blue_pursuit_target_indices array to info. Indices are
zero-based; the copy cannot change when a later step updates internal state.

The cached index lasts only for this simulation step; there is no lock,
persistence, hysteresis or FSM. A different nearest Red after dynamics does not
cause weapon reselection in that step. A cached target that dies from noncombat
loss before firing suppresses that Blue's attempt; the next step reselects.

For an externally supplied blue_actions override, movement is supplied by the
caller; the weapon restriction still uses the cached step-start nearest index.
The formal training/evaluation path uses automatic Blue actions.

## Firing

Red directly calls super()._entry_attempts with no change.
Blue decrements its unchanged attacker-wide cooldown, then checks only the cached
pair using the existing BodyFrameGaussianWeapon.qualifies. Dead/no target,
cooldown or an ineligible pursuit pair yields no shot, no ammo decrement, no new
cooldown and no hit RNG call. There is no eligible-target argmin in Blue firing.

An actual attempt decrements ammo, sets cooldown=10, and uses the inherited hit
sampler, for hit and miss alike. Target switches neither reset cooldown nor
restore ammo. Cadence remains S,S+10,...; six rounds remain the maximum.

Fire-window metrics and weapon eligibility remain their historical geometric
definition over all targets. Thus an off-target window can be recorded even
when no shot is allowed. The change is exclusively actual Blue target selection.

## Frozen components

Red firing unchanged; range 1km unchanged; azimuth/elevation ±30 degrees unchanged;
ammo=6 unchanged; cooldown=1s unchanged; Gaussian/shared-epsilon hit model and
noise unchanged. R1-R4, continuous w3/w4, signed 56D observations, controller,
Blue nearest pursuit/encoding and desired speed=250, dynamics and RK4 unchanged.
v3.25 symmetric boundary deaths are inherited; Safety, terminal and team casualty
rewards remain zero. No radar/sensor change is made.

Environment YAML differs from v3.25 only by version; MAPPO YAML is byte-identical.
Network, trainer and all training hyperparameters are unchanged. Checkpoint
evaluation/resume across v3.25/v3.26 is rejected by the existing version contract.
Recording continues to use dynamic environment dispatch.

## Validation

Run inside the uav Conda environment:

```text
python -m pytest tests/test_combat_v326.py tests/test_combat_v325.py tests/test_combat_v324.py tests/test_combat_v323.py tests/test_combat_protocol.py::test_factory_and_public_api_are_combat_only tests/test_combat_protocol.py::test_configuration_rejects_unsupported_task_and_observation_fields tests/test_training_protocol.py::test_checkpoint_version_is_saved_and_incompatible_semantics_fail_fast -q
```

197 passed: v3.26 18, v3.25 75, v3.24 37, v3.23 64, factory/schema/checkpoint 3.
Tests cover movement/fire target identity, v3.25 off-target contrast, post-motion
nearest changes, target deaths and next-step switching, hit/miss cooldown,
six-round cadence, authoritative 3D tie rules, no targets/dead Blue/reset,
unchanged Red firing, full transition identity when no off-target opportunity
exists, frozen configs, recording dispatch and checkpoint isolation.

Results are saved in outputs/v326_validation. Historical source/config hashes
are verified; only common version registration changes. No formal training runs.
