# Combat Environment v3.25

Direct baseline: CombatEnvironmentV324. There are only two behavioral changes:
symmetric noncombat boundary deaths and safety shaping disabled. The class only
overrides _resolve_noncombat_losses besides the version; step is fully inherited.

## Symmetric noncombat boundary deaths

Both Red and Blue use the existing symmetric CombatEnvironmentV30 resolver,
whose Red rules exactly match the v3.24 Red resolver. For every living aircraft:

```text
if altitude <= altitude_min:
    alive = False
    ground_losses += 1
elif horizontal_radius > arena.radius or altitude > altitude_max:
    alive = False
    boundary_exits += 1
    if altitude > altitude_max:
        ceiling_losses += 1
```

Ground takes precedence, so a ground+horizontal violation is only ground loss.
Ceiling is a boundary-loss subtype. Exact horizontal and ceiling bounds remain
inclusive: radius=5000 and altitude=6000 do not kill an aircraft; altitude=0 does.
Already dead aircraft are skipped and do not accumulate duplicate losses.

No pose correction is performed for either team: no projection, heading reversal,
altitude reset or pitch reflection. Blue return diagnostic fields are retained
for logging compatibility, but boundary/horizontal/ground/ceiling returns stay
zero. Authoritative losses are blue_boundary_exits, blue_ground_losses and
blue_ceiling_losses.

Blue noncombat loss is not a Red weapon kill. It gives no Red kill credit, weapon
hit, R1 kill reward or R2 boundary reward. The last Blue's noncombat death naturally
causes the inherited elimination outcome, with zero terminal reward. Red ground
loss remains R1=-10 and Red horizontal/ceiling exit remains R2=-10. Actual weapon
kills retain the original +10/-10 accounting.

## Safety shaping disabled

The only config changes from v3.24 are environment_version='3.25' and
reward.safety_penalty_scale=0.0 (previously .2). safe_distance=200 remains in the
schema. The validator strictly requires zero scale, maps a deep copy to v3.24
with .2, then delegates all other frozen checks to validate_v324.

No common Safety function or step implementation is modified. It remains called
by inherited step, returning zero. All actual per-agent/per-step rewards satisfy:

```text
R = R1 + R2 + R3 + R4
safe_rewards = safety_rewards = outcome_rewards = 0
```

win_reward, lose_penalty, draw_reward and team_casualty_penalty remain zero. There
is no new penalty, collision event, terminal reward or team sharing.

## Frozen components

Body-frame R3/R4, signed 56D observations, horizontal AA gates, coefficients and
continuous w3/w4 are inherited unchanged. In particular there is no 3km hard phase
switch. Weapon range=1000, ammo=6, shared-Gaussian hit model and 1-second attacker
cooldown are unchanged. Controller, Blue encoder/nearest pursuit, scenario,
point-mass dynamics, RK4, bounds and termination criteria are unchanged.

MAPPO YAML is byte-identical to v3.24. Only registration and version-aware metrics
are extended; network, PPO trainer and hyperparameters are unchanged. v3.24
checkpoint evaluation/resume into v3.25 is rejected by the existing identity
contract. Replay uses the existing dynamic dispatch.

## Validation

Run in the uav Conda environment:

```text
python -m pytest tests/test_combat_v325.py tests/test_combat_v324.py tests/test_combat_v323.py tests/test_combat_protocol.py::test_factory_and_public_api_are_combat_only -q
```

177 passed: v3.25 75, v3.24 37, v3.23 64, registration 1. Tests cover symmetric
thresholds and priority, unchanged death pose, zero Blue returns, no kill credit,
last-Blue elimination, Red accounting, actual zero Safety at multiple distances,
exact returned R1-R4 decomposition for ordinary/guide/offense/defense/weapon/
ground/boundary/timeout cases, continuous reward identity, observations, fixed
actions, eligibility/sampling/cooldown/ammo and checkpoint isolation.

At a 100m separation between living Red aircraft, v3.24 Safety=-.1 for the nearby
agent and v3.25 Safety=0. Historical source/config hashes are checked, with only
shared registration files changed. Results: outputs/v325_validation/validation.json.
No formal training is run.
