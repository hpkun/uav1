# Combat Environment v3.28

Direct baseline: CombatEnvironmentV326, not V327. The environment class only
sets its version and inherits every behavioral method from v3.26. It has no
locked_target_indices and no V327LockedTargetBluePolicy.

The only config differences from v3.26 are version='3.28', common
action.controller.normal_load_max=10 (formerly 8), and Blue desired_speed=225m/s
(formerly 250). The validator maps a deep copy back to v3.26's 8/250 values for
all frozen checks, and strictly requires formal 10/225 values.

Both Red and Blue still use action_to_control_v320's feasible A/B allocation,
PointMassDynamics and RK4. There is no extra angular-rate limit, lag, PID, drag,
stall or aerodynamic model. The common normal-load envelope now reaches 10g for
both sides. Blue uses the unchanged v3.21 action encoder with its desired speed
225. The existing +/-50m/s per-action speed target adjustment is retained: near
225 it targets 225 directly; farther away it approaches through that adjustment.
Red still has the complete 150..300m/s speed envelope and speed-action semantics.

Blue still reselects the nearest live Red in 3D at each step start, with the
existing smaller-index tie-break. This frozen target is shared by movement and
weapon checks. If it is ineligible, no off-target shot occurs even when another
Red is eligible. The next step can change to a closer live Red; there is no
persistence or new tactical mechanism.

10g and fixed Blue 225m/s are project modeling choices for the current 1km
close-combat scenario, not publicly reported exact MADSAC paper parameters.
Their purpose is to reduce turn scale and leave room for Red speed management,
slowing turns and positioning, without encoding a tactic through extra reward.
The change alone does not establish successful learning.

Reward, signed 56D observations, continuous w3/w4, R1-R4 coefficients, zero Safety
and terminal/team reward, range=1000, body gates +/-30deg, ammo=6, 1s cooldown,
shared-Gaussian hit model, simultaneous resolution, boundary/ground/ceiling death,
scenario, timing and termination remain unchanged. Blue desired speed is fixed,
not adaptive; no energy-management or lead/lag strategy is added.

MAPPO YAML is byte-identical to v3.26. MADSAC YAML differs only by environment
identity; implementation remains 3. No other algorithm has an existing formal
v3.26 config to extend. Networks, losses and all formal training hyperparameters
are unchanged. Cross-environment checkpoints remain rejected. Existing dynamic
recording dispatch supports v3.28.

## Maximum level-flight turn audit

tools/audit_combat_v328.py uses action=[1,0,0], recomputes the existing control each
simulation step, and advances the unchanged RK4 at dt=.1. Turn crossing times are
interpolated from unwrapped heading. Radius is independently fitted to RK4 xy
positions; analytical v/yaw_rate is also saved. This is a diagnostic trajectory,
not an in-arena combat episode. There is no new physical or training logic.

8g cases achieve nz=8 and phi=82.819244deg; 10g cases achieve nz=10 and
phi=84.260830deg. Rounded results:

|Speed m/s|Yaw deg/s, 8g→10g|90deg time s, 8g→10g|180deg time s, 8g→10g|Radius m, 8g→10g|
|---:|---:|---:|---:|---:|
|150|29.742→37.284|3.026→2.414|6.052→4.828|289.0→230.5|
|225|19.828→24.856|4.539→3.621|9.078→7.242|650.2→518.7|
|250|17.845→22.370|5.043→4.023|10.087→8.046|802.7→640.3|
|300|14.871→18.642|6.052→4.828|12.104→9.656|1155.9→922.1|

In particular, old Blue 250m/s+8g has radius802.7m, 90deg time5.043s and 180deg
time10.087s. New Blue 225m/s+10g has radius518.7m, 90deg time3.621s and 180deg
time7.242s: approximately35.4% smaller radius. This is maximum maneuver capability,
not a claim that the fixed pursuit policy commands maximum load in every step.

## Validation

v3.28专项25项，v3.27/v3.26/v3.25/v3.24/v3.23及factory/MAPPO checkpoint回归212项，
环境相关共237项通过。MADSAC相关66项通过，其中1项也包含在前述环境测试中。
因此共302个不同测试通过。覆盖无锁定的逐步切换、off-target抑制、10g预算、
双方同动作物理对称、Blue速度收敛、Red速度回归、配置冻结、奖励/武器/观测一致、
checkpoint隔离和历史persistent-lock行为。编译及差异检查通过。

Audit and result artifacts: outputs/v328_validation/turn_audit.json and
validation.json. Environment and historical configs/control/algorithm sources
are hash-checked; only shared version registration files change. No formal
training or hyperparameter changes are performed.
