# Combat Environment v3.32

v3.32 directly subclasses CombatEnvironmentV330, not v3.31. It replaces the
low-level load-factor controller with simplified rate-limited 3D kinematics.
The scientific variable is the action-to-motion interface; this is a decision
learning baseline, not a high-fidelity aircraft model or an exact paper controller.

Normalized actions are yaw-rate, pitch-rate and acceleration fractions. Limits
are 45 deg/s, 30 deg/s and 25 m/s^2. With dt=0.1, action.command stores the
per-step limits pi/40, pi/60 and 2.5. There is no action.controller section,
no 10G or replacement G limit, no nz/nx/phi control and no tracking constants.

_advance reuses env.v311_kinematics.advance_aircraft. It clips action, updates
heading/pitch/speed, then updates NED position from the new state. Heading is
wrapped; pitch and speed remain bounded; dead aircraft remain unchanged. Both
teams execute this same function. The flight update calls neither the old
controller nor PointMassDynamics/RK4. [1,0,0] holds level altitude and speed
without pitch compensation. Combined channels are independent until their
own aircraft state boundary is reached.

4v4/45D, self phi placeholder=0, arena, random-diameter initialization, weapon,
fire control, ammo/cooldown, hit probability, noncombat losses, simultaneous
combat resolution, transition order and outcome remain v3.30. Mutual
destruction is draw. R1/R2/R3/R4 coefficients and geometry remain exactly
v3.30, including offense .01/.02/.10 and threat -.015/-.025/-.15; v3.31's
R4 x2 is not imported. The validator maps only version/action back to v3.30.

Blue's existing nearest-live pure-pursuit policy is unchanged, uses 225m/s,
and still selects the nearest target every step with index tie breaking and
pursuit/fire coupling. Error normalization now uses the new per-step limits.
No PID, lead pursuit, target lock, hysteresis or tactical policy is introduced.

No physical controller diagnostics are fabricated: controller_diagnostics is
empty and executed_red_phi is omitted. The transition's legacy zero-vector
return is only an internal compatibility placeholder, not a roll command.
Observation retains its existing zero phi field. Historical controller
diagnostics remain available in v3.29-v3.31.

New MAPPO/MADSAC/MAAC configs preserve all formal v3.30 hyperparameters and
dimensions (45,3,4). Only existing environment metadata changes to 3.32;
MAPPO has no such config field, so its config is identical to v3.30. Strict
version/config identity rejects v3.30/v3.31 checkpoints for v3.32.

tests/test_combat_v332.py covers linearity, symmetry, channel independence,
level circles, bounds, dead states, legacy engine non-use, Blue scaling and
switching/coupling, frozen combat/reward/outcome and checkpoint isolation.
tools/audit_combat_v332_kinematics.py performs motion comparisons and 200-seed
fixed-pursuit fairness without training. Outputs are under outputs/v332_validation.
No formal training is started.
