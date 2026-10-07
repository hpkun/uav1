# Paper-Constrained Direct 4v4 Combat Environment V2.3

This document is the normative active environment contract. V2.3 is a public
reconstruction constrained by Li et al. (2023); it is not claimed to be the
authors' unreleased simulator. V2.2 introduced a true 3-D launch cone, but left
the hit errors in world-horizontal ATA/HA coordinates. A 100,000-sample audit
showed statistically different hit probabilities for physically identical
zero-off-boresight horizontal, climb and dive shots. V2.3 therefore changes only
the two Eq. (8) angular errors to the attacker's velocity frame. Dynamics,
observations, rewards, attack cadence, thresholds and noise scales are unchanged.

## Provenance table

| Component | Final value/design | Evidence class | Source/reason |
|---|---|---|---|
| 3DOF | NED point mass, paper Eq. (1)-(2) | PAPER | Section 2.1 |
| `dt` / integrator | 0.1 s / RK4 | PAPER / PREDECESSOR | Paper model interval / frozen public integrator |
| Speed, pitch, roll | `v=[150,300] m/s`, `theta=+/-pi/3`; no roll state, executed `phi` is a control | PAPER | Table 1 and Eq. (2) |
| Team sizes | Red 4, Blue 4 | RECONSTRUCTION | Direct target experiment contract |
| Action | relative `[Delta psi,Delta theta,Delta v]`, maxima `[pi,pi/3,50]` | PAPER | Table 2 and Eq. (23) |
| Controller | 2 s first-order desired rates and Eq. (2) inverse | RECONSTRUCTION | Missing author controller replaced by disclosed prototype-selected design |
| `nz` limit | 8g, proportional A/B projection | RECONSTRUCTION | Explicit frozen choice from controller prototype validation |
| Geometry | signed horizontal ATA, AA, HA, optional HCA, and true 3-D off-boresight | PAPER / DERIVED / RECONSTRUCTION | Eq. (6) fields plus disclosed physical launch-cone correction |
| Combat area | hard horizontal radius 5 km (10 km diameter) | RECONSTRUCTION | Fully disclosed finite benchmark arena |
| Initialization | random diameter; centers at radius 4 km; disclosed offsets/noise | PAPER / RECONSTRUCTION | Paper random-diameter statement plus public numeric fill-in |
| Altitude | initial `3000+/-100 m`; only ground `alt<=0` destroys | RECONSTRUCTION | Public initialization and minimal ground rule; no ceiling |
| Observation | exact 52-scalar self/allies/enemies layout below | PAPER / DERIVED | Eq. (24) content plus explicit normalization/index derivation |
| Fire gate | `d=[0,4000]`, true 3-D off-boresight `<=30 deg` | RECONSTRUCTION | Uses the paper's 4 km and 30-degree limits but corrects the separable gate so aircraft pitch participates |
| Hit model | Eq. (8) in local velocity-frame azimuth/elevation, `D_hit=4000/ln(6)`, `c4=c5=1`, independent draws | PAPER / DERIVED / RECONSTRUCTION | Equation form / calibrated distance / 3-D consistency correction |
| Cadence | one attempt on entry into the union of legal windows | RECONSTRUCTION | Prevents hidden 10 Hz repeated-fire assumption |
| Reward | R1+R2+R3+R4 only | PAPER / DERIVED | Eq. (25), with documented nearest-target and precedence rules |
| Blue | nearest Red, LOS heading/elevation, 250 m/s, same controller | PAPER / RECONSTRUCTION | Section 2.5 pursuit rule plus disclosed speed/controller |
| Termination | elimination outcomes; 1000-step Red failure; mutual destruction draw | PAPER / RECONSTRUCTION | Paper Red success criterion plus explicit edge cases |
| Sensor noise | disabled | RECONSTRUCTION | Deterministic public benchmark choice |
| MAPPO baseline | unchanged | PROJECT | Shared actor, centralized attention critic and PPO/GAE training are outside the environment contract |

## Dynamics and action

`AircraftState=[x,y,z,v,theta,psi]` uses NED, so altitude is `-z`. The existing
3DOF derivatives and RK4 implementation are unchanged. Integrator limits are
speed 150-300 m/s and pitch +/-60 degrees.

The actor output order is exactly `[a_psi,a_theta,a_v]`. After clipping each
component to `[-1,1]`:

```
psi_d   = wrap(psi   + pi*a_psi)
theta_d =      theta + (pi/3)*a_theta
v_d     =      v     + 50*a_v
```

Thus zero action asks the controller to hold the current heading, pitch and
speed. Desired rates use time constants `tau_psi=tau_theta=tau_v=2 s`.
Inverting paper Eq. (2) gives:

```
A = max(cos(theta) + v*theta_dot/g, 0)
B = v*cos(theta)*psi_dot/g
nx = sin(theta) + v_dot/g
nz_raw = hypot(A,B)
```

If `nz_raw>8`, A and B are multiplied by `8/nz_raw`. Then
`nz=hypot(A,B)` and `phi=atan2(B,A)`. This preserves the requested pitch/yaw
rate direction while enforcing `nz<=8` and `abs(phi)<=pi/2`. There is no `nx`
cap.

## Canonical geometry

For attacker `i`, target `j`, horizontal LOS
`lambda=atan2(y_j-y_i,x_j-x_i)`, horizontal distance `rho`, and 3D range `d`:

```
ATA = wrap(lambda - psi_i)
AA  = wrap(psi_j - lambda)
HA  = atan2(alt_j-alt_i, rho) = atan2(-(z_j-z_i), rho)
HCA = wrap(psi_j-psi_i)
u_fwd = [cos(theta_i)cos(psi_i), cos(theta_i)sin(psi_i), -sin(theta_i)]
u_LOS = [x_j-x_i, y_j-y_i, z_j-z_i] / d
off_boresight = acos(clip(dot(u_fwd,u_LOS),-1,1))
```

Construct the orthonormal velocity frame as
`right=[-sin(psi),cos(psi),0]` and `up=cross(right,forward)`. With LOS projections
`los_f,los_r,los_u`, the hit-model errors are
`az_error=atan2(los_r,los_f)` and
`el_error=atan2(los_u,hypot(los_f,los_r))`. They satisfy
`cos(off_boresight)=cos(el_error)cos(az_error)`.

ATA, AA, HA and HCA remain available to observations, rewards and diagnosis.
Only `off_boresight` is the active angular launch gate. At zero
range the LOS direction is undefined and `off_boresight` is defined as zero to
preserve the inclusive zero-range contract.

## Scenario

A uniform horizontal angle selects a diameter. Red and Blue centers are the two
endpoints at radius 4 km, hence their centers are exactly 8 km apart. Each team
uses lateral offsets `[-450,-150,150,450] m`. Individual altitude is
`3000+U(-100,100) m`, speed is `225+U(-10,10) m/s`, and heading is the direct
opposing-center heading plus `U(-5,5) deg`. All aircraft begin inside the 5 km
arena, every Red-Blue pair begins beyond 4 km, and no initial fire window exists.
There are no scenario modes or curriculum.

## Weapon and firing state machine

The V2.3 launch window is inclusive: `0<=d<=4000 m` and true 3-D
`off_boresight<=30 deg`. Target aspect AA and any lock/dwell state are not part
of this gate. The Eq. (8) stochastic hit calculation remains unchanged.

For each attempt, Eq. (8) is evaluated with independent velocity-frame draws
`epsilon_az,epsilon_el ~ N(0,1)`:

```
threshold = pi*exp(-d/D_hit)
abs(az_error + epsilon_az) <= threshold
abs(el_error + epsilon_el) <= threshold
D_hit = 4000/ln(6) = 2232.442506204989 m
```

Each attacker owns one `armed` flag. No legal target sets `armed=true`. The first
step with one or more legal targets selects the nearest, makes exactly one attempt,
and sets `armed=false`. Remaining continuously inside any legal window cannot fire
again; leaving all windows rearms the attacker. Both sides use the same pre-hit
snapshot and all successful hits resolve simultaneously. Multiple successful Red
attackers against one target share its `+10` kill reward equally.

## Boundary, ground and step order

The only arena rule is `hypot(x,y)<=5000 m`. Exiting destroys either side. A Red
exit receives R2=-10 and never an additional R1 death penalty; a Blue exit is not
a Red kill. Ground contact is altitude `<=0`. Red ground loss receives R1=-10.
There is no upper altitude ceiling.

One step is ordered as follows:

1. Decode actions, compute controls, and integrate both sides.
2. Resolve hard-boundary exits, then ground contact.
3. Freeze the post-transition/pre-hit snapshot.
4. Compute Red R3 and R4 and count fire windows.
5. Apply entry-trigger logic and sample one Eq. (8) attempt when armed.
6. Resolve Red and Blue successful hits simultaneously.
7. Add R1 and R2; accumulate episode R1-R4 totals.
8. Build the next observations and outcome metadata.

## Reward

Only paper Eq. (25) is active:

- R1: shared `+10` for a Blue weapon kill; `-10` when that Red aircraft is
  weapon-killed or reaches the ground.
- R2: `-10` for a Red hard-boundary exit.
- R3: for the nearest living Blue on the pre-hit snapshot, `+0.001` when
  `abs(ATA),abs(HA)<=30 deg` and `d>=4000 m`.
- R4: for the same pair and `d<=4000 m`, Red outer `abs(AA)<=30 deg` has
  precedence and awards `+0.1,+0.02,+0.01` for nested 5, 15 and 30 degree
  ATA/HA tiers. Otherwise the reverse Blue relation can apply
  `-0.15,-0.025,-0.015` at the same tiers. Unmatched states receive zero.

No progress delta, closure, generic tactical score, fire bonus or extra shaping
term is active.

## Observation indices

The observation has exactly 52 floats and no sensor noise.

| Indices | Content |
|---|---|
| 0-6 | self: `x/5000,y/5000,alt/10000,(v-225)/75,last_phi/(pi/2),psi/pi,theta/(pi/3)` |
| 7-27 | three ally slots, 7 each: relative NED displacement projected into own flight-path frame `/10000`, ally speed, relative heading, ally pitch, alive |
| 28-51 | four enemy slots, 6 each: `d/10000`, enemy speed, `AA/pi`, `ATA/pi`, `HA/(pi/2)`, alive |

Dead ally/enemy slots are all zero. A dead observing aircraft receives an all-zero
vector. The environment resets last executed bank angles to zero and stores the
actual controller bank after each step for both teams.

## Blue policy, outcomes and instrumentation

Every step, each living Blue aircraft selects the nearest living Red aircraft,
commands horizontal LOS heading, LOS elevation and 250 m/s, converts those desires
to the same normalized increment action, and passes through the exact same
controller and 8g projection.

Eliminating all Blue aircraft is a Red win. Eliminating all Red aircraft is a Blue
win. Same-step full mutual destruction is a draw. Reaching 1000 steps with both
teams alive is `red_failure_timeout`: `red_success=false`, `red_win=false`, and
`draw=false`.

Per-step and episode JSON metrics distinguish fire-window steps/pairs, attempts,
successful weapon hits, credited kills, hard exits, ground losses, first-event
steps, and R1-R4 totals for Red and Blue where applicable. Console training output
contains compact MAPPO return, win/loss, fire/kill and PPO diagnostics.

Checkpoints store `environment_version=2.3`. Resume rejects missing or different
versions before loading weights because V2.0-V2.3 use equal tensor
dimensions with incompatible environment semantics.
