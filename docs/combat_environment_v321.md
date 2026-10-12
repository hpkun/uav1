# Combat Environment v3.21

v3.21 directly inherits v3.20 and only replaces fixed_policy after initialization.
It does not override _advance: both teams use the unchanged action_to_control_v320.
Red actor actions enter that controller directly.

V321NearestTargetBluePolicy directly inherits PaperNearestTargetBluePolicy and
only adds constructor data and action_toward encoding. GLOBAL_NEAREST_PURSUIT,
nearest living target selection each step, immediate switches, desired heading
and elevation geometry, desired_speed=250, no-target zero actions and diagnostics
are all inherited. There is no tactical or flight-control redesign.

For a heading error wrapped to [-pi,pi), the encoder restores the 2-second
tracking intention in the new maneuver action space:

```text
A0 = cos(theta)
Bmax = sqrt(max(nz_max^2 - A0^2, 0))
desired_psi_rate = wrap(desired_heading - psi) / T_psi
B_required = v * A0 * desired_psi_rate / g
a_psi = clip(B_required / Bmax, -1, 1)
B = a_psi * Bmax
```

Bmax <= 1e-12 yields zero heading action. This local numerical constant is not
a configuration parameter. B is evaluated using the actual float32 heading
action returned to the common controller.

Pitch retains the historical action-limited and aircraft-clipped pitch target:

```text
legacy_a_theta = clip((desired_elevation-theta)/pitch_delta_max, -1, 1)
target_pitch = clip(theta+pitch_delta_max*legacy_a_theta, theta_min, theta_max)
desired_theta_rate = (target_pitch-theta)/T_theta
A_required = A0 + v*desired_theta_rate/g
Amax = sqrt(max(nz_max^2-B^2, 0))
A_feasible = clip(A_required, 0, Amax)
a_theta = (A_feasible-A0)/(Amax-A0)   if A_feasible>=A0 and Amax-A0>1e-12
          0                         if A_feasible>=A0 and no positive budget
          A_feasible/A0-1            otherwise
a_theta = clip(a_theta, -1, 1)
```

Heading first determines B; pitch uses the remaining Amax. Infeasible demands
saturate naturally without post-hoc scaling, negative-g, eta, PID or dynamics.
The ±60-degree formal pitch domain keeps A0 positive. At exactly 180 degrees,
the existing heading wrap chooses the negative direction.

Speed encoding is unchanged, including float32 output:
a_v=clip((desired_speed-v)/speed_delta_max,-1,1).
Speed control, feasible load controller, dynamics and fixed-control RK4 remain
unchanged. Reported tracking rates are instantaneous step-start derivatives,
not a claim of a feedback update within RK4.

The environment YAML differs from v3.20 only by version. MAPPO YAML is
byte-identical. Weapons (1000m, Eq.(8), repeat fire), rewards (3000m guide/tactical
range and R4 weights), Safety, observations, initialization and termination are
unchanged. Cross-version checkpoint evaluation/resume is rejected.

Validation in the uav Conda environment:

```text
python -m pytest tests/test_combat_v321.py tests/test_combat_v320.py tests/test_combat_protocol.py::test_factory_and_public_api_are_combat_only -q
```

56 v3.21 tests, 95 v3.20 regressions and one registration test passed.
At 225m/s level flight, 10/20-degree heading errors yield 5/10 degrees/s;
-2/-10-degree elevation errors yield -1/-2.498096 degrees/s. No training was run.
