# Combat Environment v3.20

v3.20 directly inherits v3.19. Only _advance and the version are overridden.
The new low-level controller maps actor [a_psi, a_theta, a_v] in [-1,1]^3
to feasible normal-load components rather than large desired angular rates.
This is a controller correction, not a claim of original paper action semantics.
Heading/pitch actor actions now mean normalized lateral maneuver strength and
longitudinal normal-load allocation; observation/action/team dimensions remain
56/3/5.

```text
A0 = cos(theta)
Bmax = sqrt(max(nz_max^2 - A0^2, 0))
B = a_psi * Bmax
Amax = sqrt(max(nz_max^2 - B^2, 0))
A = A0 + a_theta*(Amax-A0)   if a_theta >= 0
    A0*(1+a_theta)          otherwise
nz = hypot(A,B)
phi = atan2(B,A), or 0 when nz=0
```

The components are generated inside the load budget, with no post-hoc scale,
eta/calibration coefficient, smoothing, negative-g or inverted-flight model.
Zero pitch action retains theta_dot=0 at the start of the integration step,
including maximum lateral action. Normal load remains <=8 up to floating-point
roundoff. During RK4 a fixed control is used as before; instantaneous derivative
claims do not add a new stage-wise feedback controller.

Speed is unchanged: target_v=clip(v+50*a_v,150,300), acceleration=(target_v-v)/2,
nx=sin(theta)+acceleration/g. There is no nx or acceleration limiter.
Existing point-mass dynamics and RK4, including speed/pitch clips and heading
wrap, are unchanged. There is no pitch-boundary braking or prediction mechanism.

Red and Blue both use action_to_control_v320. Blue retains its existing
GLOBAL_NEAREST_PURSUIT strategy, target selection, and heading/elevation error
normalization by heading_delta_max/pitch_delta_max. No new Blue adapter or
GroundAware logic is added. Consequently its old normalized actions are executed
under the new low-level maneuver interpretation too.

The environment config differs from v3.19 only by environment_version.
Heading/pitch delta limits and time constants remain in the frozen schema,
although the new controller does not use them for angular-rate generation.
Speed still uses its 2s time constant. The MAPPO config is byte-identical to
v3.19, including formal widths, hyperparameters, evaluation/checkpoint protocols
and 3M target.

All v3.19 reward terms (including 1–3km R4 pair weights and Safety), weapon
range=1000m, body gates/Eq.(8), repeat fire, observation, center_radius=4000m,
scenario, termination and timeout remain inherited. Historical controller files
and v3.19 remain untouched. Cross-version checkpoint evaluation/resume is rejected.

Validation inside uav:

```text
python -m pytest tests/test_combat_v320.py tests/test_combat_v319.py -q
python -m pytest tests/test_combat_protocol.py::test_factory_and_public_api_are_combat_only -q
```

No training is run. Results and measured instantaneous rates are saved under
outputs/v320_validation/.
