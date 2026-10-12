# Combat Environment v3.15

v3.15 is a **minimal feasibility-corrected desired-command environment**.
It inherits v3.14 and fixes two interfaces: executed-phi observation feedback,
and infeasible desired pitch/speed targets. It is not a full high-fidelity flight controller.

The observation remains 56D with identical ordering. Self slot 4 is always zero,
including reset, transitions, dead aircraft and vector auto-reset. The builder
reuses v3.13; all other 55 slots retain their original definitions. Phi remains
a controller output used by dynamics and retained in Red/Blue diagnostic fields.
It is no longer presented to the actor as an aircraft state. No roll dynamics
or replacement state is introduced.

Normalized actions still clip to [-1, 1]. Heading remains
`wrap(psi + pi * a_heading)`. Desired pitch is
`clip(theta + (pi/3) * a_pitch, -pi/3, pi/3)`; desired speed is
`clip(v + 50 * a_speed, 150, 300)`. Clipping happens before the existing
`target_to_control`. Controller formulas and parameters, PointMassDynamics and
four-stage RK4 are unchanged. Both Red and Blue use the new thin wrapper.
Blue nearest-target policy and desired speed 250 are unchanged.

Reward (R1–R4, Safety, paper and extended totals), weapons, geometry, hit
sampling, simultaneous resolution, scenario, termination and all algorithm
hyperparameters are unchanged. The environment config differs from v3.14 only
in its version. The four algorithm configs are byte-identical to v3.14.

v3.13 and v3.14 retain their historical behaviors. v3.14 still exposes executed
phi in slot 4 and allows infeasible desired targets. Checkpoints must match the
environment version, even when tensor shapes agree. The old v3.14 500k policy
is used only for offline ZERO-phi regression, never for a v3.15 episode.

Validation commands (inside the uav Conda environment with CUDA):

```text
python -m pytest tests/test_combat_v315.py -q
python tools/audit_combat_v315.py
python tools/validate_combat_v315.py
python tools/smoke_formal_8v8.py --env-config configs/combat_environment_v315.yaml --output outputs/v315_validation/cuda_smoke
python -m pytest -q
python -m compileall -q env algorithm tools tests
```

The smoke command uses formal network/PPO settings, caps each algorithm at 512
transitions, saves/reloads a checkpoint, and evaluates two deterministic episodes.
Its output directory must be fresh. These checks do not launch formal training.
Results and historical baseline failures are recorded in
`outputs/v315_validation/report.md`.
