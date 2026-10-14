# Combat Environment v3.34

v3.34 is a pure initialization ablation of **v3.32**, directly implemented as
`CombatEnvironmentV334(CombatEnvironmentV332)`. It overrides only
`initial_states()`. It does not inherit v3.33.

The question is whether a 2D staggered initial formation and the resulting
concentrated pursuit topology provide better initial conditions for learned
flanking, outflanking or distraction. These behaviors are not assigned by rules.

## Complete configuration difference from v3.32

- `environment_version`: `3.32` to `3.34`.
- Replace `scenario.formation_offsets: [-300,-100,100,300]` with
  `scenario.formation_local_offsets: [[300,0],[-50,250],[-50,-250],[-200,0]]` m.

The scalar field is absent in v3.34. The validator restores those two schema
changes in a deepcopy, then delegates to unchanged `validate_v332`.
All other environment fields are identical, including center radius 4000 m,
Weapon range 1000 m, guide distance 3000 m and tactical distance 3000 m.
The three algorithm configurations change only environment identity.

## Local geometry

Let `u=(cos(alpha),sin(alpha))`, `t=(-sin(alpha),cos(alpha))`, with alpha drawn
once from the existing reset distribution. Team centers remain
`C_R=-4000u`, `C_B=4000u`. For each local offset `(parallel,lateral)`:

- Red: `p_R=C_R+parallel*u+lateral*t`.
- Blue: `p_B=C_B-parallel*u-lateral*t`.

Both local offset sums are zero, so actual centroids equal nominal centers.
Horizontal positions satisfy `p_B_i=-p_R_i`, preserving point symmetry and
identical local spacing. Blue uses forward `-u` and lateral `-t`.
Altitude 3000 m, speed 225 m/s, pitch 0, Red heading alpha and Blue heading
alpha+pi retain the v3.32 semantics, including the existing angle wrapping.

The lead, lateral and rear positions describe geometry only. They do not add
actor role IDs or scripted tactical roles. The initializer neither invokes
Blue policy nor assigns pursuit targets. Both teams' initial nearest targets
are naturally aircraft 0 by strict, rotation-invariant distance inequalities.
Moving another target nearer or removing target 0 immediately triggers the
unchanged nearest-surviving-target selection on the next decision step.

## Frozen combat semantics

Reward is exactly the inherited v3.32 implementation, including continuous
`distance_weights` w3/w4, max/max/min aggregation, R1/R2, AA threshold 30 degrees
and the original six R4 coefficients. v3.33 hard segmentation is not used.
Weapon, shared Gaussian hit model, ammo 6, cooldown 1 s and repeated-firing
semantics are unchanged. Motion retains 4.5/3 degrees and 2.5 m/s per 0.1 s step,
without controller/RK4 flight control. The Blue policy, 45D observation, sensor,
transition order and outcome are inherited unchanged.

Factory, version registry, evaluators and checkpoint protocols recognize 3.34;
network implementations and formal hyperparameters remain unchanged.
Old checkpoints cannot pass the environment version/config hash checks.

## Paper versus reconstruction

The paper supports 4v4, a 10 km battlefield and random-diameter opposite-side
initialization. The exact 2D coordinates and initial `[0,0,0,0]` pursuit pattern
are **project reconstruction**, not publicly specified paper coordinates or
an exact reproduction of Figure 7. This formation deliberately breaks the four
initial 1v1 pairings; whether MARL discovers coordinated tactics requires later
training and evaluation. Fixed-policy diagnostics alone cannot establish that.

## Diagnostics and validation

`tools/audit_combat_v334_initialization.py` requires CUDA and checks seeds 0-199
for centroids, point symmetry, boundary safety, unchanged attitude/speed and
real nearest-target distances. It reports the alpha=0 full distance matrix.
It also runs 200 episodes of 100-step zero-Red-action pursuit and 200 complete
episodes of fixed-vs-fixed nearest pursuit. Persistence and free-pursuit windows
are measurements, not constraints. Missing switch times within a finite horizon
are right-censored, not infinite persistence.

The tool counts live Red-Blue pair-step R4 opportunities at the existing
pre-fire reward stage: tactical distance, distance plus AA<=90/60/30 degrees,
actual positive offense pair reward, and body alignment<=30/15/5 degrees.
Body-only counts are separate from distance-conditional and AA30-conditional
alignment counts. These diagnostics never enter observation, reward, policy,
checkpoint ranking or checkpoint metadata. Each episode and step is recorded.

Decision action step k selects from state k-1, at time `(k-1)*dt`; firing occurs
after advance at time `k*dt`. Reports distinguish these timestamps.
`tools/validate_combat_v334_smoke.py` validates the three minimal 32-transition
CUDA smokes, checkpoint reload/evaluation and historical/hash rejection.
No formal training is run. Measured results and complete test outcomes are in
`outputs/v334_validation/report.md`.
