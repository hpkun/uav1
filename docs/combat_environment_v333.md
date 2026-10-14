# Combat Environment v3.33

v3.33 is a **paper-aligned engagement environment**, not an exact reproduction
of Li et al. (2023), *Multi-UAV Cooperative Air Combat Decision-Making Based on
Multi-Agent Double-Soft Actor-Critic*, Aerospace 10(6), 574.

## Explicitly specified by paper

Rechecked against the repository PDF: Sec. 3.3.1 (pp. 12–13) specifies a 10 km
engagement diameter, 4 Red versus 4 Blue, and opposite ends of a randomly selected
diameter. Sec. 2.4, Eq.(7), specifies |ATA| and |HA| ≤30° and
Dfiremin ≤d≤Dfiremax; Table 1 (p. 13) gives Dfiremax=4 km. Eq.(8) (p. 7)
uses distance-dependent thresholds with one shared Gaussian perturbation for
ATA and HA. It names Dhit, c4 and c5 without their numerical values.

Eq.(25) (p. 14) specifies kill +10, Red loss −10, boundary −10, R3 +0.001
when d≥4000 m and both alignment angles ≤30°, and R4 when d≤4000 m and
the applicable |AA|≤30°. R41 levels at 30°/15°/5° are +0.01/+0.02/+0.10;
R42 levels are −0.015/−0.025/−0.150. At exactly 4000 m, both R3 and R4
distance conditions hold. R4 offense does not reward a 3.5 km head-on
state (AA≈180°), even though the weapon can fire in that state.

Sec. 2.5 (p. 7) specifies that Blue chooses the nearest Red each simulation
step, switches when that target changes, and uses the same pursuit during attack.

## Retained project assumption because paper does not specify it

`CombatEnvironmentV333(CombatEnvironmentV332)` inherits all motion, transition,
sensor, observation, firing and outcome methods; only `dense_combat_reward` is
overridden. Direct yaw/pitch/speed maxima remain 4.5°/3°/2.5 m/s per 0.1 s step.
There is no flight controller. The existing 3D body azimuth/elevation implements
the alignment constraints; AA retains the project's horizontal definition.

Team centers are 4.8 km from the origin, leaving boundary margin for the retained
−300/−100/100/300 m transverse formation offsets. Speed 225 m/s, altitude 3000 m,
initial headings and all initialization semantics except center radius are retained.
The paper does not establish these exact offsets or center margin.

Range minimum 0, ammo 6, cooldown 1 s, inherited firing state/re-entry behavior,
simultaneous combat resolution and mutual destruction as draw remain project
assumptions. No firing state machine is added. The inherited transition is
authoritative, including Red's eligible-target selection and Blue's pursuit-target
fire coupling. Dhit=2232.442506204989 and c4=c5=0.9984711359155588 are retained
project calibration values, **not values sourced from the paper**.

Aggregation remains guide=max over live targets, offense=max over live targets,
defense=min over live reverse threats; tactical=guide+offense+defense. Dense reward
does not call `distance_weights`, w3 or w4. Observations remain 45D for 4 agents;
sensor range remains 10 km and off-boresight 60°. No additional reward is added.

## Complete environment configuration difference from v3.32

| Field | v3.32 | v3.33 |
|---|---:|---:|
| environment_version | 3.32 | 3.33 |
| scenario.center_radius (m) | 4000 | 4800 |
| weapon.range_max (m) | 1000 | 4000 |
| reward.guide_distance_min (m) | 3000 | 4000 |
| reward.tactical_distance_max (m) | 3000 | 4000 |

The sixth research difference is implementation: continuous 1–3 km reward weights
become hard 4 km gates. All other environment fields are identical. The validator
reverses those five field changes and delegates to the unchanged strict v3.32
validator. Algorithm YAML files differ only in environment version; official
hyperparameters and actor/critic implementations are unchanged. Version registry
and evaluator/protocol integration recognize 3.33 while preserving historical
branches. Checkpoint version and environment/algorithm hashes remain strict.

## Validation

`tests/test_combat_v333.py` covers canonical offense/defense geometry, inclusive
4 km gates, weapon eligibility without AA, baseline inheritance, config drift,
historical checkpoint rejection and diagnostic hit probabilities.
`tools/audit_combat_v333_engagement.py` requires CUDA and runs seeds 0–199 for
initialization and symmetric fixed pursuit in both versions, recording first
fire windows, first actual volley distances, hit rates and reward components.
The comparison is **scenario-level**, since initial center radius differs;
it is not a same-trajectory counterfactual. Blue-equivalent rewards are diagnostic
only, sampled at the same pre-fire stage as Red rewards.

`tools/validate_combat_v333_smoke.py` checks the three 32-transition CUDA smokes,
updates, replay where applicable, checkpoint reload, evaluation and identity
rejection. Smoke runtime overrides use existing runner smoke options; formal
YAML settings remain unchanged. No formal training is run. See
`outputs/v333_validation/report.md` for measured results and test counts.
