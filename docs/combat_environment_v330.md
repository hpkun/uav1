# Combat Environment v3.30: isolated 4v4 scale experiment

v3.30 directly subclasses CombatEnvironmentV329. Its only scientific change
is 5v5 -> true 4v4, with derived 56D -> 45D observations and four formation
offsets. No fifth aircraft is constructed, masked or padded in production.
The separate mutual-destruction classification bug fix described below changes
outcome labels only; it does not change the scale experiment's mechanics.

The question is whether current learning difficulty mainly comes from the
5v5 scale or from the existing reward/weapon design. This version does not
tune either mechanism and does not initiate formal training.

## Configuration contract

Compared with v3.29 only these environment fields change:

- environment_version: 3.29 -> 3.30
- scenario.team_size: 5 -> 4
- formation_offsets: [-400,-200,0,200,400] -> [-300,-100,100,300]

The validator maps a copy back to v3.29 and invokes its strict validator.
Centers remain 8000m apart, with adjacent aircraft spacing 200m, the same
single random diameter orientation and no added position/altitude/speed noise.

Reward R1/R2/R3/R4, max/min pair aggregation, continuous w3/w4, zero safety
and terminal/team reward remain unchanged. Weapon remains 1km, +/-30deg
body azimuth/elevation, six rounds, 1s cooldown and the same Gaussian hit model.
Controller remains shared paper increment tracking, 10G and 2s constants.
Blue remains 225m/s, step-start nearest live Red in real 3D with smaller-index
tie breaking, and strictly coupled pursuit/fire target. No persistent lock.
Arena, altitude/speed bounds, dt=.1, max_steps=1000, scenario centers, RK4,
PointMassDynamics remain frozen. Single-team elimination, timeout and ongoing
outcomes retain v3.29 semantics; mutual destruction is classified as draw.

## 45D observation

| Slice | Fields |
|---|---|
| 0:7 | self: normalized x,y,altitude,speed,zero roll field,heading,pitch |
| 7:25 | 3 allies x 6, increasing aircraft index excluding self |
| 25:45 | 4 enemies x 5, increasing aircraft index |

Ally fields preserve flight-frame relative position and normalized speed,
heading,pitch. Enemy fields preserve distance,speed,AA,signed body azimuth,
signed body elevation. Field definitions and scales exactly match v3.24/v3.29.
Dead own rows and dead entity slots are zero. Blue-team builder uses the same
layout with team/opponents exchanged. There is no 56D padding.

## Execution-path hardcoding audit

The constructor bypasses v3.11's five-element ammo/phi allocation. v3.30 owns
its four-element constructor, metrics reset, potential and observation builder.
Its transition helper copies the existing core's arithmetic and operation order,
changing only action shape and team-sized outcome/reward allocations. The
existing wrapper diagnostics/reward aliases are reconstructed in the same order.
Historical v3.0/v3.11/v3.36/v3.29 implementations are not edited.

Inherited v3.23 cooldown arrays and firing loop already use team_size; v3.26
Blue firing enumerates actual aircraft; v3.29 advance and geometry diagnostics
enumerate actual states. v3.29 reshape(-1,5) represents five statistic columns
(nz,abs_phi,three action magnitudes), not five agents. range(1,5) enumerates
four reward components R1-R4, not aircraft slots. Both remain intentionally.
Fire-state objects remain None, as in v3.29 repeat fire; no pair-entry matrix
or virtual fifth fire-state slot is introduced.

Production arrays have shapes ammo=(4,), cooldown=(4,), target_indices=(4,),
alive=(4,), reward=(4,), action=(4,3), observation=(4,45), eligibility=(4,4).
The regression test embeds four live aircraft into a historical 5v5 fixture
solely to compare frozen arithmetic/RNG; this does not occur in v3.30 runtime.

## Algorithms

New MAPPO/MADSAC/MAAC configs set obs=45, action=3, agents=4. Formal hidden
sizes, heads, learning rates, discount/entropy/target constants, batches,
replay, warmup, update schedules, evaluation and checkpoint schedules are
identical to their v3.29 counterparts. No architecture replacement or loss
change is made. MAAC's per-agent shared attention gradient averaging uses
1/N (1/4 here, 1/5 historically), a scale-derived normalization; its original
five-task reference metadata is not a new tunable parameter.

Strict protocols permit only the explicit 3.29/56/3/5 or 3.30/45/3/4 contracts,
retaining version/config/architecture checkpoint isolation. MADSAC/MAAC replay
and MAPPO rollout all have a real four-agent axis. Attention matrices are
[B,heads,4,4]. MAPPO's joint+focal MLP input has (4+1)*45=225 features.

## Validation and fairness

Run in uav (WSL where available, otherwise Windows). CUDA is mandatory for
training smoke, evaluation and checkpoint audits. Only minimal smoke runs
are authorized; the environment-only fixed-policy audit performs no training.

tools/audit_combat_v330.py runs at least 200 distinct seeds. Red uses a separate
stateless instance of the same nearest pursuit mathematics, avoiding pollution
of Blue's diagnostic target-switch counters. Both teams use 225m/s, the shared
10G controller and identical weapons. Blue coupling and Red eligible-target
selection are intentionally retained as environmental mechanics.

Current v3.30 outcome convention: if Red and Blue are both eliminated in the
same simulation step, mutual destruction is classified as draw:

```text
terminated = True
red_win = False
blue_win = False
red_success = False
draw = True
termination_reason = "draw_mutual_destruction"
```

The fix changes only outcome classification fields: red_win, blue_win, draw,
red_success and termination_reason. Combat resolution, attack kills, aircraft
states, R1/R2/R3/R4, episode reward, Weapon and Controller remain unchanged.
Single-team elimination, timeout and ongoing still inherit v3.29 semantics.

The latest fixed-vs-fixed fairness audit ran 200 seeds (0-199):

| Metric | Count |
|---|---:|
| Red win | 27 |
| Blue win | 26 |
| Draw | 147 |
| Timeout | 0 |
| Mutual destruction | 147 |
| Red attack kills | 774 |
| Blue attack kills | 773 |
| Red noncombat losses | 0 |
| Blue noncombat losses | 0 |

All 147 mutual-destruction episodes are included in draw and no longer count
as Red wins. Actual single-team wins are 27 vs 26 and attack kills are 774 vs
773. This fixed-pursuit mirror sanity check shows no obvious Red/Blue physical
capability bias in these 200 seeds. Per-seed comparisons confirm that all
non-outcome audit fields, including kills, losses and R1-R4, match the pre-fix
run exactly.

Artifacts are written under outputs/v330_validation: fairness.json,
smoke_validation.json and validation.json. The latest outcome-corrected audit
is outputs/v330_validation/outcome_fix/fairness.json, with comparison results
in outcome_fix/regression.json; the older fairness.json predates this fix.
No formal training is launched.
