# Combat Environment v3.7: MADSAC hit model and reward fidelity

v3.7 directly inherits v3.6. Initialization, 5v5, 66D observations, actions,
controller/dynamics, shared radar, Blue v3.5 paper-style tiered policy and boundary
return, finite ammunition and strict pair-entry semantics are unchanged.
Historical versions retain their own weapon and reward configuration.

## Paper definitions and distance adaptation

MADSAC specifies absolute ATA and HA <= pi/6 and a distance envelope with original
maximum fire range 4 km. This project retains the v3.6 adaptation: fire 2 km,
radar 4 km with 90 degree FOV, guide region 2 < d <= 4 km and tactical region
d <= 2 km. AA is not a fire eligibility condition.

Equation (8) uses one epsilon ~ Normal(0,1) per actual attempt, shared by both:

```
ATA + c4 * epsilon <= pi * exp(-d / Dhit)
HA  + c5 * epsilon <= pi * exp(-d / Dhit)
```

Both inequalities must hold. Angles are nonnegative absolute radians. There is
no second Gaussian, uniform draw or Bernoulli stage. Eligibility and strict
pair-entry selection happen before sampling. All simultaneous entries are
consumed as in v3.6, including unselected pairs; each attacker selects at most
one nearest target. Red attempts precede Blue attempts using the same environment
RNG; the Blue policy RNG remains independent. Kill resolution remains simultaneous.

## Mathematical calibration of parameters omitted by the paper

The paper provides the equation and physical meanings, but no numerical values
for Dhit, c4 or c5. The supplied project calibration is:

- Dhit = 2000 / ln(6) = 1116.2212531024945 m.
- c4 = c5 = (pi/6) / Phi^-1(0.7) = 0.9984711359155588 rad.
- Phi^-1(0.7) = 0.5244005127080407.

At 2 km the noise-free threshold equals pi/6. At 2 km with ATA=HA=0 the
probability is 0.7. This value is only a calibration anchor: v3.7 has no constant
hit_probability configuration or constant probability combat sampler.

The pure diagnostic function returns Phi(min((threshold-ATA)/c4,
(threshold-HA)/c5)). It never determines combat via an additional draw.
Expected values:

| Distance m | ATA=HA degrees | Expected hit probability |
|---|---|---|
| 2000 | 0 | 0.700000 |
| 2000 | 5 | 0.668944 |
| 2000 | 15 | 0.603416 |
| 2000 | 30 | 0.500000 |
| 1500 | 0 | 0.794100 |
| 1500 | 30 | 0.616511 |
| 1000 | 0 | 0.900519 |
| 1000 | 30 | 0.776407 |

## Reward and diagnostic contract

Formal reward is MADSAC R1 + MADSAC R2 + adapted MADSAC R3 + adapted MADSAC R4
+ project-specific safety penalty.

- R1: +10 kill credit, -10 own death (including inherited ground loss handling).
  Existing shared kill credit remains equally split among credited attackers.
- R2: -10 own boundary loss.
- R3: guide +0.001 in the adapted visible 2–4 km region with ATA/HA <=30 degrees.
- R4: strongest offense .01/.02/.10 for 30/15/5 degree levels, plus strongest
  defensive threat -.015/-.025/-.15. Aspect <=30 degrees and tactical distance
  <=2 km apply as in v3.6. Offense uses radar visibility; defensive threats retain
  v3.6 geometry. R3/R4 are cached after dynamics/noncombat handling, before hits.
- Safety: unchanged 200 m close approach penalty, scale .2. This is project
  specific and is not claimed to be part of MADSAC Equation (25).

Compatibility-only disabled fields team_casualty_penalty, win_reward,
lose_penalty and draw_reward are frozen at zero. Other agents receive no casualty
externality. Win/loss/draw/timeout statistics and termination remain enabled,
with zero outcome reward in all fixed slots and zero episode outcome total.
PBRS remains disabled; all phi fields are zero and pbrs_enabled=False.

`r1_rewards` / `episode_r1_total` now denote combat, r2 boundary, r3 guide,
r4 offense+defense. The same true aliases are available as madsac_r1_total through
madsac_r4_total. individual_event_rewards excludes boundary; boundary_rewards
reports R2 separately. tactical_rewards excludes guide; tactical_total_rewards
retains the inherited guide+offense+defense diagnostic. safety_rewards is separate.
Episode diagnostic arrays and totals follow these same definitions.
Legacy event_rewards=R1+R2, adv_rewards=R3+R4 and safe_rewards=safety are retained
for tool compatibility. team_casualty_rewards and outcome_rewards are all zero.

Finite ammo=6, strict pair-entry, safety, the v3.5 Blue policy and boundary return,
and strongest-opportunity/strongest-threat multi-target aggregation are project
choices, not asserted as MADSAC paper designs. No policy or PPO parameter is tuned.

## Validation scope

Use configs/combat_environment_v37.yaml and the four *_5v5_v37.yaml algorithm
configs (byte-identical to v3.6). MADDPG/MADSAC version allowlists also accept 3.7.
Validation includes focused and full pytest, compileall, spawn vector reset, and
four formal-width CUDA smokes capped at 512 transitions each, one real update,
checkpoint save/reload and two deterministic episodes each. No large audit,
parameter sweep or formal training is part of this change.
