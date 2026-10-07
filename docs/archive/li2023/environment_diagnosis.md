# Archived environment diagnosis for the discontinued Li et al. (2023) reconstruction

## Scope and evidence rules

This report diagnoses the frozen canonical environment; it does not tune it.
No canonical parameter, reward, controller, scheduler, or MADSAC update was
changed, and no RL training was run. The evidence labels mean:

- **PAPER-EXPLICIT**: a value or operational statement appears in the paper.
- **PAPER-EQUATION**: the mathematical form appears in a numbered equation,
  but this does not imply that all coefficients are published.
- **PAPER-INFERRED**: a reasonable reading of paper prose/figures, not an
  implementation value stated by the authors.
- **PAPER-UNSPECIFIED**: the paper does not resolve the implementation choice.
- **CURRENT-ASSUMPTION**: the frozen reconstruction's selected value/semantics.

Primary paper evidence is Section 2, Eqs. (1)-(8), Section 3.3, Tables 1-2,
Eq. (23)-(25), and Section 4.1/Figures 8-9. Machine-readable diagnostic output
is `outputs/environment_diagnosis/diagnosis.json`. It was produced by
`scripts/diagnose_environment.py` using 100,000 Eq. (8) samples per geometry,
1,000 reset seeds, and 200 fixed seeds for each engagement mode.

## A-C. Paper values, missing values, and current assumptions

| Item | Paper status and evidence | Frozen current value / semantics | Final classification |
|---|---|---|---|
| Simulation `dt` | Section 2: all models use a 0.1 s solution interval | 0.1 s | PAPER-EXPLICIT |
| Episode horizon | No maximum episode length published | 2,000 steps (200 s) | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |
| Battlefield diameter | Section 3.3.1: 10 km diameter | 10,000 m; boundary radius 5,000 m | PAPER-EXPLICIT |
| Initial layout rule | Both sides are at opposite ends of a randomly selected diameter | Symmetric team centers at radius 4,000 m | Wording PAPER-EXPLICIT; team-center mapping CURRENT-ASSUMPTION |
| Initial team-center distance | A literal team-center reading of opposite endpoints of a 10 km diameter is approximately 10 km; within-team geometry is not given | `center_distance=4000 m` from origin per side, hence 8,000 m center-to-center | PAPER-INFERRED versus PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |
| Formation spacing | Not published | 150 m tangential spacing | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |
| Initial altitude | Not published | 3,000 m | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION |
| Initial speed | Only the allowed interval 150-300 m/s is published | 225 m/s | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |
| Aircraft speed limits | Table 1 | 150-300 m/s | PAPER-EXPLICIT |
| Pitch/roll limits | Table 1: pitch +/-pi/3, roll +/-pi/2 | Same | PAPER-EXPLICIT |
| Sensor formula | Eqs. (3)-(5), including clipped Gaussian noise | Shared position noise, shared attitude noise, one speed noise draw | PAPER-EQUATION |
| Sensor `c1/c2/c3` | Symbols only; no values | 10 m, 0.01 rad, 1 m/s | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |
| Sensor `b1/b2/b3` | Symbols only; no values | 3, 3, 3 | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |
| `D_firemin` | Symbol in Eq. (7), no value in Table 1 | 0 m | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |
| `D_firemax` | Table 1: 4 km | 4,000 m | PAPER-EXPLICIT |
| `psi_firemax/theta_firemax` | Table 1: pi/6 | 30 degrees / 30 degrees | PAPER-EXPLICIT |
| Hit law | Both noisy ATA and HA inequalities in Eq. (8); one printed shared `epsilon_fire` | Same shared Gaussian draw | PAPER-EQUATION |
| `D_hit` | Symbol only | 2,000 m | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |
| `c4/c5` | Symbols only | 0.05 rad, 0.05 rad | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |
| Fire retry | Paper says the attack model decides firing in real time; no miss/retry lifecycle | Every alive attacker satisfying Eq. (7) samples Eq. (8) again on the next 0.1 s step | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |
| Missile cooldown | Not published | None | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |
| Missile ammunition | Not published | Unlimited/no ammunition state | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |
| Point-mass motion | Eqs. (1)-(2) | NED six-state model, RK4 | PAPER-EQUATION; numerical integrator PAPER-UNSPECIFIED |
| Action-to-target law | Table 2 and Eq. (23) | `[delta_psi,delta_theta,delta_v]` added to current state | PAPER-EXPLICIT/PAPER-EQUATION |
| Low-level controller law | Paper only says desired state is converted to `[phi,nz,nx]`; no law/gains | Proportional inverse-dynamics bridge | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |
| Controller gains | Not published | `k_yaw=k_pitch=k_speed=1` | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |
| Controller limits | Not published | `nx [-3,3]`, `nz [-6,6]`, yaw rate 1 rad/s, pitch rate 0.7 rad/s, acceleration 50 m/s2 | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |
| Observation fields | Eq. (24) | Derived fixed 45-dimensional expansion | PAPER-EQUATION; dimension PAPER-INFERRED |
| Observation normalization | Not published | position/5,000; speed/300; fixed ID slots; dead slots zero | PAPER-UNSPECIFIED; CURRENT-ASSUMPTION; HIGH IMPACT |

The paper reports MADSAC converging near return 41 and about 0.6 Red UAV losses
per battle. The seed-2023 8M run instead has full-training return 2.13 and Red
losses 3.78; its 80 fixed-seed evaluations average return 6.18 and Red losses
3.32. These result values are observations, not evidence for choosing missing
environment parameters.

## D. Initial geometry measured over 1,000 resets

Seeds `30000000..30000999` were reset without stepping or training.

| Quantity | Mean | Std | Min | P10 | P50 | P90 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Red center radius (m) | 4000.000 | ~0 | 4000.000 | 4000.000 | 4000.000 | 4000.000 | 4000.000 |
| Blue center radius (m) | 4000.000 | ~0 | 4000.000 | 4000.000 | 4000.000 | 4000.000 | 4000.000 |
| Center-to-center (m) | 8000.000 | ~0 | 8000.000 | 8000.000 | 8000.000 | 8000.000 | 8000.000 |
| Minimum Red-Blue pair distance (m) | 8000.000 | ~0 | 8000.000 | 8000.000 | 8000.000 | 8000.000 | 8000.000 |
| Maximum Red-Blue pair distance (m) | 8012.646 | ~0 | 8012.646 | 8012.646 | 8012.646 | 8012.646 | 8012.646 |
| Absolute initial ATA (deg), all 16 pairs | 1.342 | 1.039 | 0 | 0 | 1.074 | 3.219 | 3.219 |
| Absolute initial HA (deg) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Absolute initial AA (deg) | 178.658 | 1.039 | 176.781 | 176.781 | 178.926 | 180 | 180 |
| Initial closing speed (m/s) | 449.803 | 0.227 | 449.290 | 449.290 | 449.921 | 450 | 450 |

The random diameter changes only orientation. It does not randomize range,
altitude, speed, heading relation, or formation shape. Every episode starts as
an almost perfectly aligned, same-altitude, 450 m/s head-on closure.

Theory gives `(8000-4000)/450 = 8.8889 s = 88.8889 steps` to the 4 km
envelope. Straight-flight rollout observes first fire at step 89 and 3,995 m,
confirming the calculation.

## H. Weapon Eq. (8) mathematical diagnosis

Canonical `D_hit=2000 m`, `c4=c5=0.05`. Each cell below is a 100,000-sample
Monte Carlo probability with ATA=HA at the printed angle. A fixed RNG seed was
used. The critical 4 km/0 degree result was separately reproduced by 100,000
direct calls to `WeaponModel.sample_hit`.

| Distance (m) | Threshold rad | Threshold deg | 0 deg | 5 deg | 15 deg | 25 deg | 30 deg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 2.446675 | 140.184 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 1000 | 1.905472 | 109.176 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 2000 | 1.155727 | 66.218 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 3000 | 0.700984 | 40.163 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.999760 |
| 3500 | 0.545927 | 31.279 | 1.000000 | 1.000000 | 1.000000 | 0.985880 | 0.672390 |
| 4000 | 0.425168 | 24.360 | **1.000000** | **1.000000** | 0.999410 | 0.412310 | 0.024460 |

At 4 km and near-zero ATA/HA, the threshold is 24.36 degrees while weapon
noise has a standard deviation of only 0.05 rad (2.865 degrees). Failure needs
an approximately 8.5-sigma draw. The measured single-attempt probability is
1.000000 at 100,000-sample resolution. It is not merely a close-range effect:
shots at 3.5 km remain nearly certain through 25 degrees.

## I. Repeated-fire semantics

`env.step()` visits every alive attacker on every 0.1 s step. If its nearest
alive target satisfies Eq. (7), exactly one `sample_hit()` call is made. A miss
does not set ammunition, cooldown, in-flight missile, or launch history, so the
next step may immediately try again: up to about 10 attempts/s/attacker while
the gate stays true. Successive Gaussian draws are statistically independent,
although they come from the environment's shared RNG and geometry evolves.

For a per-attempt probability `p`, `P(hit within N)=1-(1-p)^N`:

| 4 km geometry | N=1 | N=2 | N=5 | N=10 |
|---|---:|---:|---:|---:|
| ATA=HA=0 deg | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 15 deg | 0.999410 | 1.000000 | 1.000000 | 1.000000 |
| 25 deg | 0.412310 | 0.654620 | 0.929896 | 0.995085 |
| 30 deg | 0.024460 | 0.048322 | 0.116462 | 0.219360 |

Retry semantics are high impact in general. In the measured canonical first
engagements, however, every actual attempt hit because first-fire angles were
well aligned. Thus the first-volley mass exchange is caused primarily by the
current lethality coefficients plus initial geometry; retry mainly accelerates
cleanup in the following one or two steps.

## E-G, J-K. First-engagement results

The casualty windows include the first-fire step and end at
`first_fire_step + ceil(window/dt)-1`. The observer receives already sampled
events and never calls the weapon or RNG. Tests compare observed and unobserved
environments step by step and confirm identical observations, rewards,
termination, kills, and RNG trajectory.

### Scripted straight-flight Red, 200 episodes

Seeds `40000000..40000199`; Red actions are exactly zero and Blue keeps the
paper fixed nearest-target pursuit policy.

| Quantity | Result |
|---|---:|
| First Red can-fire step | 89 in 200/200 |
| First Blue can-fire step | 89 in 200/200 |
| First-fire distance | 3995 m |
| First-fire absolute ATA/HA | approximately 0 / 0 deg |
| First successful-hit step | 89 in 200/200 |
| First-casualty step | 89 in 200/200 |
| Episode length | 89 in 200/200 |
| Episode length minus first fire | 0 |
| First-step Red attempts/proposals/kills | 4 / 4 / 4 in 200/200 |
| First-step Blue attempts/proposals/kills | 4 / 4 / 4 in 200/200 |
| Final Red losses / Blue losses | 4 / 4 in 200/200 |
| Termination | `all_blue_destroyed` in 200/200 due literal Red success tie handling |

This is deterministic first-contact mutual annihilation. Simultaneous proposal
application allows both teams to die on the same step; the current formal
success condition counts all-Blue-destroyed as Red success even in a 4v4 tie.

### Deterministic 8M MADSAC, 200 episodes

Checkpoint `checkpoint_8000016.pt`, CPU deterministic actor, seeds
`41000000..41000199`.

| Quantity | Mean | Std | Min | P10 | P50 | P90 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| First-fire step | 90.690 | 0.462 | 90 | 90 | 91 | 91 | 91 |
| First-casualty step | 90.690 | 0.462 | 90 | 90 | 91 | 91 | 91 |
| Episode length | 92.115 | 0.657 | 91 | 91 | 92 | 93 | 93 |
| Episode length minus first fire | 1.425 | 0.494 | 1 | 1 | 1 | 2 | 2 |
| First Red fire distance (m) | 3988.293 | 5.576 | 3970.874 | 3981.276 | 3988.338 | 3996.160 | 3999.596 |
| First Blue fire distance (m) | 3989.012 | 5.496 | 3974.505 | 3982.075 | 3988.642 | 3997.236 | 3999.977 |
| First Red absolute ATA (deg) | 6.631 | 2.676 | 0.002 | 2.017 | 7.377 | 9.619 | 10.533 |
| First Blue absolute ATA (deg) | 0.821 | 0.879 | 0.086 | 0.225 | 0.503 | 1.952 | 4.464 |
| First Red absolute HA (deg) | 0.547 | 0.453 | 0.018 | 0.079 | 0.423 | 1.267 | 1.940 |
| First Blue absolute HA (deg) | 0.396 | 0.229 | 0.004 | 0.083 | 0.404 | 0.680 | 1.131 |
| Final Red losses | 3.110 | 0.760 | 2 | 2 | 3 | 4 | 4 |
| Final Blue losses | 3.895 | 0.307 | 3 | 3 | 4 | 4 | 4 |

All 200 first successful hits and casualties occur on the first-fire step.
Every episode terminates one or two steps later: only 0.1-0.2 s after first
contact. Losses within 0.5 s are already equal to final losses: 3.11 Red and
3.895 Blue on average. Red wins 179/200 (89.5%); 21/200 terminate with all Red
destroyed.

At the first successful exchange:

- Red makes 2.825 successful proposals and kills 2.220 Blue UAVs on average.
- Blue makes 3.725 successful proposals and kills 1.465 Red UAVs on average.
- Proposal counts exceed deaths because several attackers can select the same
  nearest target; duplicate successful proposals credit only one death.
- 87/200 (43.5%) first exchanges kill at least two UAVs on both sides.

| First-step actual Red deaths vs Blue deaths | Episodes |
|---|---:|
| 1 vs 1 | 29 |
| 1 vs 2 | 43 |
| 1 vs 3 | 40 |
| 2 vs 1 | 1 |
| 2 vs 2 | 52 |
| 2 vs 3 | 30 |
| 3 vs 2 | 1 |
| 3 vs 3 | 4 |

The MADSAC actor delays contact by only 1.69 steps relative to straight flight.
Blue arrives almost perfectly aligned, while Red is still only 6.63 degrees
off-axis on average. Under current Eq. (8) parameters both are effectively in
the certain-hit region. Across the full episode, Red and Blue average 6.0 and
5.93 fire attempts, respectively; every measured attempt is a successful hit
proposal before duplicate-target resolution.

**Conclusion for the 90-95-step hypothesis:** confirmed. The canonical
environment is a first-contact-dominated, high-lethality regime. Approximately
89-91 steps are spent closing from 8 km to the 4 km gate, first fire and first
casualty coincide, and termination follows in at most two additional steps.

## L. Direct environmental support for high 8M Red losses

Yes. The evidence is direct rather than inferred from training curves:

1. Current initialization is a nearly perfectly aligned 450 m/s head-on merge.
2. Current 4 km/near-zero-angle single-shot probability is indistinguishable
   from one at 100,000-sample resolution.
3. Straight flight produces 4v4 same-step mutual annihilation in 200/200 runs.
4. The trained actor's first-fire angles remain inside the near-certain region.
5. Its first exchange kills 1.465 Red UAVs immediately and 43.5% of episodes
   lose at least two UAVs on both sides on that first step.
6. Within 0.5 s, Red losses already equal their final value of 3.11.

The diagnostic 3.11 Red-loss mean uses 200 new fixed seeds and is consistent
in scale with the saved 8M evaluation history (3.32 average across 80 points)
and full-training summary (3.78). This does not prove that the algorithm is
optimal; it does show that the frozen environment itself strongly creates the
observed high-loss regime.

## M. Sensitivity priority, without proposed paper values

These are groups to test one at a time. No candidate should be called a paper
value, and the canonical configuration remains unchanged.

1. **Weapon lethality group:** `D_hit,c4,c5`. Highest priority because all
   observed first-contact attempts hit and these values are entirely
   PAPER-UNSPECIFIED.
2. **Initial geometry group:** literal endpoint/team-center interpretation,
   formation spacing, initial heading/altitude/speed dispersion. It currently
   creates a deterministic 8 km, same-altitude head-on closure.
3. **Weapon lifecycle group:** cooldown, ammunition, in-flight missile, and
   miss/retry semantics. It does not cause the already-certain first volley but
   can strongly change off-angle cumulative probability and cleanup.
4. **Controller group:** low-level law, gains, rate/load/acceleration limits.
   These determine whether a learned command can evade or flank before 4 km.
5. **Launch minimum-range group:** `D_firemin`, currently zero and unpublished.
6. **Sensor and observation group:** `c1-c3,b1-b3`, normalization and dead-slot
   encoding. High epistemic uncertainty, but less direct evidence for the
   immediate mass exchange than groups 1-4.
7. **Episode horizon:** important for other strategies/timeouts, but not causal
   here because measured episodes end near step 92, far below 2,000.

## Final attribution judgment

Of the three required choices, the diagnostic supports:

**2. Mainly paper-unspecified environment assumptions.**

The algorithm is not absolved: the deterministic actor fails to create the
large angular/tactical separation needed for low-loss victory. But it has
learned a strong asymmetry relative to straight flight (89.5% wins and fewer
immediate Red deaths), whereas the remaining return/loss gap is tightly linked
to a qualitatively extreme environment regime produced by unpublished values:
near-certain 4 km lethality, fixed head-on geometry, no missile lifecycle, and
unpublished controller behavior. The paper's reported 0.6 Red losses is not
compatible with assuming that these current placeholders have been validated.
This is a causal-priority judgment from measured environment behavior, not
reverse parameter fitting to Figure 8 or Figure 9.
