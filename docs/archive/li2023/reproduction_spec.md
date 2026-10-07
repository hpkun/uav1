# Archived Li et al. (2023) reproduction evidence table

The 2023 repository PDF remains the normative source. The author-team 2022
predecessor is used only for missing simulator conventions. Fine-grained source
decisions are recorded in `docs/parameter_provenance.md`; `UNSPECIFIED` below
still means an executable placeholder, never a paper value.

## Module-by-module evidence

| Paper module | Paper location | Original meaning | Equation / table / figure | Code location | Status |
|---|---|---|---|---|---|
| Dynamics | Section 2.1, pp.4-5 | Six-state NED point-mass kinematics and dynamics; all models solve at 0.1 s | Eq.(1)-(2), Fig.1 | `src/uav_combat/dynamics.py`, `integrator.py` | PAPER; RK4 is UNSPECIFIED |
| Controller | Section 2.1, p.5; Section 3.3.2, p.13 | Controller maps desired `[psi_d,theta_d,v_d]` to `[phi,nz,nx]`, but its law is not published | Eq.(2), Eq.(23) | `src/uav_combat/controller.py` | Mapping PAPER; controller law/limits/gains UNSPECIFIED |
| Sensor | Section 2.2, p.5 | Position and attitude each use one printed clipped Gaussian epsilon shared by their three components; speed uses one clipped Gaussian sample | Eq.(3)-(5) | `environment/sensor.py` | Formula PAPER; coefficients UNSPECIFIED |
| Geometry | Section 2.3, p.6 | ATA/AA/HA/HCA describe the red-blue 3-D relation | Fig.2, Eq.(6) | `environment/geometry.py` | AA/HA/HCA PAPER; ATA resolution DERIVED due to paper inconsistency |
| Weapon | Section 2.4, pp.6-7 | Automatic launch requires ATA, HA and range gates; hit requires both noisy angular inequalities | Eq.(7)-(8), Table 1 | `environment/weapon.py` | Formula and maxima PAPER; `D_firemin,D_hit,c4,c5` UNSPECIFIED |
| Blue fixed strategy | Section 2.5, p.7 | At every simulation step select the nearest surviving Red UAV, pure-pursue it, switch immediately when nearest changes, and use the weapon model | Fig.4 | `environment/fixed_policy.py`, `env.py` | PAPER; low-level pursuit speed UNSPECIFIED |
| MADSAC architecture | Section 3.2, pp.10-11; Section 4.1, p.15 | Shared stochastic actor; two independent centralized attention critics; two attention heads; actor and critic use two 256-unit hidden layers | Fig.6, Eq.(16)-(17) | `madsac/actor.py`, `attention_critic.py` | PAPER; activation, log-std bounds and exact layer wiring UNSPECIFIED |
| MADSAC objectives | Section 3.2.2-3.2.3, p.11 | Per-agent `r_i` target; target actor and target double critics; entropy term; independent critic losses; each `Q_i` policy term differentiates only through `a_i` | Eq.(18)-(21) | `madsac/trainer.py` | PAPER; summing own-action terms into one shared-actor gradient is DERIVED |
| Scenario | Section 3.3.1, pp.12-13 | 4 Red vs 4 Blue in a 10 km diameter area, initialized at opposite ends of a randomly selected diameter; leaving the area means death | Fig.7, Table 1 | `environment/scenario.py`, `env.py` | Counts/diameter/opposite ends/boundary death PAPER; formation details UNSPECIFIED |
| Action | Section 3.3.2, p.13 | Actor produces `[delta_psi,delta_theta,delta_v]`, added to current state to obtain desired state | Table 2, Eq.(23) | `controller.py`, `configs/paper_environment.yaml` | PAPER |
| Observation | Section 3.3.3, p.14 | Own `(pe,vo,phi,psi,theta)`, three teammates `(pe,vo,psi,theta)`, four enemies `(dr,vo,AA,ATA,HA)`; transformed under observer body coordinates | Eq.(24) | `environment/observation.py` | Fields PAPER; 45-D count DERIVED; full teammate position transform supported by 2022 Eq.(17); scalar/dead encoding UNSPECIFIED |
| Reward | Section 3.3.4, pp.14-15 | `R=R1+R2+R3+R4`; `R4` is one piecewise choice between R41 and R42 | Eq.(25) | `environment/reward.py`, `env.py` | Coefficients/thresholds PAPER; multi-enemy target and overlap handling UNSPECIFIED |
| Training | Algorithm 1, p.12; Section 4.1, p.15 | M parallel environments, `T+=M`, threshold triggers n critic updates; delayed branch performs n actor updates and then target updates; 24 distinct parallel seeds; >8M samples | Algorithm 1 | `training/runner.py`, `vector_env.py` | Structure/M=24/>8M PAPER; threshold/n/d and mapping of `t` UNSPECIFIED |
| Evaluation | Section 3.3.1, p.13; Section 4.1, p.15 | 20 test seeds completely different from training; average 20 results; test once every two training cycles; five runs and 95% CI | Fig.8-9 | `training/evaluator.py`, `scripts/aggregate_training_runs.py` | Counts/runs/CI PAPER; cycle-to-step mapping and CI estimator UNSPECIFIED |

## Equation (6) resolution and physical truth table

Figure 2 places `o1` at Red, `o2` at Blue and `o3` at Blue's horizontal projection. The printed first line of Eq.(6) uses `-V²_xy` for ATA, which would make ATA algebraically identical to the printed AA line and conflicts with the ATA arc drawn at Red. The implementation therefore uses the Figure 2/air-combat meaning:

- `LOS = atan2(y_target-y_own, x_target-x_own)`
- `ATA = wrap(LOS - psi_own)`
- `AA = wrap(psi_target - LOS)`; no reverse LOS
- `HA = atan2(-(z_target-z_own), horizontal_distance)` in NED
- `HCA = wrap(psi_target-psi_own)`

The sign convention is UNSPECIFIED; the paper reward and weapon equations use absolute ATA/AA/HA. The following expectations are physical, not self-referential tests:

| Case | Geometry | ATA_r | AA_r | ATA_b | AA_b | HA_r |
|---|---|---:|---:|---:|---:|---:|
| A | Red directly behind Blue, same heading | 0 | 0 | pi | pi | 0 |
| B | Head-on | 0 | pi | 0 | pi | 0 |
| C | Red directly at Blue's side, same heading | pi/2 | -pi/2 | -pi/2 | pi/2 | 0 |
| D | Blue directly behind Red, same heading | pi | pi | 0 | 0 | 0 |
| E | Case A with Blue 100 m above and 100 m ahead | 0 | 0 | pi | pi | pi/4 |

## Equation (25) exact operational form

- `R1`: +10 per Blue UAV destroyed by this Red UAV; -10 if this Red UAV is destroyed by attack.
- `R2`: -10 if this Red UAV leaves the engagement area.
- `R3`: +0.001 when `|ATA_r|<=30°`, `|HA_r|<=30°`, and `d_r>=4000 m`; otherwise 0.
- `R4`: one piecewise term. If `|AA_r|<=30°` and `d_r<=4000 m`, use R41. Otherwise, if `|AA_b|<=30°` and `d_b<=4000 m`, use R42. It is never `R41+R42`.
- R41/R42 check the strongest `5°` tier first, then `15°`, then `30°`. This is DERIVED from the three intended nested strength levels; the printed cases overlap and do not state precedence.
- If both outer R4 cases hold, the first printed R41 case takes precedence. This is deterministic formula-order handling of an UNSPECIFIED edge case.
- At exactly 4000 m, R3 and R4 can both contribute because they are separate terms and both printed inequalities are inclusive.

For multiple Blue UAVs, the paper does not identify the enemy used by each local R3/R4. This reproduction uses the nearest surviving Blue UAV and its reciprocal Blue-centered geometry. `env.step()` returns the four distinct Eq. (25) values `[r_1,...,r_4]`; replay and Eq. (18) preserve that vector. “All agents use the same reward function” means the homogeneous agents use the same Eq. (25), not that the four realized values are summed and broadcast. This local interpretation is DERIVED from the Markov-game definition and Eq. (18), with medium-high confidence. Target selection, pre-attack geometry timing, and simultaneous application of same-step hits remain UNSPECIFIED.

For logging only, each episode accumulates four per-agent returns. The Figure-8
`average_return` is the sum of those four episode returns; the mean is also
logged as `average_agent_return`. The team sum is DERIVED and is not stated as
an explicit implementation detail by the paper.

## Equations (18)-(21) gradient semantics

- Eq. (18) consumes the replay vector element `r_i` for target `y_i`; target
  actor actions remain a complete joint next action under `torch.no_grad()`.
- For the agent-i Eq. (21) term, only action `a_i` remains attached to the
  shared actor graph. Every `a_j`, `j != i`, is held constant before evaluating
  `Q_i(s, a_1,...,a_4)`.
- Eq. (19) and Eq. (20) use `mean_batch(sum_alive_agents(value))`. The replay
  batch is the expectation dimension; the agent dimension is the printed sum.
  A sample with no alive agent contributes zero without changing the batch
  denominator.
- Because all actors share parameters, the summed alive own-action gradients
  accumulate into one shared actor optimizer step. Dead agents contribute no
  actor term. Mean Q and entropy remain alive-slot means because they are
  diagnostics rather than backward objectives.

## Formal success metric

The paper states: Red wins if it destroys all Blue UAVs; otherwise the mission fails and Blue wins. Formal evaluation therefore uses:

`red_success = (number of surviving Blue UAVs == 0)`

`win_rate = successful Red episodes / all episodes`

Timeout with any surviving Blue is failure. `termination_reason` is diagnostic only. For same-step mutual elimination, the literal all-Blue-destroyed condition is applied, so it is Red success; this rare edge case is UNSPECIFIED because the paper does not state it explicitly.

## Algorithm 1 scheduler

Each synchronous vector step samples M transitions and performs `T += M`. When `T >= steps_per_update` and replay has a minibatch:

1. run `update_steps_n` critic updates;
2. if global vector step `t mod policy_delay_d == 0`, run `update_steps_n` actor updates and then soft-update actor and both critic targets once;
3. a critic-only trigger does not update any target network;
4. set `T=0`.

Current UNSPECIFIED values are `steps_per_update=24`, `update_steps_n=1`, and `policy_delay_d=2`. `algorithm1_t_counter: global_vector_step` records the current minimal implementation assumption. Algorithm 1 visually places `t` inside an episode, but early termination/auto-reset semantics are absent, so this mapping remains STILL-UNSPECIFIED.

## Current high-impact UNSPECIFIED values

- Horizon: 2000 steps.
- Formation: center distance 4000 m from origin, 150 m same-team spacing, altitude 3000 m, speed 225 m/s.
- Sensor: `c1=10`, `c2=0.01`, `c3=1`, `b1=b2=b3=3`; shared epsilon is used exactly as printed.
- Weapon: `D_firemin=0`, `D_hit=2000`, `c4=c5=0.05`. These can make near, well-aligned shots nearly deterministic and require sensitivity analysis before claims about Figure 8/9.
- Controller: `nx[-3,3]`, `nz[-6,6]`, yaw rate 1 rad/s, pitch rate 0.7 rad/s, acceleration 50 m/s², proportional gains all 1.
- Observation: own absolute NED position; full yaw-pitch-roll `F_g -> F_b` relative teammate positions; global teammate yaw/pitch remain an unresolved scalar convention; enemy distance and signed AA/ATA/HA; position divided by 5000 and speed by 300; fixed ID slots; dead slots zeroed.
- Actor uses Eq.(24) observation only. Figure 6(a) depicts a previous action input while Algorithm 1 and policy equations write policy as a function of observation; the paper does not resolve this inconsistency.
- Actor and critic ReLU are supported by the 2022 author work but not stated by 2023; actor log-std clamp `[-5,2]`, exact embeddings/head split/final MLP remain UNSPECIFIED.
- Reward target selection and same-step timing described above; critic rewards are now per-agent Eq. (25), while the logged Figure-8 team sum is DERIVED.
- Evaluation-cycle mapping: `assumed_sampled_steps_per_training_cycle=50,000`, so the paper's “every two training cycles” is operationally every 100,000 sampled steps. The 50,000 mapping is STILL-UNSPECIFIED and is not a paper parameter.
- Five-run CI uses a two-sided Student-t interval with 4 degrees of freedom.
- Checkpoint every 500,000 sampled steps. Replay is not saved; resumed training does not preserve replay continuity.
- Runtime `scheduler_update_blocks` counts Algorithm-1 update-block triggers. It is deliberately not named `training_cycles` and is saved/restored in checkpoint metadata. Resume restores networks, optimizers and counters, but starts with empty replay and freshly reset episodes, so it is not bitwise-exact continuation.
- Seed rule: training seed is `base_seed + episode_index*M + env_id`; evaluation uses `10,000,000..10,000,019`, with assertions against overlap/reuse.
- Evaluation currently uses deterministic mean/tanh actor actions. The paper does not state deterministic versus stochastic test execution, so this remains UNSPECIFIED.
- Every value in `configs/sensitivity_candidates.yaml` is candidate-only and must be applied one group/profile at a time; none is a paper parameter.
- The canonical code semantics may be frozen for a pilot while these explicit parameter uncertainties remain; a pilot does not promote placeholders to paper values.
