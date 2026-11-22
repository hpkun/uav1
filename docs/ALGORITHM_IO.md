# MAPPO and MADDPG input/output contract

One transition is a decision by five defenders. The five attackers are scripted and never occupy policy action slots. `Interception/2` adds target allocation; `Interception/1` remains available for existing checkpoints.

## Interception/2: assigned target interface

Use `configs/interception_assignment.yaml` for distance matching or `configs/interception_assignment_nearest.yaml` for independent nearest selection. Each active defender receives `float32[54]`:

| Slice | Content | Scale |
|---|---|---|
| `0:2` | Own absolute world position | arena half extent, 100 m by default |
| `2:4` | Own absolute world velocity | defender speed maximum, 5 m/s by default |
| `4:24` | Four visible teammates, nearest first; relative position, relative velocity, valid bit per slot | position / sensing radius; velocity / (defender + attacker speed maxima) |
| `24:49` | Five visible attackers in the same format | same scale |
| `49:54` | Assigned attacker relative position, relative velocity, valid bit; zero when unassigned or currently unseen | same scale |

Unseen and retired agents have zero slots. Retired defenders receive a zero observation. The allocator in `assignment/distance.py` recomputes at step zero, every `assignment_period` steps (20 by default), when an attacker is detected for the first time by *any* defender, or after capture or penetration. It does not replan for a new defender-to-attacker visibility edge, a temporary loss of sensing, or a changed distance ranking. It maintains a system-level detected-target mask across the episode. At a trigger, only edges currently within the defender's sensing radius are eligible. The primary method maximizes the number of one-to-one pairs, then minimizes their total Euclidean distance; an exact five-agent dynamic-programming solver gives the same objective result as distance-based Hungarian matching without requiring SciPy. The optional independent-nearest method uses the same triggers but can send multiple defenders to one attacker. The previous per-edge-triggered nearest method remains available only for checkpoints that saved that historical configuration.

The training-only critic state is `float32[75]`: five defenders and five attackers, each with `(x,y,vx,vy,alive)` in stable agent order (50 values), then the row-major 5×5 defender-to-attacker assignment matrix (25 values). Positions are divided by arena half extent and velocities by the side's speed maximum. Retired UAV rows are zero except for the zero alive bit. In the distance-matching method, each assignment-matrix row and column has at most one `1`. MAPPO receives this state and emits one scalar team value. Each MADDPG critic receives the same state plus five acceleration actions and emits its own Q value. The 25 assignment indicators follow Jia et al.'s centralized physical state plus assignment matrix design. The selected target's five relative features in the actor observation are this project's concrete way to expose the assignment; they are not a quoted paper dimension. No additional 12 script or time features are included.

Both algorithms retain their existing sampling and learning loops. The global reward is `20 × captures − 100 × first penetration`, and each defender's local reward is `0.05 × (distance to its pre-step assigned target before movement − distance to that same target after movement) − 2 × its boundary violations`. An unassigned defender gets zero approach reward. The MAPPO team signal adds the mean local reward to the shared global reward; MADDPG critic `i` uses global plus local reward `i`. The assignment is updated only after the movement and reward for that step. The unchanged position update uses the old velocity, then acceleration changes velocity; consequently the direct effect of an acceleration on distance shaping begins on the next step. Capture retires the matched defender and attacker. All attackers must be captured or penetrated before the episode ends.

The historical `Interception/1` interface is described below.

## Interception/1: historical interface

## Defender observation and action

For new runs, each defender receives `o_i: float32[49]`, in this exact order:

| Slice | Content | Scale |
|---|---|---|
| `0:2` | Own world position `(x,y)` | divide by 100 m |
| `2:4` | Own world velocity `(vx,vy)` | divide by 8 m/s |
| `4:24` | All four other defender slots | each: absolute world position / 100 m, absolute world velocity / 8 m/s, valid bit |
| `24:49` | All five active attacker slots | same five fields per slot |

Both friendly and attacker slots are visible within the same 80 m sensing radius. Sort each group by distance and then agent number. An unavailable slot is all zero, including its valid bit. Every position and velocity is absolute in the world frame; the base is absent from actor observations. The centralized state `s: float32[245]` is `o_0 || o_1 || ... || o_4`; it contains exactly the five actor observations, including their visibility masks. MAPPO's value critic receives this 245-vector. Each MADDPG critic receives the same 245-vector plus all five 2-D actions (255 numeric inputs). Historical runs retain their saved relative layouts (`36` or `51` per actor) and their 60 m friendly communication range.

Both actors emit world-frame acceleration `a_i=(ax,ay)` in m/s². The executed vector lies in the radius-2 disk. There is no heading, firing, or discrete action. At evaluation, the action is deterministic. At training, MAPPO samples its Gaussian latent before the disk transform; MADDPG adds exploration noise to its deterministic action and projects it to the disk.

The defender active mask has shape `[5]`. In the current `resolve_all_one_penalty_trade` scenario, a capture removes exactly one defender and one attacker. Retired defenders have mask zero, receive a zero observation, cannot move or capture again, and disappear from friendly observation slots. Both policies emit zero actions for masked defenders. Historical protocols keep defenders active after capture. The active mask is separate from each observation slot's visible bit.

## MAPPO

| Item | Shape | Meaning |
|---|---|---|
| Actor input | `[B,5,49]` | Each agent's local observation; one shared actor network |
| Actor output | `[B,5,2]` | Physical acceleration vectors |
| Central critic input | `[B,245]` | Concatenated defender observations |
| Central critic output | `[B]` | Team value `V(s)` |
| Learning reward | `[B]` | Shared capture/breach events plus mean defender shaping |
| Rollout extras | `[B,5]` log probabilities, `[B,5,2]` raw actions | PPO likelihood and update contract |

In new runs, each attacker is removed when captured or penetrated. Each capture also retires its matched defender; simultaneous contacts are assigned by shortest distance with agent IDs resolving ties, and each defender can capture at most one attacker. The episode terminates after all five attackers are resolved; unresolved attackers at the step limit count as penetrations. Historical first-breach runs can still truncate at the step limit. A rollout stores the actual terminal observation separately from the next episode's reset observation; GAE does not bootstrap across either episode boundary.

For new runs, `r_global` is `20 × captures − 100 × first-penetration-indicator` at each step and is shared by all defenders. The penetration indicator can activate only once per episode, even when several attackers breach or remain unresolved at the time limit. The local term contains boundary discipline and actual distance reduction to the nearest attacker visible before the step, using both UAVs' before/after positions. Historical `resolve_all` checkpoints retain their per-attacker penetration penalty; older first-breach checkpoints keep their saved behavior.

New MAPPO runs seed PyTorch before network initialization and stochastic action sampling. The value critic predicts a normalized team return using running return statistics; rollout values and GAE remain on the original reward scale. Value clipping uses the normalization statistics from the rollout, before those statistics are updated with the new returns. New checkpoints save the running statistics and CPU/CUDA RNG states. Historical MAPPO checkpoints without ValueNorm state retain their original unnormalized critic behavior. Checkpoints still do not contain parallel worker states or unfinished rollouts, so they are not exact training resumes.

## MADDPG

| Item | Shape | Meaning |
|---|---|---|
| Actor `i` input | `[B,49]` | Defender `i` local observation; five separate actors |
| Actor `i` output | `[B,2]` | Defender `i` physical acceleration |
| Critic `i` input | `[B,245]` state and `[B,5,2]` joint action | 255 numeric inputs total |
| Critic `i` output | `[B]` | `Q_i(s,a_0,...,a_4)` |
| Learning reward | `[B,5]` | Shared capture/breach events plus defender `i`'s shaping for critic `i` |

Replay stores each defender's local observation, the joint state and action, all five individual rewards, the actual terminal next observation/state, `done`, and active masks. Actor `i` updates through critic `i`, replacing only its own action in the current-policy joint action; the other four actions are detached. This cooperative current-policy choice differs from the original MADDPG replay-other-actions actor gradient. The five critics must not receive a broadcast team reward. Checkpoints using this reward interface have schema `interception_maddpg_2`.

The airbase-interception paper lists both MAPPO and MADDPG as using global plus local rewards, but does not specify how its MAPPO baseline aggregates the local components. The later DT-GAT-MARL paper places capture and breach events in the shared task reward and uses individual distance shaping. The scalar team value above is the conditional value of the shared team reward given the common centralized state; duplicating that same state and target five times would not create distinct agent credit assignment.

## Dual-critic shared policy

For comparison, the shared deterministic actor also takes `[B,5,49]` and emits `[B,5,2]`. Its two centralized critics use the 245-dimensional state and 10-dimensional joint action, and receive two scalar learning signals: `r_global` and `mean_i(r_local,i)`.

With the current configuration, the dual-critic global signal is identically zero. This baseline needs a separate global reward definition before it is useful for new training runs.
