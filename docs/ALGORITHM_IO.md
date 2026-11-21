# MAPPO and MADDPG input/output contract

This contract describes `Interception/1`. One transition is a decision by five defenders. The five attackers are scripted and never occupy policy action slots.

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
