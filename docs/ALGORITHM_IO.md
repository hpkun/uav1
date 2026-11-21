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

The defender active mask has shape `[5]` and is currently all ones: defenders are not destroyed. It remains part of the learner interface and is separate from the visible-bit inside observation slots. Captured attackers disappear from subsequent attacker slots.

## MAPPO

| Item | Shape | Meaning |
|---|---|---|
| Actor input | `[B,5,49]` | Each agent's local observation; one shared actor network |
| Actor output | `[B,5,2]` | Physical acceleration vectors |
| Central critic input | `[B,245]` | Concatenated defender observations |
| Central critic output | `[B]` | Team value `V(s)` |
| Learning reward | `[B]` | `r_team = r_global + mean_i(r_local,i)` |
| Rollout extras | `[B,5]` log probabilities, `[B,5,2]` raw actions | PPO likelihood and update contract |

`terminated` and `truncated` both set `done=1`. A rollout stores the actual terminal observation separately from the next episode's reset observation; GAE does not bootstrap across either episode boundary.

For the current scenario configuration, `r_global` is zero. The approach term uses defender displacement toward a pre-step visible attacker position, so an attacker moving closer while the defender stands still produces no approach reward.

New MAPPO runs seed PyTorch before network initialization and stochastic action sampling. The value critic predicts a normalized team return using running return statistics; rollout values and GAE remain on the original reward scale. Value clipping uses the normalization statistics from the rollout, before those statistics are updated with the new returns. New checkpoints save the running statistics and CPU/CUDA RNG states. Historical MAPPO checkpoints without ValueNorm state retain their original unnormalized critic behavior. Checkpoints still do not contain parallel worker states or unfinished rollouts, so they are not exact training resumes.

## MADDPG

| Item | Shape | Meaning |
|---|---|---|
| Actor `i` input | `[B,49]` | Defender `i` local observation; five separate actors |
| Actor `i` output | `[B,2]` | Defender `i` physical acceleration |
| Critic `i` input | `[B,245]` state and `[B,5,2]` joint action | 255 numeric inputs total |
| Critic `i` output | `[B]` | `Q_i(s,a_0,...,a_4)` |
| Learning reward | `[B,5]` | `r_i = r_global + r_local,i` for each critic |

Replay stores each defender's local observation, the joint state and action, all five individual rewards, the actual terminal next observation/state, `done`, and active masks. Actor `i` updates through critic `i`, replacing only its own action in the current-policy joint action; the other four actions are detached. This cooperative current-policy choice differs from the original MADDPG replay-other-actions actor gradient. The five critics must not receive a broadcast team reward. Checkpoints using this reward interface have schema `interception_maddpg_2`.

The airbase-interception paper lists both MAPPO and MADDPG as using global plus local rewards, but does not specify how its MAPPO baseline aggregates the local components. The scalar team value above is the conditional value of the shared team reward given the common centralized state; duplicating that same state and target five times would not create distinct agent credit assignment.

## Dual-critic shared policy

For comparison, the shared deterministic actor also takes `[B,5,49]` and emits `[B,5,2]`. Its two centralized critics use the 245-dimensional state and 10-dimensional joint action, and receive two scalar learning signals: `r_global` and `mean_i(r_local,i)`.

With the current configuration, the dual-critic global signal is identically zero. This baseline needs a separate global reward definition before it is useful for new training runs.
