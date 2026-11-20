# MAPPO and MADDPG input/output contract

This contract describes `Interception/1`. One transition is a decision by five defenders. The five attackers are scripted and never occupy policy action slots.

## Defender observation and action

Each defender receives `o_i: float32[36]`, in this exact order:

| Slice | Content | Scale |
|---|---|---|
| `0:2` | Own world position `(x,y)` | divide by 100 m |
| `2:4` | Own world velocity `(vx,vy)` | divide by 5 m/s |
| `4:6` | Base position minus own position | divide by 200 m |
| `6:21` | Three nearest defender slots | each: relative position / 200 m, relative velocity / 8 m/s, valid bit |
| `21:36` | Three nearest active attacker slots | same five fields per slot |

Defender slots are visible within 60 m; attacker slots are visible within 80 m. Sort each group by distance and then agent number. An unavailable slot is all zero, including its valid bit. The centralized state `s: float32[180]` is `o_0 || o_1 || ... || o_4`; it contains joint local observations and no privileged physical oracle.

Both actors emit world-frame acceleration `a_i=(ax,ay)` in m/s². The executed vector lies in the radius-2 disk. There is no heading, firing, or discrete action. At evaluation, the action is deterministic. At training, MAPPO samples its Gaussian latent before the disk transform; MADDPG adds exploration noise to its deterministic action and projects it to the disk.

The defender active mask has shape `[5]` and is currently all ones: defenders are not destroyed. It remains part of the learner interface and is separate from the visible-bit inside observation slots. Captured attackers disappear from subsequent attacker slots.

## MAPPO

| Item | Shape | Meaning |
|---|---|---|
| Actor input | `[B,5,36]` | Each agent's local observation; one shared actor network |
| Actor output | `[B,5,2]` | Physical acceleration vectors |
| Central critic input | `[B,180]` | Concatenated defender observations |
| Central critic output | `[B]` | Team value `V(s)` |
| Learning reward | `[B]` | `r_team = r_global + mean_i(r_local,i)` |
| Rollout extras | `[B,5]` log probabilities, `[B,5,2]` raw actions | PPO likelihood and update contract |

`terminated` and `truncated` both set `done=1`. A rollout stores the actual terminal observation separately from the next episode's reset observation; GAE does not bootstrap across either episode boundary.

## MADDPG

| Item | Shape | Meaning |
|---|---|---|
| Actor `i` input | `[B,36]` | Defender `i` local observation; five separate actors |
| Actor `i` output | `[B,2]` | Defender `i` physical acceleration |
| Critic `i` input | `[B,180]` state and `[B,5,2]` joint action | 190 numeric inputs total |
| Critic `i` output | `[B]` | `Q_i(s,a_0,...,a_4)` |
| Learning reward | `[B,5]` | `r_i = r_global + r_local,i` for each critic |

Replay stores each defender's local observation, the joint state and action, all five individual rewards, the actual terminal next observation/state, `done`, and active masks. Actor `i` updates through critic `i`, replacing only its own action in the joint action. The five critics must not receive a broadcast team reward. Checkpoints using this reward interface have schema `interception_maddpg_2`.

## Dual-critic shared policy

For comparison, the shared deterministic actor also takes `[B,5,36]` and emits `[B,5,2]`. Its two centralized critics use the 180-dimensional state and 10-dimensional joint action, and receive two scalar learning signals: `r_global` and `mean_i(r_local,i)`.
