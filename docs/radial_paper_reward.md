# Radial paper-inspired per-agent MADDPG reward (Interception/12)

This is an **isolated experiment**, not a change to the existing
`Interception/11` radial benchmark, replays, physical dynamics or old
checkpoints. It follows the distinction between per-agent reward channels
in Lowe et al. (MADDPG, 2017) and the self/teammate/team outcome reward
categories in Liu et al. (TM-MADDPG, 2026). The numeric weights below are
**experimental choices**, not a reproduction of Liu et al.'s coefficients.

## Reward actually learned

Let `c_i(t)` be 1 if defender i makes its one allowed capture at t,
`C(t)=sum_i c_i(t)`, `F(t)` be the number of newly failed attacker targets,
and `W(t)` be 1 exactly once when all five attackers are caught without
any failures. For each defender i, including defenders already physically
retired, the replay reward is:

```text
r_i(t) =
    0.2 * (assigned_distance_before_i - assigned_distance_after_i)
    - 2 * out_of_bounds_i
    + 20 * c_i(t)
    + 5 * (C(t) - c_i(t))
    - 50 * F(t)
    + 50 * W(t)
```

Distance and boundary terms are **zero after physical retirement**.
Teammate capture, target failure and team success remain available as
team-result learning signals even if defender i has retired.

We preserve the original assigned-target distance definition rather than
silently changing the reward target to the nearest attacker (as described
in Liu et al., Equation 22). In the old benchmark, the explicit matching
algorithm still supplies the Actor's target. Removing matching requires a
separate observation/algorithm experiment.

The new Critic i learns `r_i` over the **complete team episode** and stops
bootstrapping only when the environment episode terminates. A retired
defender's action is always zero and its Actor is never updated at retired
states. Thus no defender can capture a second attacker; Critic learning
and physical action eligibility are deliberately separated.

## Reward accounting / terminal output

The new `team_return` (shown as `return_sum_agents`) is the
**undiscounted sum of all five agents' individual rewards**:

```text
team_return = sum_t sum_i r_i(t)
```

The reward components on the console are on this exact scale, so they
must sum to `return_sum_agents`. Example: for one capture at time t,
`capture_self=+20`, `capture_teammate=+20` (four peers × +5). For one
failed target, the reward vector contains −50 for each of five agents:
the episode reward **sum** contributes −250 even though the environment
counts only *one* failed target. For a 5/5 clean victory,
`all_captured=+250` across all five reward channels.

This is intentional: it is the **sum of learning rewards**, not the
count-once physical event ledger. Event counts are still separately
reported through `captured_count` and `penetrated_count`.

The old `Interception/11` environment continues to use
`team_return = sum_t (global_reward + mean_i local_i)`. Results from
the two reward contracts must not be compared by raw Return magnitude.

Individual Critic targets use discounted returns (`gamma=0.99`) and
are not numerically equal to undiscounted episode returns.

## Run and verification

To train the isolated profile:

```bash
python -m experiment.cli train \
  --env-config configs/interception_radial_paper_reward.yaml \
  --algorithm-config configs/maddpg_low_level_interval10.yaml \
  --algorithm maddpg --attack-mode radial_straight \
  --total-sampled-steps 2000000 --num-envs 1 \
  --seed 7 --device cuda --eval-episodes 30 \
  --log-every-steps 10000 --output-dir runs/radial_paper_reward_seed7
```

Run focused regression tests before a long experiment:

```bash
python -m pytest -q tests/test_radial_paper_reward.py tests/test_radial_straight.py tests/test_per_target_breach.py
```

**Controlled comparisons**: Match initial seeds, environment geometry,
motion limits, Actor observations, optimizer settings, replay strategy and
total sampled steps. Report both capture rate and all-captured rate,
Critic TD errors and Q calibration; do not treat a positive raw reward or
a lower TD loss as proof of improved interception.

## Sources

- Lowe et al. (2017), *Multi-Agent Actor-Critic for Mixed Cooperative-
  Competitive Environments*, Section 4.1, Equations (5)–(6).
- OpenAI `maddpg/experiments/train.py`, where the episode reward is the
  sum of rewards over agents and individual reward channels are retained.
- Liu et al. (2026), *UAV swarm interception strategy based on temporal
  memory MADDPG*, Acta Aeronautica et Astronautica Sinica, Section 3.2,
  Equations (21)–(25). Its Equation (22) has a sign ambiguity; we retain
  positive reward for decreasing distance, consistent with its prose.
