# MADDPG reference alignment audit

Audit baseline: project MADDPG-v2, before the changes in this task. Environment v3.38 is frozen. `Current` below describes that baseline; the final decisions are implemented in v3 and evaluated separately. This is classic deterministic CTDE adaptation, not a claim of exact reproduction of the published MPE code.

Sources: [PAPER, Sections 4.1/5.2 and Appendix Algorithm 1/Experimental Results](https://arxiv.org/html/1706.02275v4); [OFFICIAL trainer](https://github.com/openai/maddpg/blob/master/maddpg/trainer/maddpg.py); [OFFICIAL clipping helper](https://github.com/openai/maddpg/blob/master/maddpg/common/tf_util.py); [OFFICIAL distributions](https://github.com/openai/maddpg/blob/master/maddpg/common/distributions.py); [OFFICIAL experiment defaults](https://github.com/openai/maddpg/blob/master/experiments/train.py); [OFFICIAL uniform replay](https://github.com/openai/maddpg/blob/master/maddpg/trainer/replay_buffer.py).

| Item | Paper | Official implementation | Current | Classification | Decision |
|---|---|---|---|---|---|
| 1 Actor locality | Own observation at execution | Local input | Own 45D observation | MATCH | Keep |
| 2 Independent actors | Separate policies | Agent scopes | Four independent MLPs | MATCH | Keep |
| 3 Joint critic input | Joint observations/actions, Eq.5–6 | Concatenation | 180+12=192D | MATCH | Keep |
| 4 Independent critics | Q_i learned separately | Per-agent Q | Four scalar Qs | MATCH | Keep |
| 5 Target actors | Target policies | Target policy scopes | Independent deep copies | MATCH | Keep |
| 6 Target critics | Target Q | Target Q scopes | Independent deep copies | MATCH | Keep |
| 7 Bellman target | All target policies, own target Q | Same dependency | Same, plus continuation mask | MATCH | Keep |
| 8 Actor DPG | Own-action derivative, Eq.5 | Own sampled action replaces placeholder | Deterministic own action | PROJECT-ADAPTATION | Keep deterministic objective |
| 9 Other actor actions | Replay a_-i in sampled gradient | Replay placeholders | Detached replay actions | MATCH | Keep; no joint current-policy regeneration |
| 10 Replay transition | Joint experience | Aligned per-agent buffers | Single joint buffer | PROJECT-ADAPTATION | Equivalent alignment; keep uniform |
| 11 Terminal next state | Next state in experience | Before reset | transition_next_observations | MATCH | Preserve, test actual terminal data |
| 12 done/truncated | Terminal convention not spelled out in Eq.6 | Uses done, ignores separate episode-length flag | done OR truncated | PROJECT-ADAPTATION | Keep finite-horizon v3.38 episodic semantics |
| 13 Agent death | No aircraft death-slot specification | No alive-mask equivalent | next_alive_i suppresses bootstrap; alive-before loss mask | PROJECT-ADAPTATION | Preserve death reward; exclude post-death samples |
| 14 Polyak direction | tau online +(1−tau) target | Same direction | Same | MATCH | Keep |
| 15 Update order | Critic, actor, targets | Q then policy then targets | All critics, all actors, targets | MATCH | Independent parameters make grouping consistent |
| 16 Update frequency | Appendix: every 100 added samples | t modulo 100 | Every 100 eligible transitions with remainder | PROJECT-ADAPTATION | Preserve cadence; expose repeats per event |
| 17 Replay sampling | Random minibatch | Uniform indices with replacement | Uniform joint sampling, separate RNG | MATCH | Fresh minibatch for every gradient update |
| 18 Exploration | Additive random process in Algorithm 1 | Distribution-dependent sampling | Additive OU then bounded clipping | PROJECT-ADAPTATION | Preserve OU parameters |
| 19 Gradient clipping | Threshold/scope unspecified | 0.5 per parameter tensor | Norm measured, never clipped | LIKELY-BUG | Add real per-agent global norm clipping=.5; scope difference documented |
| 20 Actor regularization | Not part of Eq.5 objective | Distribution flat-parameter penalty | None | OFFICIAL-ONLY | NOT PORTED; keep −E[Q_i] |
| 21 Optimizer | Appendix: Adam | Adam | Adam | MATCH | Keep |
| 22 Initialization | Exact tensors unspecified | TF layer defaults | PyTorch Linear defaults; target hard-copy | PROJECT-ADAPTATION | Keep fixed across candidates |
| 23 Learning rate | Appendix experiment setting .01 | Default .01 | 1e−4 | PROJECT-ADAPTATION | Keep A/B; only conditional C uses 3e−4 |
| 24 Gamma | Appendix experiment setting .95 | Default .95 | .99 | PROJECT-ADAPTATION | Keep |
| 25 Tau | Appendix experiment setting .01 | Online weight .01 | .001 | PROJECT-ADAPTATION | Keep A/B; assess lag, not label as error |
| 26 Batch size | Appendix says 1024 episodes | Actually samples 1024 transitions | 1024 transitions | PROJECT-ADAPTATION | Explicit transition units |
| 27 Warmup | Separate length unspecified | batch×episode length (default 25,600) | 10,000 | PROJECT-ADAPTATION | Keep |
| 28 Replay size | Appendix 1e6 | 1e6 | 1e6 transitions | MATCH | Keep |
| 29 Target frequency | Soft update with optimization | Each optimization | Each gradient update | MATCH | Keep; B has more target steps per environment time |
| 30 Action clipping | Bound handling unspecified | Depends on action distribution/environment | tanh + OU then clip [-1,1], dead=0 | PROJECT-ADAPTATION | Keep |

## Three distinct levels

**PAPER:** deterministic CTDE, independent policies/Qs, own-action gradient, all target actions and replay. The Appendix actually reports lr=.01, gamma=.95, tau=.01, buffer=1e6 and 100-sample cadence; these are its experimental settings, not universal MADDPG requirements. Section 5.2 uses 64-width MLPs unless specified otherwise (some Appendix tasks use 128). Do not call these numbers paper-unspecified, nor prescribe them unchanged for air combat.

**OFFICIAL CODE:** the archived repository warns it was restructured after publication. Box actions use a diagonal Gaussian distribution; discrete actions use a differentiable soft categorical sampler. The policy penalty acts on distribution parameters (Box mean and log standard deviation), not directly on bounded deterministic executed actions. It is not mathematically equivalent to adding action-squared loss or weight decay to this project's tanh actor. **OFFICIAL-ONLY / NOT PORTED.**

**PROJECT ADAPTATION:** 4v4, 45/3 local dimensions, 256-width networks, deterministic tanh policy, OU exploration, alive masks, finite-horizon timeout terminal convention, lr=1e−4, gamma=.99, tau=.001, warmup=10000. None of these numerical differences alone is a demonstrated bug. User-requested `clip_grad_norm_` clips the global norm of one agent network; official `minimize_and_clip` clips each parameter tensor separately. The threshold is shared, the scope is not.

## Pre-change evidence and predeclared diagnostic plan

Available `outputs/maddpg_v2_v338_seed1_3m` is incomplete: no run_summary/latest/final; observed training rows stop near 1.954M transitions, evaluation at 1.9M. This audit does not complete or resume it. Its loaded best checkpoint is at 800,000 steps (10% selection win); evaluation win rates observed are 0–10%. No current-process status can be inferred from file completeness alone.

Recorded actor gradient norms never exceeded .5 in available v2 updates, while per-agent critic norms exceeded .5 in roughly 43–67% of updates. Missing clipping is real, but it cannot alone be asserted to explain all learning failure. Actor/critic nominal UTD is 1/100=.01 per eligible transition; at 3M and warmup=10k there would be 29,900 full gradient-update groups, not 16 times that. MAAC's configured four critic/policy steps per approximately 100 transitions are stronger; its runner discards overshoot (12 envs typically 108 transitions/cycle), so observed ratios must also be distinguished from nominal values.

Implementation v3 adds clipping and diagnostics, while keeping J_i=E[Q_i]. A and B both use 300,000 transitions, seed=1, identical original environment/config settings, initialization, 16 environments and 20 fixed evaluation seeds. A uses one gradient update/event; B uses four, all fresh minibatches. Initial evaluation and 50k evaluation cadence are shared diagnostics, not changes to the environment. Formal config retains one update/event; B is separately named project calibration. No checkpoint resume or multi-million-step training.

Learning is assessed by active-batch Q/TD distributions, pre/post clipping, gradient/parameter motion, actor saturation, noise clipping, alive sample counts, target distances, plus evaluation return. Fixed-batch critic overfit and frozen-critic actor improvement are prerequisite tests, not claims of solving air combat. Only if both A/B show no credible learning signal will a separate C use lr=3e−4, tau=.005 and four updates/event, with the same budget and seeds. Do not select C merely from a low win rate.

B also makes four Polyak steps per 100 transitions: effective per-event online weight is 1−(.999)^4≈.003994 vs A's .001. Thus A/B compares update intensity including its target cadence, not a pure critic-gradient count holding target speed fixed. Initial references and RNG state are preserved for audit; metadata does not claim exact resume.
