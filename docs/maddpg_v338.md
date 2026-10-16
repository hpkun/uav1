# MADDPG-v2 on Combat Environment v3.38

This is an adaptation of the existing MADDPG implementation, not a new SAC variant.
Reference: Lowe et al. (2017), [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275).
The reference supplies the decentralized deterministic actors, centralized per-agent critics, replay, target networks and behavior exploration structure. Numerical settings below are project choices, not claimed original paper hyperparameters.

## Environment and networks

The environment is frozen: v3.38, four learning Red agents, 45D local observation and 3D continuous normalized action. Blue is a scripted opponent, not a fifth learner. No reward, weapon, observation, motion, boundary, initialization, termination or action semantics are changed.

There are four independently parameterized actors, four critics and four targets of each kind. Each actor is `45 → 256 ReLU → 256 ReLU → 3 tanh`, using only its own observation. Each critic is `192 → 256 ReLU → 256 ReLU → 1`. Its joint input concatenates `4×45=180` Red observation entries and `4×3=12` Red action entries; total **192**. There is no parameter sharing, attention, recurrent network, twin Q, entropy loss, policy delay or target smoothing.

## Mathematics and alive masks

For agent i, the target is

`y_i = r_i + gamma * (1-done) * next_alive_i * Q'_i(o', [mu'_j(o'_j)]_j)`.

All next actions come from target actors; each target critic supplies its own Q. `done` includes termination and truncation under the existing environment protocol. Dead observations and actions are replaced by zero using `torch.where` before centralized critic input, including target actions. A death-transition reward remains learnable when alive-before=1; post-death samples are excluded from that agent's loss. An all-dead minibatch for a particular agent skips that optimizer step; event counters and per-agent optimizer counts are recorded separately.

Actor i maximizes Q_i with **only action i replaced** by `actor_i(o_i)`. All other actions are the actually executed replay actions, detached. Other current actors are not called to regenerate actions. Critic parameters are frozen for the actor step. Each event updates all eligible critics, then all eligible actors, then one Polyak group: `target ← (1−tau)target + tau online`.

## Replay and behavior exploration

Joint replay stores `[B,4,45]` observations/next observations, `[B,4,3]` executed clipped actions, `[B,4]` local rewards and both alive masks, and `[B]` done flags. Auto-reset observations are never substituted for terminal next observations. Sampling uses a separate NumPy generator (`seed XOR 0x31A5`), whose state is recorded. OU uses its own independent per-environment/per-agent streams (`seed XOR 0x0A17`).

Training behavior is `clip(actor(o) + exploration_scale * OU, -1, 1)`, then dead actions are zeroed. Only noise states of completed environments reset. OU theta=.15, sigma=.20, mu=0; scale decays linearly 1.0→.1 over 3M environment transitions. Deterministic evaluation has no OU noise.

## Transition schedule and numerical provenance

Formal config: `configs/maddpg_4v4_v338.yaml`. Project settings: Adam actor/critic lr=1e-4, gamma=.99, tau=.001, batch=1024, replay=1M, warmup=10,000, interval=100, 16 environments, seed=1, 3M transitions. Hidden width, alive masks and normalized actions are environment adaptations; all numerical values, warmup, schedule and exploration decay are project experimental choices.

`sampled_steps` counts **environment transitions**, not agent transitions or vector steps. For a vector batch spanning `(previous, current]`, newly eligible credit is `max(0, current-max(previous, learning_starts))`. Only transitions strictly after warmup count. While credit≥100, sample a **fresh minibatch**, run one update event and subtract 100; never discard overshoot or accumulate a warmup backlog.

With 16 environments, six eligible vector steps give credit=96 and no update; the seventh gives one update and remainder=12. Six more give one update and remainder=8. A large vector batch can cause multiple updates. Theoretical formal events at 3M: `(3,000,000−10,000)//100 = 29,900`, remainder=0; this arithmetic does not start training. Actor/critic event counts are group counts, not four times that number; actual per-agent optimizer counts are separate.

## Protocol and checkpoints

v3.38 config explicitly declares `environment_version: '3.38'` and `team_size: 4`. Matching dimensions alone are insufficient; v3.37 paired with this config is rejected. Dimension registrations belong to `algorithm/maddpg/protocol.py`, not Trainer's algorithm logic.

Implementation version is **2**. v1 checkpoints are rejected before loading weights. Historical YAML is preserved, and its legacy `gradient_steps_per_vector_step` schedule cannot start a v2 run; create an explicitly declared transition-schedule config for a new experiment. There is no automatic migration or reinterpretation.

Checkpoints record environment and algorithm config SHA256, network architecture/counts, training seed/env count/budget, dimensions, gamma/tau, sampled-step units, interval, warmup, replay/batch sizes, remainder, OU type/state, replay RNG, evaluation mode/history and all update counters. Schedule identity is checked at both root and extra metadata; counts and remainder must match warmup/interval arithmetic. Actor/critic/target list sizes must all be N. `formal_exact_resume_supported=false` and `replay_buffer_in_checkpoint=false`: load is for evaluation/audit; `resume()` explicitly fails. Saved optimizer/RNG state does not imply exact training resume.

## Evaluation and logging

Formal checkpoint selection uses the same 20 development seeds `10,000,000…10,000,019` as MAPPO/MAAC, without exploration. Train/eval seed ranges are checked disjoint. `evaluation_selection_key` ranks win rate, return, then fewer Red losses. New best is `[EVAL*]`; other evaluations `[EVAL]`. This is a selection set, not an independent final test set.

Console: `[START]`, `[TRAIN]`, `[EVAL*]`/`[EVAL]`, `[CKPT]`, `[DONE]`. Train combat statistics use the most recent 100 completed episodes (mixed recent behavior policies); Qloss/Aloss use the most recent actual update. `fire` means Red attempt episode rate, not fire-window geometry rate. No completed episodes or no update prints `N/A`, never `None`/`nan`.

Machine logs retain the common combat aggregator's full output rather than only console fields:

- `training_metrics.jsonl`: one vector-step record, completed-this-step episode statistics (JSON null if none), completed total, sampled/vector steps, replay size, per-step reward and environment-provided R1–R4 means, exploration scale, remainder, actor/critic event counts. Completed episode means are not the console rolling window; consumers must weight by completed episode counts.
- `optimization_metrics.jsonl`: exactly one record per actual event, step/replay/exploration/remainder, mean and per-agent losses/grad norms/Q means, target mean/max and counters. If multiple events occur at one vector step, their sampled step is identical but their event counts and fresh minibatches differ.
- `evaluation_history.csv`: full common evaluator metrics, exact actual sampled steps, deterministic mode and seed provenance.
- `run_config.json`: effective protocol and config hashes, with the requested config saved separately for smoke/CLI provenance.
- `run_summary.json`: status, final steps/replay/remainder, best step/evaluation, latest evaluation, event/per-agent counts, final optimization metrics and full protocol.

## Validation and short smoke

Run repository checks in the `uav` Conda environment. CUDA is mandatory for training and checkpoint/evaluation audits. CPU tensor unit tests test algebra only. Use WSL Ubuntu if available, otherwise Windows PowerShell `uav`.

```powershell
python -m pytest tests/test_maddpg.py -q
python -u algorithm/train_maddpg.py --env-config configs/combat_environment_v338.yaml --algorithm-config configs/maddpg_4v4_v338.yaml --device cuda --seed 1 --num-envs 16 --smoke --output-dir outputs/maddpg_v338_validation/smoke
python -u tools/validate_maddpg_v338_smoke.py
```

Smoke uses 8,192 transitions, batch=64, replay=4096, warmup=128 and two final evaluation episodes, preserving the real environment, 256-width networks and update mathematics. Expected events=80, vector steps=512, remainder=64. The source YAML remains formal. Output directories must be fresh. The old `tools/validate_maddpg_smoke.py` contains v1-era 57-event expectations and is not a v2 validator; historical v1 audits require the historical implementation. Use the v338 validator for v2. The training CLI now defaults to the v3.38 environment and new v2 config. No formal 3M or multi-seed run is part of this change.

## Structural comparison

| Algorithm | Actor | Critic/training distinction |
|---|---|---|
| MADDPG-v2 | Four independent deterministic local actors; OU only for behavior | Four centralized scalar MLP Qs; joint replay; target actors/critics; own-action gradient |
| MAPPO | Stochastic local policy with deterministic evaluation | On-policy clipped policy objective and state-value critic; no DDPG replay target |
| MAAC | Shared stochastic continuous policy in this project's implementation | Attention Q and counterfactual policy-gradient baseline with entropy |
| MADSAC | Stochastic actors in its own protocol | Entropy-regularized off-policy target/critic structure; not inherited by MADDPG |
