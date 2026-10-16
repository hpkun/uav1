# MADDPG-v3 on Combat Environment v3.38

This is an adaptation of the existing MADDPG implementation, not a new SAC variant.
Reference: Lowe et al. (2017), [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275).
**PAPER:** the reference supplies decentralized deterministic actors, centralized per-agent critics, replay and target networks. Its experimental appendix also explicitly reports lr=.01, gamma=.95, tau=.01, replay=1M, updates every 100 samples and a batch described as 1024 episodes; these are experimental settings, not universal algorithm requirements.

**OFFICIAL IMPLEMENTATION:** OpenAI's archived TensorFlow implementation samples 1024 replay transitions, uses distribution-parameter regularization and clips each individual parameter-gradient tensor to norm .5. Its continuous policy distribution parameterization differs from this project's deterministic tanh actor. See the [30-item source audit](maddpg_reference_alignment_audit.md) for primary source links and exact distinctions.

**PROJECT ADAPTATION:** v3.38 geometry, masks, tanh actions, networks, lr=1e-4, gamma=.99, tau=.001, warmup and OU protocol below are explicit project choices. This does not claim complete reproduction of the original experiments.

## Environment and networks

The environment is frozen: v3.38, four learning Red agents, 45D local observation and 3D continuous normalized action. Blue is a scripted opponent, not a fifth learner. No reward, weapon, observation, motion, boundary, initialization, termination or action semantics are changed.

There are four independently parameterized actors, four critics and four targets of each kind. Each actor is `45 → 256 ReLU → 256 ReLU → 3 tanh`, using only its own observation. Each critic is `192 → 256 ReLU → 256 ReLU → 1`. Its joint input concatenates `4×45=180` Red observation entries and `4×3=12` Red action entries; total **192**. There is no parameter sharing, attention, recurrent network, twin Q, entropy loss, policy delay or target smoothing.

## Mathematics and alive masks

For agent i, the target is

`y_i = r_i + gamma * (1-done) * next_alive_i * Q'_i(o', [mu'_j(o'_j)]_j)`.

All next actions come from target actors; each target critic supplies its own Q. `done` includes termination and truncation under the existing environment protocol. Dead observations and actions are replaced by zero using `torch.where` before centralized critic input, including target actions. A death-transition reward remains learnable when alive-before=1; post-death samples are excluded from that agent's loss. An all-dead minibatch for a particular agent skips that optimizer step; event counters and per-agent optimizer counts are recorded separately.

Actor i maximizes Q_i with **only action i replaced** by `actor_i(o_i)`. All other actions are the actually executed replay actions, detached. Other current actors are not called to regenerate actions. Critic parameters are frozen for the actor step and their previous `requires_grad` flags are restored afterward. Each gradient update trains all eligible critics, then all eligible actors, then one Polyak group: `target ← (1−tau)target + tau online`.

## Replay and behavior exploration

Joint replay stores `[B,4,45]` observations/next observations, `[B,4,3]` executed clipped actions, `[B,4]` local rewards and both alive masks, and `[B]` done flags. Auto-reset observations are never substituted for terminal next observations. Sampling uses a separate NumPy generator (`seed XOR 0x31A5`), whose state is recorded. OU uses its own independent per-environment/per-agent streams (`seed XOR 0x0A17`).

Training behavior is `clip(actor(o) + exploration_scale * OU, -1, 1)`, then dead actions are zeroed. Only noise states of completed environments reset. OU theta=.15, sigma=.20, mu=0; scale decays linearly 1.0→.1 over 3M environment transitions. Deterministic evaluation has no OU noise.

## Transition schedule and numerical provenance

Formal config: `configs/maddpg_4v4_v338.yaml`. Project settings: Adam actor/critic lr=1e-4, gamma=.99, tau=.001, batch=1024, replay=1M, warmup=10,000, interval=100, 16 environments, seed=1, 3M transitions. Hidden width, alive masks and normalized actions are environment adaptations; all numerical values, warmup, schedule and exploration decay are project experimental choices.

`sampled_steps` counts **environment transitions**, not agent transitions or vector steps. For a vector batch spanning `(previous, current]`, newly eligible credit is `max(0, current-max(previous, learning_starts))`. Only transitions strictly after warmup count. While credit≥100, subtract 100 once and trigger one event, then perform `gradient_updates_per_event` gradient updates. **Every gradient update samples a fresh minibatch.** Never discard overshoot or accumulate a warmup backlog. The explicit reference default is `gradient_updates_per_event=1`; the diagnostic moderate setting is 4. Sixteen parallel environments increase throughput, not UTD.

With 16 environments, six eligible vector steps give credit=96 and no event; the seventh gives one event and remainder=12. Six more give one event and remainder=8. Theoretical events at 3M: `(3,000,000−10,000)//100 = 29,900`, remainder=0. With g=1 this gives 29,900 actor and critic gradient groups (UTD=.0099667); g=4 gives 119,600 each (UTD=.0398667). These are group counts, not four times that number; actual per-agent optimizer counts can be lower when an entire minibatch is dead. Nominal UTD ignoring warmup is g/100: .01 or .04. This arithmetic does not start formal training.

Each gradient update also updates targets. At tau=.001 the target half-life is about 693 gradient updates. Changing g from 1 to 4 changes both optimization intensity and target speed per environment transition: one event blends .001 versus `1-(1-.001)^4=.003994`. A/B therefore does not isolate gradient intensity while holding target speed constant.

## Gradient clipping and actor regularization

Both actor and critic steps compute pre-clip global L2 norm, execute `torch.nn.utils.clip_grad_norm_(module.parameters(), .5, error_if_nonfinite=True)`, then compute the actual post-clip norm before the optimizer step. Clipping scope is **one agent's complete actor or critic**, independently of other agents. This requested global norm scope differs from OpenAI's per-parameter-tensor norm scope. Pre/post arrays and fractions of eligible agent updates clipped are logged; legacy `actor_grad_norms`/`critic_grad_norms` remain pre-clip aliases.

The official `p_reg` penalizes flattened policy-distribution parameters (for continuous Box, mean and log standard deviation). There is no strictly equivalent mapping to a deterministic tanh action penalty or weight decay. It is **OFFICIAL-ONLY / NOT PORTED**. Actor loss remains `-E[Q_i]`; no regularizer is silently added.

## Protocol and checkpoints

v3.38 config explicitly declares `environment_version: '3.38'` and `team_size: 4`. Matching dimensions alone are insufficient; v3.37 paired with this config is rejected. Dimension registrations belong to `algorithm/maddpg/protocol.py`, not Trainer's algorithm logic.

Implementation version is **3**. v1 and v2 checkpoints are rejected before loading weights because real clipping and the multi-gradient schedule change the training contract. Historical YAML is preserved; legacy `gradient_steps_per_vector_step` cannot start a v3 run. New configs must explicitly declare `gradient_clip_norm` and `gradient_updates_per_event`. There is no automatic migration or reinterpretation.

Checkpoints record config SHA256, architecture, seed/env count/budget, dimensions, gamma/tau, sampled-step units, interval, warmup, replay/batch sizes, remainder, OU state, replay RNG, evaluations and update counters. The v3 contract adds clip norm/scope, gradients per event, nominal/observed actor and critic UTD, trigger-event count and immutable initial parameter references. Root and extra schedule identity, event/gradient arithmetic and reference sizes are validated. Actor/critic/target list sizes must all be N. `formal_exact_resume_supported=false` and `replay_buffer_in_checkpoint=false`: load is for evaluation/audit; `resume()` explicitly fails. Saved optimizer/RNG state does not imply exact training resume.

## Evaluation and logging

Formal checkpoint selection uses the same 20 development seeds `10,000,000…10,000,019` as MAPPO/MAAC, without exploration. Train/eval seed ranges are checked disjoint. `evaluation_selection_key` ranks win rate, return, then fewer Red losses. New best is `[EVAL*]`; other evaluations `[EVAL]`. This is a selection set, not an independent final test set.

Console: `[START]`, `[TRAIN]`, `[EVAL*]`/`[EVAL]`, `[CKPT]`, `[DONE]`. Train combat statistics use the most recent 100 completed episodes (mixed recent behavior policies); Qloss/Aloss use the most recent actual update. `fire` means Red attempt episode rate, not fire-window geometry rate. No completed episodes or no update prints `N/A`, never `None`/`nan`.

Machine logs retain the common combat aggregator's full output rather than only console fields:

- `training_metrics.jsonl`: one vector-step record, completed-this-step episode statistics (JSON null if none), completed total, sampled/vector steps, replay size, reward and R1–R4 means, exploration scale, remainder, trigger and gradient counts, UTD, alive-coordinate OU RMS/absolute mean, scaled noise RMS and fraction of actor+noise coordinates clipped at ±1. Completed episode means are not the console rolling window; weight them by completed episode counts.
- `optimization_metrics.jsonl`: exactly one record per actual **gradient update**. It includes trigger-event index and gradient index within event; pre/post clip norms/flags; each agent's alive count/fraction; active-only Q and target mean/std/min/max; TD signed/absolute mean/max; deterministic heading/pitch/speed mean/std and saturation at .95/.99, globally and per agent. Legacy global `target_mean`/`target_abs_max` include all slots; use `target_*_per_agent` for active-only comparisons. At the first and every 100th gradient update it adds actor/critic initial-reference drift and online-target L2 distance. With g=4 there are four rows per trigger and four fresh minibatches.
- `evaluation_history.csv`: full common evaluator metrics, exact actual sampled steps, deterministic mode and seed provenance.
- `run_config.json`: effective protocol and config hashes, with the requested config saved separately for smoke/CLI provenance.
- `initial_diagnostics.json`: initial zero drift and online-target distance. `run_summary.json`: status, final steps/replay/remainder, best/latest evaluations, trigger/gradient/per-agent counts, UTD, fresh final parameter distances and full protocol.

## Validation and short smoke

Run repository checks in the `uav` Conda environment. CUDA is mandatory for training and checkpoint/evaluation audits. CPU tensor unit tests test algebra only. Use WSL Ubuntu if available, otherwise Windows PowerShell `uav`.

```powershell
python -m pytest tests/test_maddpg.py -q
python -u algorithm/train_maddpg.py --env-config configs/combat_environment_v338.yaml --algorithm-config configs/maddpg_4v4_v338.yaml --device cuda --seed 1 --num-envs 16 --smoke --output-dir outputs/maddpg_v338_validation/smoke
python -u tools/validate_maddpg_v338_smoke.py
```

Smoke uses 8,192 transitions, batch=64, replay=4096, warmup=128 and two final evaluation episodes, preserving the real environment, 256-width networks and update mathematics. Expected events=80, vector steps=512, remainder=64 (g=1). The source YAML remains formal. Output directories must be fresh. The old `tools/validate_maddpg_smoke.py` contains v1-era expectations; use the v338 validator for v3. CUDA tests include fixed-batch critic overfit, frozen-critic actor improvement, real clipped SGD parameter deltas and Adam moment/parameter checks.

The source-alignment diagnosis uses equal 300k budgets, seed=1, 16 environments, initial evaluation and six 50k evaluations on the same 20 development seeds. A changes only clipping relative to v2; B additionally sets g=4. Gamma, lr, tau, batch, networks, noise and environment stay fixed. Conditional C (lr=3e-4, tau=.005, g=4) is authorized only if A and B both show no credible learning signal. Initial references and code/config hashes make comparisons auditable. Evaluation seed reuse is for diagnostics/selection, not an independent final test. No formal multi-million-step run is part of this change.

## Structural comparison

| Algorithm | Actor | Critic/training distinction |
|---|---|---|
| MADDPG-v3 | Four independent deterministic local actors; OU only for behavior | Four centralized scalar MLP Qs; joint replay; target actors/critics; own-action gradient; explicit clipping/update contract |
| MAPPO | Stochastic local policy with deterministic evaluation | On-policy clipped policy objective and state-value critic; no DDPG replay target |
| MAAC | Shared stochastic continuous policy in this project's implementation | Attention Q and counterfactual policy-gradient baseline with entropy |
| MADSAC | Stochastic actors in its own protocol | Entropy-regularized off-policy target/critic structure; not inherited by MADDPG |
