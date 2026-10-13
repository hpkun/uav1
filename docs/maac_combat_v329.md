# Continuous MAAC for Combat Environment v3.29

This is an independent implementation of the MAAC structures and equations in
Iqbal & Sha, **Actor-Attention-Critic for Multi-Agent Reinforcement Learning**
(ICML 2019), adapted to the existing continuous-action UAV environment. It does
not import the MADSAC trainer or reuse its networks, losses or replay class.
The environment, MAPPO and MADSAC implementations are unchanged.

Sources:

- Paper: https://proceedings.mlr.press/v97/iqbal19a/iqbal19a.pdf
- Supplement: https://proceedings.mlr.press/v97/iqbal19a/iqbal19a-supp.pdf
- Public author implementation: https://github.com/shariqiqbal2810/MAAC

The title *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
Environments* refers to Lowe et al.'s MADDPG (2017), not this MAAC paper.
Author code is public. We do not claim exact replication of that code or its
discrete benchmark experiments.

## Confirmed protocol

The user superseded the initial on-policy, shared deterministic actor draft
with the paper protocol, independent actors, Gaussian+tanh continuous policies,
Eq.(8) score-function policy gradients and Monte Carlo counterfactual baselines.

- CTDE over the five learning Red agents; their local 56D observations already
  contain the existing Blue-related observation fields. Blue remains the
  unchanged fixed environment policy; it is not an additional learned actor.
- Five independent local Gaussian actors, each with its own Adam optimizer.
  Local observation alone is sufficient for execution. Tanh produces 3D actions.
- One centralized multi-agent attention critic, with five separate observation-
  action encoders and Q heads. Shared Q/K/V projections have four heads and
  scaled dot-product attention. Self is excluded; dead agents are also excluded.
- Value projection has LeakyReLU. Head i receives concatenated own embedding
  and attention context, as in paper Eq.(5). This is not a context-only head.
- One target attention critic and five target actors, initialized identically
  to their online counterparts and frozen. No twin Q, minimum-Q or delayed actor
  mechanism is added.
- Off-policy joint replay, retaining the agent axis and true transition-next
  observation/alive mask before the vector environment's automatic reset.

## Losses and Monte Carlo baseline

Eq.(7), with UAV terminal/death masking:

```text
a'_i ~ target_pi_i(o'_i)
y_i = r_i + gamma * (1-done) * alive'_i
                   * (target_Q_i(o',a') - alpha*log target_pi_i(a'_i|o'_i))
L_Q = mean_batch sum_i alive_i * (Q_i(o,a) - y_i)^2
```

Both termination and the environment's finite-horizon timeout end bootstrap.
Death stops an agent's future value but does not discard its immediate death
reward if it was alive at transition start. The entropy correction is inside
the discounted, terminal-masked bootstrap, following the paper equation.

For each policy update, joint actions are freshly sampled from **all current
policies**, not taken from replay's stored behavior actions. Before any actor
optimizer step, draw independent alternative actions from each policy to form:

```text
b_i(o,a_-i) = (1/K) sum_k Q_i(o,(alternative_a_i[k],a_-i))
L_pi_i = mean_live log pi_i(a_i|o_i)
                   * stop_gradient(alpha*log pi_i(a_i|o_i) - (Q_i-b_i))
```

Only i's action is replaced for baseline estimation; other sampled actions are
held fixed. Baseline samples are separate from the selected policy-gradient
action. K=16 is an explicit engineering choice for the paper's continuous-policy
sampling option, not a reported paper hyperparameter.

Gaussian samples use `Normal.sample()`, not `rsample()`. The pre-tanh action is
held fixed when evaluating its log probability. Q, baseline and the coefficient
are detached, so the update is the Eq.(8) likelihood-ratio/score gradient, not
the reparameterized SAC `alpha*log_prob-Q` action-path gradient. Tanh's log
Jacobian is computed in a stable form without inverse-tanh of saturated actions.

## Appendix parameter configuration

| Parameter | Formal value |
|---|---:|
| hidden dimension | 128 |
| attention heads | 4 |
| activation | LeakyReLU |
| actor/critic learning rate | 0.001 / 0.001 |
| gamma | 0.99 |
| alpha | 0.01 |
| tau (online weight) | 0.005 |
| parallel environments | 12 |
| joint replay capacity | 1,000,000 |
| replay minibatch | 1,024 |
| minimum sampled transitions between cycles | 100 |
| critic/policy updates per cycle | 4 / 4 |

Each update samples a separate replay minibatch; all critic updates precede
all policy updates, then targets update once per cycle. Counts refer to
environment transitions, not agent transitions. With 12 synchronous workers,
the 100-transition threshold is crossed at 108 transitions; the counter resets
after a cycle as in supplementary Algorithm 1.

Target interpolation uses `target=(1-tau)*target+tau*online`, with tau=.005.
The supplement's Algorithm 1 and its prose use conflicting tau conventions;
this implementation follows the prose's .005 online weight and author code.

## Explicit adaptation and engineering choices

- UAV environment remains v3.29, 5v5, 56D/3D, with its own 1000-step maximum;
  the appendix's 100-step benchmark reset is not imposed on this environment.
- Continuous Gaussian mean/log-std heads and log-std clipping [-5,2] replace
  the author's discrete benchmark policy representation.
- Replay warmup is 1024 transitions, sufficient to sample the first full batch.
- Counterfactual K=16, total budget 3M, evaluation seeds and checkpoint/log
  schedules are project choices; none are mislabelled as appendix values.
- Actor gradient clipping .5, critic clipping 50 and averaging shared attention
  parameter gradients across five tasks follow author-code choices. The loss
  equations do not add its discrete-policy regularizers or critic weight decay.
- Dead-agent masking is an environment adaptation. A lone survivor has zero
  other-agent context but still gets Q from its own embedding. All-dead rows
  return finite zero Q/attention.
- Chunked replay preserves a one-million capacity while allocating storage
  lazily, rather than allocating over 2GB when a short run starts.

## Identity, checkpoint and logs

MAAC implementation version is 2 and variant is
`paper_maac_continuous_eq8_mc_v2`. Strict validation includes environment version,
56/3/5 dimensions, architecture, variant and environment/algorithm hashes.
The previous version-1 deterministic draft checkpoint is rejected, as are
v3.28, 4v4/52D, MAPPO, MADSAC and mismatched smoke/formal checkpoints.

Checkpoints save online and target networks, optimizers and update counters.
Loading is supported for evaluation/recording. Exact training resume is not
offered: replay, environment and full RNG continuation are not saved. Metadata
explicitly sets `formal_exact_resume_supported=False` and
`replay_buffer_in_checkpoint=False`.

START/TRAIN/EVAL console lines are compact and omit entropy. JSONL includes
actor/critic loss, gradient norms, replay size, cycles, return/win/fire/kill,
controller diagnostics and the existing firing-geometry funnel. `fire_rate`
means at least one actual Red attempt per episode; `kill_rate` means at least
one Red weapon kill. Evaluation JSON also keeps the distinct fire-window rate.
MAPPO's existing console fire-window metric is not silently redefined.

## Commands

Use WSL Ubuntu with `conda activate uav`; if WSL is unavailable use Windows
PowerShell/uav. Training, evaluation and checkpoint recording require CUDA.

```bash
python -m pytest tests/test_maac.py -q
python -u algorithm/train_maac.py --env-config configs/combat_environment_v329.yaml --algorithm-config configs/maac_5v5_v329.yaml --device cuda --seed 1 --num-envs 2 --total-sampled-steps 32 --smoke --output-dir outputs/maac_v329_validation/cuda_smoke_paper_v2
python algorithm/evaluate_maac.py --checkpoint outputs/maac_v329_validation/cuda_smoke_paper_v2/latest.pt --episodes 2 --output outputs/maac_v329_validation/evaluation_paper_v2.json
```

Output directories must be fresh. Smoke uses width=32, batch=16, replay=128,
K=4 and update threshold=4 to exercise the same learning paths quickly; formal
config values remain unchanged. Formal-width CUDA unit tests also exercise
128/4 heads, batch=1024 and K=16. Smoke results only verify execution and
numerical health, not learning performance. No formal long training is started.
