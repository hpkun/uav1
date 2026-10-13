# MADSAC for Combat Environment v3.26

The current MADSAC implementation version is 3. Attention critic architecture
and console presentation are corrected as described below. All environment
sources/configs and replay semantics remain unchanged. The actor, SAC update
and loss/reduction formulas, targets and delayed/Polyak schedule are unchanged.
Trainer-file changes are limited to implementation identity and load validation.

Formal config: configs/madsac_5v5_v326.yaml. It explicitly declares environment
3.26/5v5 and network observation/action/agents=56/3/5. Actor/critic hidden layers
remain 256/256 with two attention heads. Training retains actor/critic lr=1e-4,
gamma=.99, alpha=.1, tau=.001, replay=1M, minibatch=1024, 24 envs, 3M transitions
and CUDA. learning_starts=10000, policy_delay=2 and one gradient step per vector
step remain project implementation choices rather than claimed original paper
parameters. Existing evaluation and checkpoint intervals are preserved.

The validator adds v3.26 while retaining supported historical protocols. The
v3.26 branch requires exact integer 56/3/5 dimensions, environment declaration
and formal architecture. Checkpoints still require implementation identity,
environment version, both config hashes and training seed. v3.26 additionally
validates architecture/provenance fields. Legacy 4v4/52D checkpoints are rejected,
including a test that forges new metadata but retains actual old actor weights.

Checkpoint extra records environment_version=3.26, 56/3/5, shared_actor=true,
twin_independent_critics=true, attention_heads=2 and actual hidden_dim. Formal
hidden_dim=256; the unchanged smoke mode uses hidden_dim=32 and explicitly marks
training_smoke=true. formal_exact_resume_supported=false and
replay_buffer_in_checkpoint=false remain intact: formal exact resume is not
supported because replay is omitted, as before.

## Implementation-v3 critic reconstruction

For each agent, e_i=embedding([o_i,a_i]). Each of the two heads uses q_i=W_q e_i,
k_j=W_k e_j and v_j=LeakyReLU(W_v e_j). Attention logits are q_i^T k_j, with no
sqrt(head_dim) division. All living keys, including self, enter softmax. Heads
aggregate sum_j alpha_ij v_j and concatenate. Q_i=MLP(context_i); no raw self
embedding is concatenated, so q_network's first input width is hidden_dim.

Dead keys get zero weight. Living query weights sum to one; dead query weights
and Q are zero. A single survivor attends to itself with weight one. All-dead
inputs produce zero weights/Q without NaN.

PAPER concepts followed here are centralized Q/K/V attention with the self term
in Eq.(16), g(W_v e), twin critics, target actor/critics and entropy SAC. Existing
alpha=.1 and two-head settings are retained. Paper-consistent reconstruction:
g is PyTorch LeakyReLU with slope .01 (not an exact paper-reported slope), and
unscaled dot-product follows the published Eq.(17) expression. This does not
assert that unpublished author code used this unique implementation. Project
choices learning_starts, gradient_steps_per_vector_step and policy_delay remain
explicitly separate.

MADSAC_IMPL_VERSION is 3. Protocol metadata adds attention_self_inclusive=true,
attention_value_activation=leaky_relu, attention_value_negative_slope=.01,
attention_logit_scaling=none and critic_raw_self_shortcut=false. All four online/
target critic first Q-layer weights must have shape [hidden_dim,hidden_dim].
Implementation-v2 checkpoints are rejected. Falsifying version=3 cannot hide an
old [hidden_dim,2*hidden_dim] Q layer: both protocol and direct evaluation loading
reject it. No weight migration is attempted.

## Combat-facing console

TRAIN uses the latest recent_episode_window completed episodes (formal default
100), not only the completions in the current vector step. Rates mean episodes:
win uses red_success; fire uses red_first_attempt_step, not fire-window; kill uses
red_first_kill_step. Timeout uses the shared episode_is_timeout helper. Losses,
length and team return are window means. Empty windows display n/a.

EVAL uses the corresponding existing evaluator aggregates. EVAL* appears only
when the unchanged checkpoint selection key improves. START shows experiment
identity/dimensions/seed/device/environment count/steps/width/heads/batch/alpha/
evaluation mode. CKPT is a compact progress message. Console intervals, training
schedule and evaluation selection logic are unchanged.

Formatting test examples (not a claim of 20k training in this validation):

```text
[START] MADSAC-v3 | env=v3.26 | 5v5 | seed=1 | cuda | envs=24 | steps=3.0M | hidden=256 | heads=2 | batch=1024 | alpha=0.1 | eval=stochastic
[TRAIN] 20k/3.0M (0.7%) | ep=3 | win=50% | Rloss=3.00 | Bloss=2.00 | fire=50% | kill=50% | timeout=50% | len=600 | R=-20.00
```

Actual smoke evaluation output:

```text
[EVAL*] 1.2k | stochastic | win=0% | Rloss=5.00 | Bloss=0.00 | fire=0% | kill=0% | timeout=0% | len=288 | R=-50.69
[CKPT] 1.2k | saved
```

Detailed training/optimization JSONL writers are unchanged. Loss, Q, entropy,
log-probability, gradient, action saturation and update-count fields remain
available, as do replay size/vector steps and detailed summary/checkpoint data.

Unit tests now cover [B,5,56] observations, [B,5,3] actions, [B,5] masks/Q and
[B,2,5,5] attention. Dead-agent influence, twin parameter independence, initial
target copies/frozen targets, Polyak updates, own-action-only gradients, delayed
actor updates, replay terminal transitions, evaluation RNG and checkpoint reload
are verified. A separate CUDA test instantiates and reloads actual 256-wide
formal networks.

Validation:

```text
python -m pytest tests -k madsac -q
python -m pytest tests/test_combat_protocol.py::test_factory_and_public_api_are_combat_only tests/test_combat_v326.py -q
```

65 MADSAC-related tests and 19 environment/protocol regressions passed. Python
compilation and diff whitespace checks passed. Protected environment and core
algorithm hashes are unchanged.

CUDA smoke (not formal training):

```text
python -u algorithm/train_madsac.py --env-config configs/combat_environment_v326.yaml --algorithm-config configs/madsac_5v5_v326.yaml --device cuda --seed 1 --num-envs 4 --total-sampled-steps 1200 --smoke --output-dir outputs/madsac_impl3_v326_smoke_20261013_172646
```

Completed 1200 environment transitions / 300 vector steps. 297 critic updates,
148 actor updates, replay size 256; every recorded loss and gradient norm is
finite. The smoke's existing effective parameters are hidden=32, replay=256,
batch=16 and learning_starts=8. These do not change formal YAML parameters.
Evaluation completed; checkpoints best_eval, checkpoint_1200, latest and final
were saved. The actual saved checkpoint passed official protocol validation and
the official CUDA recording-policy loader.

Artifacts: outputs/madsac_v3_validation/validation.json and the smoke directory.
No unresolved integration failures were observed. This verifies sampling/update/
evaluation/checkpoint compatibility, not successful combat learning from only
1200 transitions. No formal long training or version-control mutation was run.
