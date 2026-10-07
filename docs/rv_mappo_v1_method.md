# RV-MAPPO V1

RV-MAPPO V1 is a minimal follow-up to the DAWE V1 `SAFETY_FAIL` and its
fixed-bank KL decomposition. DAWE's `rho=0.25` mechanically magnified the
mean-KL component by approximately `1/rho^2=16`; RV therefore restores the
source policy's standard exploration scale and performs no rho sweep.

For each training seed, the actor at sampled step 1,505,280 is copied into a
fully frozen reference actor. The stochastic behavior distribution is
`Normal(mu_current(o), sigma_reference(o))`. Only the reference scale is used;
the reference mean is ignored. The current mean and shared backbone remain
trainable. The current log-standard-deviation head parameters are frozen and
its state-dependent output is diagnostic only. Those frozen parameters remain
in the actor Adam parameter groups deliberately, preserving the complete
source optimizer state; `grad=None` prevents Adam from updating them.

Rollout sampling, old/new log-probabilities and the squashed Monte-Carlo entropy
all use that same RV distribution. Deterministic deployment remains exactly
`tanh(mu_current(o))` and does not use the reference actor.

RV checkpoints embed the complete reference actor, its hash, source checkpoint
hash, source step, source seed and module state. A strict resume therefore does
not depend on the original source file and fails closed if reference state is
missing or mutated.

V1 has no KL guard, variance regularization, learnable gate, wave-specific
variance, recurrent/context/attention mechanism, curriculum, replay or reward
change. The matched 300k screen is development evidence only, not final paper
evidence and not proof that RV improves performance.
