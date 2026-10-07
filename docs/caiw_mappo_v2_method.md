# CAIW-MAPPO V2 development method

## Motivation and scientific question

IWSC V1 is closed as `IWSC_NOT_SUPPORTED`. Its state-only critic frequently collapsed, and its normalized one-step forecast revisions produced actor gradients whose scale was unrelated to the raw probability change. V2 asks a narrower question: can a reliable action-conditioned predictor provide useful, policy-relative counterfactual credit for preserving the ability to clear later waves without degrading tactical MAPPO?

The persistent trajectory is decomposed as `tau1 ⊕ tau2 ⊕ tau3`. Define `C2=1[Wave2 cleared]` and `C3=1[Wave3 cleared]`. Wave1 state-actions supervise separate `C2` and `C3` heads; Wave2 state-actions supervise `C3`; Wave3 has no inter-wave credit. Current-wave completion remains entirely in the frozen tactical reward/GAE path.

## Outcome critic

`InterWaveActionOutcomeCritic` consumes joint 52D observations, executed 3D bounded actions, alive masks, source-wave one-hot and remaining horizon. A `56→256→256` state pathway is added to a zero-initialized `3→256` action projection, followed by two-head self-attention and a `512→256→256→2` event network. It returns raw logits for Wave2-clear and Wave3-clear. The zero action path makes initial predictions action invariant and allows action sensitivity only when learned from data.

Training uses `BCEWithLogitsLoss` on three equal-weight tasks: W1→W2, W1→W3 and W2→W3. Every task is segment-balanced between positive and negative classes and is skipped unless both classes contain at least eight train segments. An entire completed episode is deterministically assigned to train or validation by `episode_group_id % 5`, preventing Wave1/Wave2 or state-level leakage.

Balanced training changes the class prior. For recent natural prevalence `p`, raw balanced logit `z` becomes the prior-corrected outcome probability `sigmoid(z + log(p/(1-p)))`, with `p` clipped to `[1e-4,1-1e-4]`. This is not claimed to be perfectly calibrated.

## Freshness and reliability

Replay stores the exact executed action, pre-tanh latent action and behavior log probability. Current-policy log probability is recomputed with the existing squashed-Gaussian implementation; a state is fresh only when every alive agent satisfies `log(.8) <= log π_current-log π_behavior <= log(1.2)`. Dead agents are ignored. Historical replay trains only the CAIW critic and never re-enters PPO. Validation segments never enter critic optimization or recent-prior estimation; recent natural prevalence is estimated only from train-split completed episodes, so readiness is genuinely held-out.

Every ten PPO updates, untouched validation segments are evaluated at segment level using the mean prediction over at least four fresh stored states. Each task independently requires at least 64 valid segments, 16 positives, 16 negatives, AUROC≥.60, and Brier skill strictly greater than zero for three consecutive checks. Any failed check resets readiness immediately. No temperature or bias calibrator is fit on validation.

## Policy-relative counterfactual action credit

Before the first actor optimizer step, the current rollout's old policy is frozen implicitly and all CAIW advantages are cached. For each alive agent, four alternative latent actions are generated as two antithetic Gaussian pairs around the old-policy mean using an independent NumPy RNG. Other agents' executed actions remain fixed. The baseline is the mean prior-corrected future quality under those alternatives:

`A_i^CAIW = Q(S,A) - (1/K) Σ_k Q(S,A_-i,a_i^k)`.

Wave1 averages only ready W1→W2/W1→W3 heads; Wave2 uses ready W2→W3; Wave3 is inactive. The result is per-agent credit, not a broadcast team scalar. It is cached with stop-gradient and reused across PPO epochs.

There is no mean subtraction, standard-deviation normalization, rank transform, sign transform, minimum-effect amplification, or beta sweep. Because probabilities are in `[0,1]`, the advantage naturally remains in `[-1,1]`; an effect of `.003` remains `.003`.

## Tactical-preserving fusion

The usual PPO/entropy gradient `gT` is unchanged. If `gT·gC<0`, only CAIW is projected. The projected auxiliary is then scaled so its norm is at most `.25||gT||`; when `||gT||<1e-12`, CAIW scale is zero. The gradients are combined and the existing global `.5` clip is applied, followed by exactly one actor optimizer step. Tactical critic optimization, reward, GAE, advantage normalization and the Plain actor learning-rate schedule are unchanged. A cold gate calls the original Plain update path exactly.

## Data lifecycle and checkpointing

Only real pre-action transitions from source waves 1/2 are retained. Segments cross rollout boundaries, retain first/last/clear rows when capped to 64, and are finalized only at true episode completion. Train class deques, natural validation deques, prior windows, gate streak/readiness, activation counters, critic/optimizer, independent RNG and gradient counters are checkpointed. On resume, environment state is not restored, so pending segments are discarded and counted; the completed episode counter is restored so future split identity remains stable.

## CTDE, relation to COMA, and limitations

The centralized action-conditioned critic and per-agent counterfactual baseline are related to COMA's credit-assignment idea, adapted to continuous squashed-Gaussian actions and future-wave events. Deployment remains decentralized: only the unchanged 52D actor is needed.

This method estimates a **policy-relative counterfactual action-conditioned future-outcome advantage**. It does not establish an oracle causal action effect. The critic remains a freshness-filtered recent-policy approximation, and counterfactual estimates still depend on function approximation and action-distribution support.

## Difference from V1

V1 used constant scalar segment targets, a state-only probability predictor, local `V(s')-V(s)`, unconditional per-wave standardization, and negative-dot projection only. V2 uses separate binary events, real joint actions, class-balanced BCE, recent-prior correction, held-out task-specific readiness, continuous per-agent old-policy counterfactuals, raw probability scale, and a `.25` auxiliary trust cap. V1 source, configuration, checkpoints and analysis remain untouched.
