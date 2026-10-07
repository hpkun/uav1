# BRSC-MAPPO V1

BRSC-MAPPO V1 (Boundary-Redistributed Segment Credit MAPPO) is a training-only development method for the frozen `persistent_wave_v2` task. The deployed actor remains the unchanged 52D, two-layer Plain MAPPO actor; BRSC changes neither environment reward nor tactical critic targets.

## Scientific motivation

IWSC used stepwise state-quality differences and did not produce a supported method. CAIW then tested action-conditioned counterfactual step credit. Its identifiability audit found that zeroing or shuffling the action barely damaged future-outcome prediction and that the identifiable action contribution was only about 0.04%–0.24%. The final CAIW critic was therefore primarily a state-quality predictor. This closes the single-step action-credit route while retaining evidence that wave-entry state quality predicts later completion.

BRSC consequently removes the action pathway entirely. At a real `spawned_next_wave` event it records the post-spawn `transition_next_observations`, next alive mask, source wave, and remaining horizon. The action, raw action, and behavior log-probability are not boundary-critic inputs. A two-head state-only critic predicts conditional completion of Wave 2 and Wave 3.

## Supervision and isolation

Episode completion supplies `C2 = [waves_cleared >= 2]` and `C3 = [waves_cleared >= 3]`. Wave-1 entries supervise W1-to-W2 and W1-to-W3; Wave-2 entries supervise W2-to-W3. Episodes with `episode_group_id % 5 == 0` are validation episodes. Their boundaries never enter training replay or the train-only natural-prior window. Bounded replay makes the critic a recent-policy approximation, not an oracle fixed-policy value function.

Balanced BCE-with-logits training uses equal task weighting. Balanced class sampling is corrected at scoring time with the recent train-split natural prevalence. Readiness is held out and task-specific: coverage, AUROC at least 0.60, positive Brier skill, and three consecutive passes are required; any failed validation deactivates the task.

## Causal update order

At update entry, the current rollout is scored with the pre-update boundary critic, readiness flags, and priors. Plain tactical PPO or fused BRSC PPO then runs. Only afterwards are newly completed episode labels ingested, the boundary critic trained, and held-out validation performed. Thus a trajectory label cannot train the critic and immediately credit the same trajectory; new readiness applies only to the next update.

## Boundary-anchored on-policy segment credit

For a ready boundary, raw credit is `Z = prior-corrected quality - natural prior`. There is no centering, standardization, ranking, sign transform, threshold, or amplification. The team scalar is redistributed backward inside the maximal contiguous same-wave block in the current 256-step rollout:

`A_BRSC(t) = Z * (gamma * lambda)^(b-t)`.

The backward walk stops at rollout start, episode reset, a prior wave boundary, or a wave-index change. It never enters a prior rollout, prior episode, or prior wave. Reusing `gamma * lambda` introduces no new decay or segment-length hyperparameter, bounds the accumulated influence of long blocks, and concentrates credit near decisions that form the entry state. This is boundary-anchored on-policy segment credit, not complete whole-wave causal attribution.

Whole-wave frozen-policy collection was deliberately rejected because the synchronous 24-environment runner updates every 256 steps. Pausing environments, waiting for aligned boundaries, or replaying actor transitions would change baseline sampling and optimization or introduce off-policy PPO data.

## Actor protection

The tactical reward, GAE, normalized tactical advantages, value targets, entropy term, and LR schedule remain unchanged. BRSC uses actual current-rollout actions and the ordinary clipped PPO ratio. Wave-1 and Wave-2 auxiliary losses are averaged equally. Conflicting auxiliary gradients alone are projected away from the tactical gradient, then capped to 25% of its norm. Each minibatch still performs one actor optimizer step. If no ready and credited boundary exists, the original `_update_flat` path is called exactly.

## Limitations

BRSC is not full-wave causal attribution. Rollout start truncates long segments, every alive agent shares one team-level boundary credit, and the boundary critic tracks a bounded recent-policy distribution rather than an exact fixed-policy value. These constraints isolate temporal credit granularity without simultaneously changing agent attribution, environment reward, actor inputs, rollout cadence, or deployment architecture.
