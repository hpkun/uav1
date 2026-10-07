# IWSC-MAPPO development protocol

1. **Scope.** IWSC-MAPPO is a development-only candidate for the frozen three-wave `persistent_wave_v2` task.
2. **Invariant baseline.** The 52D feed-forward actor, centralized tactical critic, action space, R1–R4 reward, Blue controller, weapons, rollout cadence, and PPO hyperparameters remain unchanged.
3. **Targets.** A completed episode supplies `Y1=(c2+c3)/2` and `Y2=c3`; Wave 3 has no future-wave target.
4. **No stale policy replay.** Completed historical segments are supervised data for the quality critic only. Historical actions never re-enter PPO.
5. **Pending lifecycle.** Each vector environment accumulates Wave-1/2 pre-action states across rollout boundaries until true episode termination.
6. **Boundary semantics.** A spawned next-wave observation is appended to the source-wave segment as a boundary state; the same physical state may next appear under its new-wave semantic role.
7. **Bounded replay.** Each wave has 128 segment slots; each completed segment retains at most 64 uniformly spaced states and always keeps its boundary.
8. **Quality critic.** An independent attention critic consumes 52D entities plus a 3D source-wave one-hot and remaining-horizon scalar, then alive-means logits and applies sigmoid.
9. **Balanced supervision.** Sampling is uniform over segments and then states. Wave-1 and Wave-2 losses are equally averaged in each supervised update, preventing long segments from dominating.
10. **Readiness.** A wave becomes actor-active only after 16 completed segments, target standard deviation at least 0.05, and a valid quality-critic update.
11. **Actor credit.** Frozen predictions form `A_IW=q_next-q_t`, without gamma. Terminal `q_next=0`; transition `q_next` keeps the source-wave identity; Wave 3 is masked.
12. **Normalization.** Team-level deltas are normalized independently within Wave 1 and Wave 2; a near-zero rollout standard deviation disables that wave.
13. **Balanced actor loss.** The current on-policy PPO ratio and clipping are reused and team credit is broadcast only to alive Red actions. When multiple active source waves coexist in one PPO minibatch, their auxiliary losses are equally averaged. This is not strict epoch-level stratified sampling. The disabled ablation pools every active alive action directly.
14. **Tactical preservation.** Entropy remains solely in tactical PPO. If auxiliary and tactical gradients conflict, only the auxiliary gradient is projected before `g_T + beta*g_IW` is applied.
15. **Reproducibility and decision rule.** Plain actor/critic construction and PPO RNG are isolated from IW components; checkpoints include critic/replay/IW RNG. The preregistered three-seed 3M final gate permits only `IWSC_SUPPORTED` or `IWSC_NOT_SUPPORTED`, with no best-checkpoint rescue.
