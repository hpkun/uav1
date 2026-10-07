# HTA-MAPPO V2: Progressive Two-Timescale Worker Consolidation

## Evidence and question

HTA-MAPPO V1 failed its exact-3M primary gate. All three development seeds nevertheless produced strong intermediate policies followed by weaker final policies. Those intermediate checkpoints did not establish a stable overall advantage over Plain MAPPO, and temporal abstraction therefore has no established causal benefit. The retained diagnosis is `CASE_E_MIXED_WORKER_RETENTION_FAILURE`: later-wave conditional signals exist, but neither local Worker KL instability, dominant residual runaway, nor Manager collapse is supported as the single main cause.

V2 tests one hypothesis only: after temporal abstraction has formed, a continually moving low-level Worker may impair Manager–Worker co-adaptation. The sole intervention is a stateless progressive multiplier on the complete Worker Actor learning rate after 1.5M sampled steps.

## Frozen HTA core and intervention

The HTA core remains V1 (`HTA_MAPPO_VERSION = 1`): four latent options, `K=16`, the same semi-MDP Manager PPO, option-residual Worker, tactical option critic, wave/rollout boundary semantics, rewards and RNG lineage. V2 is HTA core V1 plus `HTA Worker Consolidation V1`.

The existing actor schedule remains 3e-4 through 600k, linearly reaches 1e-4 at 900k, and remains 1e-4 thereafter. The Worker multiplier is exactly 1 through 1.5M, decreases linearly to 0.25 at 2.5M, and remains 0.25. Thus the Worker LR is 1e-4 at 1.5M, 6.25e-5 at 2.0M, and 2.5e-5 from 2.5M onward. Manager Actor LR remains 1e-4 after 900k. Tactical and Manager critic LRs remain 3e-4.

The intervention does not freeze the Worker and does not add an optimizer, network parameter, loss, reward, random draw, target network, replay buffer, KL anchor, boundary shaping, option balancing or semantic option. Freezing was rejected because it would eliminate adaptation rather than test a progressive two-timescale hypothesis. KL anchoring would add a reference-policy objective; boundary shaping would change the task reward; option diversity would change Manager learning. Each would confound the single intervention.

## Protocol and closure

Training is from scratch for seeds 5301, 5302 and 5303. Development evaluation remains the existing deterministic common 44M range. The 45M future-final block remains untouched and marked `executed=false`.

The primary comparison is paired Plain MAPPO exact-3M versus HTA V2 exact-3M. HTA V1 is secondary mechanism evidence only. Best checkpoints cannot replace the primary endpoint. If the frozen primary gate fails, the result is `HTA_ROUTE_CLOSED`; no HTA V3, Worker freeze, KL anchor, adaptive K, option diversity, boundary shaping, or further LR sweep is permitted.

