# MARC-SM V1: development-only State Memory Actor

Method: `marc_mappo_state_memory_v1`. This is not a replacement for formal
MARC V2, feed-forward Credit+Balance or the separate phase-initialized WS-GRU.

## Exact topology

Environment observation stays52D. A fixed3D current-wave one-hot is generated
by the Actor/trainer, not by the environment, and shared by alive homogeneous
agents. The Actor uses x=[observation,one-hot] (55D), then GRUCell(55,128).
The concatenation [x,new_hidden] is183D and directly enters
Linear(183,256)→ReLU→Linear(256,256)→ReLU→existing mean/log_std heads.
Sampling, log_std bounds, tanh and deterministic tanh(mean) are unchanged.
There is no projection, adapter, phase parameter, auxiliary objective or gate.

Only the Actor topology changes. The centralized2-head attention critic keeps
its52D input, zero context dimension and feed-forward topology. Extra Actor
construction is protected by fork_rng so same-seed critic initialization is
exactly matched. The changed Actor input width necessarily prevents a claim
of identical whole-Actor initialization to the feed-forward control.

## Lifecycle and sequence training

Initial episode hidden is literal zero. Same-wave history carries, including
across sampling rollouts. A wave-clearing action still uses the old wave's
one-hot and hidden. After spawn, before the next decision, hidden is zero and
one-hot refers to the new wave. Dead-agent hidden stays zero. Resume restores
model/optimizer state but uses the existing fresh episode reset semantics;
stale process-local hidden is not restored.

The existing actor_recurrent_phase_reset_flags [T,E] field is reused as explicit
episode/wave boundary-reset provenance; its name does not imply a learned
phase parameter for this method. Wave-aware chunks are chronological within
one environment, never cross episode/wave boundaries, cover every rollout
step exactly once, and have length≤32. Boundary chunks start from literal
zeros; ordinary length-cut chunks start from detached saved pre-action hidden.

MARC full-horizon GAE still crosses wave boundaries. Local GAE only stops the
lambda recursion while retaining its TD bootstrap. A_cont=A_global−A_local;
eta=(1/3,1/2,1). Normalization order and tempered actor weights (tau=.5,
bounds=.5..2, mean preserving) are unchanged. The critic target remains the
full-horizon return. Critic weighting and entropy are unchanged.

The combined recurrent MARC route asserts10 actual PPO epochs and the planned
actor/critic optimizer counts. Actual chunk and minibatch counts are logged;
wave-segment splits can increase Actor minibatch count relative to the flat
control. This optimization-geometry difference must remain visible in analysis.

## Protocol and diagnostics

Config extends the existing Credit+Balance control; default budget2M,
seed5301, 24 environments, rollout256, minibatch512 and all common PPO/LR
parameters unchanged. Frozen persistent_wave_v2, Red4, Blue[4,3,3], H3000.
The deterministic50-episode development bank remains44000000..44000049.
The future45M holdout is not executed.

Enabled modules are exactly actor_lr_decay, milestone_aware_retention_credit,
recurrent_memory. V2 deployment coefficient=0 disables elite ingestion,
sampling and state-only loss. Formal V2 positive retention remains untouched.
The new recurrent config explicitly records fixed3D context, hidden128,
sequence32, and zero initialization; incompatible settings are rejected.

Rollout diagnostics include hidden/GRU gradient norms, boundary-reset count,
chunk/minibatch/update counts and alive-agent one-hot counts by wave.
Canonical evaluation trace includes pre-action wave, one-hot, reset flag,
pre-action per-agent hidden norm, and action. Periodic/best/final/standalone
modular evaluation share the canonical lifecycle; recording also supports it.
Unsupported historical callers that omit explicit boundary provenance fail.

## Changed/new files

- algorithm/modules/recurrent_memory.py
- algorithm/modular_mappo/networks.py
- algorithm/modular_mappo/trainer.py
- algorithm/modular_mappo/runner.py
- algorithm/modular_mappo/evaluation.py
- algorithm/modular_mappo/factory.py
- algorithm/modular_mappo/protocol.py
- configs/dev_marc_state_memory_v1_2m.yaml
- tools/preflight_marc_state_memory.py
- tools/smoke_marc_state_memory.py
- tools/record_combat_episode.py
- tools/preflight_actor_only_gru_history.py (source-audit guard-name adaptation;
  no old-GRU runtime semantics change)
- tests/test_marc_state_memory.py
- docs/marc_state_memory_v1.md

## Validation scope

Preflight is read-only and requires CUDA. Tiny smoke takes8 sampled environment
steps, one constructed three-phase PPO batch, checkpoint/resume and a6-step
canonical deterministic evaluation with controlled transition signals. These
signals validate plumbing, not learned ability to reach or clear later waves.
No2M/3M training, multi-seed experiment or formal50-episode evaluation is launched
by these tools. The user runs long experiments manually.

Verification: syntax/import checks PASS; relevant suite288 passed /0 failed;
CUDA preflight PASS on RTX5060 Laptop; tiny smoke PASS. Constructed update
executed10 epochs,6 chunks,1 recurrent minibatch/epoch and10 Actor/10 Critic
optimizer steps. GRU gradient norm was0.0723017910, with finite metrics.
The new formal2M output directory was absent after verification.

For attribution, compare matched sampled-step endpoints. A2M treatment final
must not be treated as a budget-matched comparison with a1M control final.
