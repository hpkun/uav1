# Jiao et al. (2025) reproduction protocol

## Scope and source

This is a **paper-aligned algorithm reproduction on the project backbone**, not
an exact reproduction of the authors' simulator. The source is Yongkang Jiao et
al., “Collaborative decision-making for UAV swarm confrontation based on
reinforcement learning,” *IET Control Theory & Applications*, 19, e12781
(2025), DOI: [10.1049/cth2.12781](https://doi.org/10.1049/cth2.12781).

The frozen target environment is `persistent_wave_v2`: homogeneous 4v4, 3-D,
three waves, `max_steps=3000`, with the existing nearest-target Blue policy.
Nothing in `env/` is changed by this reproduction.

## Source-to-code mapping

| Item | Paper-specified | Current adaptation | Reason |
|---|---|---|---|
| Round information | Observation/state explicitly contains round number `F`. | `WaveContextModule(encoding: scalar_round)` emits the unnormalised scalar `F = wave_index` in `{1,2,3}` to both actor and critic. | Preserves the paper's scalar semantics; the project's default rich one-hot/progress/remaining encoding remains unchanged. |
| Actor GRU | Algorithm 1 and Sec. 3.1 use a GRU and restore hidden state for trajectory chunks. | Actor backbone → `GRUCell(128)` → Gaussian heads. Contiguous sequence BPTT restores the hidden state saved at each chunk start. | `128` is **PAPER_UNSPECIFIED_ADAPTATION**. |
| Critic GRU | Algorithm 1/Sec. 3.1 specify recurrent policy and value networks. | Project centralized critic backbone → `GRUCell(128)` → value head, trained on the same contiguous chunks. | `128` is **PAPER_UNSPECIFIED_ADAPTATION**. The attention encoder is a **COMMON_PROJECT_BACKBONE_ADAPTATION**, not a Jiao contribution. |
| Hidden lifecycle | Trajectories store recurrent hidden states and are split into length-`L` chunks. | Hidden state persists at every ordinary transition and across W1→W2/W2→W3. It is reset only at true episode termination/truncation; dead-agent hidden vectors are masked to zero. Chunk length is 32. | Chunk length 32 is **PAPER_UNSPECIFIED_ADAPTATION**. A wave spawn is not an episode boundary in `persistent_wave_v2`. |
| PopArt | Algorithm 1 explicitly normalizes reward-to-go with PopArt. | Existing `PopArtValueNormalizer` normalizes value targets and applies output-preserving affine rescaling to the recurrent critic value head. State is saved/restored in checkpoints. | Reuses the hardened implementation. `beta=0.999`, `epsilon=1e-5` are **PAPER_UNSPECIFIED_ADAPTATION**. |
| PPO losses | Eq. (9): clipped PPO actor objective plus entropy. Eq. (10): clipped value objective. Table 2 gives clip and entropy coefficients. | Existing stable tanh-squashed Gaussian log-ratio, GAE, alive mask, advantage normalization and clipped value loss are retained. | Required for stable continuous 3-D actions; implementation details are **COMMON_PROJECT_BACKBONE_ADAPTATION**. |
| Multi-round R2 | Eq. (12): Blue death `+(j+1)(L+1)`; Red death `-(i+1)`. Only the kill term grows with round. | Paper `L+1` maps to the current 1-based `wave_index`. Rewards are emitted only for new 1→0 deaths, simultaneous events are summed, and the team signal is copied to Red agents alive at transition start. | Prevents repeated reward after death and preserves fully cooperative semantics. |
| Paper hyperparameters | actor LR 0.0005; critic LR 0.0005; 10 PPO epochs; clip 0.1; entropy 0.01; GAE lambda 0.95; gamma 0.99; episode length 150. | The numerical parameters are used. The paper's 150 is mapped to `rollout_steps=150`; environment `max_steps` stays 3000. | Cross-environment adaptation: rollout horizon 150 is not claimed to reproduce the paper's 2-D termination horizon. |
| Parallelism/minibatch | Not reported. | 24 environments; minibatch 512. | **PAPER_UNSPECIFIED_ADAPTATION**, matched to the current project infrastructure. |
| Network width | The paper mentions two fully connected layers plus GRU but does not fully report widths. | Project actor width 256 and centralized critic width 256 with two attention heads. | **PAPER_UNSPECIFIED_ADAPTATION** and **COMMON_PROJECT_BACKBONE_ADAPTATION**. |
| Action space | Paper: 2-D `[v, omega]`. | Frozen 3-D tanh-Gaussian `[delta_psi, delta_theta, delta_v]`. | Algorithm transfer to the current environment; changing environment/action semantics is forbidden. |
| Scenario | Paper: 3v3 planar rounds and Hungarian-controlled Blue team. | Frozen homogeneous 4v4 3-D `persistent_wave_v2`, three waves, nearest-target Blue pursuit. | This evaluates transfer on the project's task and is not an exact simulator reproduction. |
| Training budget | Not a paper parameter for this comparison. | 1,500,000 sampled steps, seed 2023. | **PAPER_UNSPECIFIED_ADAPTATION**, matched sample budget versus All-Off/M5. |
| Training validation | Not specified by the paper. | Deterministic seeds 10,000,000–10,000,019 (20 episodes), every 100k sampled steps. | **PAPER_UNSPECIFIED_ADAPTATION**; exactly matches the existing All-Off/M5 checkpoint-selection information budget. |
| Fresh comparison | Not specified by the paper. | Deterministic seeds 38,000,000–38,000,099 (100 episodes), used only after both Jiao runs finish. | **PAPER_UNSPECIFIED_ADAPTATION**. The originally planned 37M range is permanently excluded because two of its seeds were consumed by an earlier development smoke. |

## Two preregistered variants

- **Jiao2025-Core** (`configs/jiao2025_core_1p5m.yaml`): scalar `F` + actor/critic GRU + PopArt + paper PPO hyperparameters; training reward is the unchanged environment reward.
- **Jiao2025-Full** (`configs/jiao2025_full_1p5m.yaml`): Core plus Eq. (12) as a **replacement** training reward. Raw environment reward remains logged and is the only return used for cross-method performance comparison.

Both variants explicitly disable wave balancing, warm start, curriculum and
policy anchoring. The planned fresh output directories are
`outputs/jiao2025_core_1p5m_seed2023` and
`outputs/jiao2025_full_1p5m_seed2023`.

## Eq. (12) transition semantics

For transition-start wave `w`, Blue alive mask `b`, next Blue alive mask `b'`,
Red alive mask `r`, and next Red mask `r'`:

```text
blue_component = sum_{j: b_j=1 and b'_j=0} (j+1) * w
red_component  = sum_{i: r_i=1 and r'_i=0} -(i+1)
team_R2        = blue_component + red_component
agent_reward_i = team_R2 * r_i
```

On a clear transition, the environment may already expose the newly spawned
Blue wave. The adapter therefore treats the just-cleared old Blue mask as zero
for event detection, then runner state advances to the new all-alive Blue mask.
This makes every death a one-transition event and handles multiple/mutual kills.

## Reporting rules

Training logs separately expose `raw_environment_reward`,
`jiao_training_reward`, `paper_R2_blue_kill_component`,
`paper_R2_red_loss_component`, and `paper_R2_wave1/2/3`. Cross-method tables use
fresh raw-environment metrics (W1/W2/W3, waves, losses, K/L, boundary, ground,
timeout and episode length). Jiao R2 is reported in a separate training-signal
table and is never compared with environment return as if they were equivalent.
