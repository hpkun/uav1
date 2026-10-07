# EA-HWB-MAPPO development design

EA-HWB-MAPPO (Entity-Aware Hierarchical Wave-Balanced MAPPO) is a strictly
on-policy extension of the established WB-MAPPO path. It addresses three
separate defects: a flat vector does not express entity identity; later waves
are conditionally rare; and, even within a wave, informative positive-advantage
samples can be diluted by many routine samples. The environment, opponent,
weapons, rewards, mission transitions, centralized critic and squashed-Gaussian
action mathematics are unchanged.

## Entity-aware decentralized actor

The fixed 52D local observation is split as follows:

- self: `[0:7]`;
- allies: three 7D blocks `[7:14]`, `[14:21]`, `[21:28]`, with alive flags at
  global indices 13, 20 and 27;
- enemies: four 6D blocks `[28:34]`, `[34:40]`, `[40:46]`, `[46:52]`, with
  alive flags at global indices 33, 39, 45 and 51.

Alive flags are masks, not encoder inputs. Self is encoded by `7→32→32`, each
ally by one shared `6→32→32` encoder, and each enemy by a separate shared
`5→32→32` encoder. ReLU follows every layer. Two independent two-head attention
blocks use encoded self as query and allies/enemies as their respective key and
value sets. Dead keys are masked before softmax and their returned probability
is explicitly zeroed and renormalized. For an all-dead group, one dummy key is
temporarily unmasked solely to keep the kernel finite; its context and weights
are then set exactly to zero.

The concatenated `[h_self,c_ally,c_enemy]` is 96D and is fused through
`96→256→256`, followed by the existing mean/log-standard-deviation heads,
Normal sampling, tanh and exact latent-space Jacobian correction. The actor
remains decentralized. Entity attention v1 deliberately rejects simultaneous
wave context or recurrent memory. With entity attention disabled, the original
actor topology and state dictionary are retained exactly.

## Hierarchical wave/advantage balancing

Existing wave balance supplies a wave weight `w_k` using its unchanged capped,
alive-count inverse-frequency rule. Immediately after GAE and before global
advantage normalization, raw advantages are copied. For alive samples in wave
`k`:

```text
z_i = (A_raw_i - mean_k(A_raw)) / (std_k(A_raw) + epsilon)
q_i = 1 + alpha * min(max(z_i, 0), z_clip)
p_i = q_i / mean_k(q)
```

Defaults are `alpha=0.5`, `z_clip=2`. Thus every represented wave has
`mean_k(p)=1`; a one-sample wave is safe and receives priority one. “Highlighted”
means `z_i > 0`.

The provisional actor weight is `w_k p_i`. A binary search finds a nonnegative
scale `s` such that

```text
g_i = min(s w_k p_i, 4),       mean_alive(g) = 1.
```

This retains both the hard cap and baseline loss scale. It is intentionally not
implemented as clip-then-divide, which can violate the cap. The final losses are

```text
L_actor = -sum(mask_i g_i min(r_i A_i, clip(r_i) A_i)) / sum(mask_i)
L_value = 0.5 sum(mask_i w_k value_error_i) / sum(mask_i)
```

where `A_i` is the existing globally normalized advantage. Priority is actor
only. The critic retains only the existing wave weight, while entropy remains
an ordinary unweighted alive mean. When priority is disabled, actor weights are
bit-for-bit the existing WB weights; when all new modules are disabled, the old
MAPPO update path is unchanged.

## PPO stability guard

Stabilization leaves PPO clipping at 0.2. After every complete actor epoch it
re-evaluates all real alive rollout samples and computes

```text
log_ratio = new_log_prob - old_log_prob
ratio = exp(log_ratio)
KL_i = (ratio - 1) - log_ratio
epoch_KL = mean_alive(KL_i).
```

If `epoch_KL > 0.030`, remaining actor epochs stop. The critic has an independent
loop and always completes all planned epochs. `target_kl=0.015` is diagnostic;
it does not adapt clipping. Actor learning rate is deterministic from checkpoint
progress:

```text
progress = clip(sampled_steps / total_sampled_steps, 0, 1)
lr = 1e-4 + (3e-4 - 1e-4) * (1 - progress).
```

The critic stays at `3e-4`. Resume restores sampled steps and reconstructs actor
LR, so no wall-clock or output-directory state is involved.

## Protocol and interpretation

There is no replay buffer or historical data: balancing and priority use only
the current on-policy rollout and affect training losses only. At inference,
only the entity-aware actor adds work; the centralized critic is training-only.
All new functions are opt-in and are identified by module-config SHA plus
development feature versions, without changing historical MAPPO or Modular
MAPPO v2 version meanings.

Expected screening effects are deliberately separated:

- EA should improve target/teammate selection and survivor/kill efficiency;
- HWB should improve later-wave reach/clear probability without erasing
  within-wave learning signals;
- Stable should reduce destructive actor updates and hard-stop diagnostics,
  not change the task or reward optimum.

Diagnostics report attention entropy/top-1/alive count/dead mass, per-wave
priority statistics, combined weight statistics, actor/critic learning rates,
actor/critic epoch counts and full-epoch KL stops. Evaluation flattens reliable
existing per-wave records for red/blue survivors, red attack/boundary/ground
losses, entry step and duration; it does not infer missing waves.
