# MAAC v3: Li et al. 2023 continuous shared-policy baseline

The scientific target is the MAAC comparison baseline in Li et al. (2023),
*Multi-UAV Cooperative Air Combat Decision-Making Based on Multi-Agent
Double-Soft Actor-Critic*. Section 3.1.3 supplies Eq.(14)-(15); Section 4
explicitly says MAAC was adapted to continuous actions and actor parameter
sharing. Table 3's MAAC feature contract is CTDE, attention, maximum entropy
and sharing, without double centralized critics or delayed policy updates.

Primary source: https://www.mdpi.com/2226-4310/10/7/574
Original MAAC reference: https://proceedings.mlr.press/v97/iqbal19a/iqbal19a.pdf
Original supplementary: https://proceedings.mlr.press/v97/iqbal19a/iqbal19a-supp.pdf
Original author code: https://github.com/shariqiqbal2810/MAAC

This is a continuous UAV adaptation of that baseline, not a claim of exact
reproduction of Li et al.'s unpublished continuous-baseline engineering details.
The initial v1 on-policy draft and v2 independent-actor implementation are
incompatible historical protocols, not the formal baseline.

## Authoritative shared policy

There is exactly one GaussianLocalActor, one shared target_actor copied from it,
and one actor_optimizer. No actors/target_actors/actor_optimizers ModuleLists
remain. Formal actor: 56 -> 128 -> 128 -> separate 3D mean/log_std heads,
LeakyReLU hidden activations, Normal sampling and tanh squashing.

All five learning Red UAVs use the same parameters with their own local
observations. Identical observations give identical distribution mean/std;
stochastic samples may still differ. The network also processes [B,5,56]
directly because its layers operate only on the final feature dimension.
Dead-agent actions are zero. Blue is the unchanged fixed environment policy.

The one target actor generates all next actions. It initializes identically
to the online actor, is frozen, and updates using .005 online Polyak weight.
Target actor parameter storage is separate from online storage.

## One shared-policy update

For one call to update_policies:

1. Sample current joint actions with the one shared actor before any step.
2. Evaluate Q_i and independent Monte Carlo counterfactual baselines for all i.
3. Calculate all local log probabilities at the same policy parameter snapshot.
4. Aggregate all living agent contributions into one scalar loss.
5. Perform one zero_grad/backward/clip/actor_optimizer.step.

```text
L_pi = sum_{batch,i} alive_i * log pi_phi(a_i|o_i)
       * stop_gradient(alpha*log pi_phi(a_i|o_i) - Q_i + b_i)
       / sum_{batch,i} alive_i
```

This is the Eq.(15) score-function structure. Normal.sample(), not rsample(),
produces actions. Pre-tanh samples are detached when computing log probabilities;
Q, baseline and the advantage coefficient are detached. The actor receives no
critic/action-path reparameterization gradient. Contributions are aggregated
before a single step, rather than five sequential steps on the same actor.

actor_updates counts actual shared actor optimizer steps. policy_updates counts
successful update_policies calls. Each nonempty call increments both by one.
An entirely dead batch increments neither. A configured training cycle still
does four critic calls and four policy calls, followed by one target update;
there is no delayed-policy skip mechanism.

## Preserved critic, replay and target

The centralized attention critic is unchanged from v2:
five agent-specific (o_i,a_i) encoders, shared scaled Q/K/V attention
transformations, LeakyReLU values, j != i attention with dead-agent masking,
and Q_i=f_i(concat(e_i,context_i)). There is one critic and one target copy,
not twin Q. No min-Q target is introduced. A lone survivor has zero context
but retains its own embedding shortcut.

Eq.(14) target with the existing terminal/death masks is unchanged:

```text
a'_i ~ shared target_pi(.|o'_i)
y_i = r_i + gamma*(1-done)*alive'_i
                 * (target_Q_i(o',a') - alpha*log target_pi(a'_i|o'_i))
```

Joint off-policy replay, true transition-next observations before auto-reset,
entropy alpha=.01 and soft target updates remain. Replay source and critic
network source are hash-protected during this change.

For each i, baseline K=16 independent alternatives are sampled using the same
shared actor on o_i. Only action i is replaced; other sampled agent actions
are held fixed. K=16 is CONTINUOUS-ACTION ADAPTATION / ENGINEERING CHOICE,
not a parameter explicitly published for Li et al.'s MAAC baseline.

## Hyperparameter provenance

Formal numeric parameters are unchanged:

| Parameter | Value |
|---|---:|
| hidden / heads | 128 / 4 |
| actor / critic lr | .001 / .001 |
| gamma / alpha / tau online weight | .99 / .01 / .005 |
| train environments | 12 |
| replay / batch | 1e6 / 1024 |
| minimum sampled transitions per cycle | 100 |
| critic / policy updates per cycle | 4 / 4 |

These are ORIGINAL-MAAC / AUTHOR-CODE-BASED HYPERPARAMETERS, not a
MAAC-specific parameter table published by the MADSAC paper. YAML metadata
uses original_maac_reference and separately identifies Li paper modifications:
continuous-action adaptation and shared actor parameters. The MADSAC paper's
256/2/1e-4 settings are not imported into MAAC.

Engineering choices remain Gaussian+tanh, log_std bounds [-5,2], K=16 baseline,
warmup=1024, UAV's unchanged 1000-step horizon, total budget and reporting schedules.
The configured horizon is not replaced by the original MAAC benchmark's 100-step
reset. Gradient clipping and shared attention gradient scaling retain their
original-author-code provenance. No new reward, dynamics or controller is added.

## Identity and checkpoints

IMPLEMENTATION_VERSION=3
variant=madsac_paper_maac_continuous_shared_eq15_mc_v3

Metadata records shared_actor=True, independent_actors=False, target_actors=1,
single_attention_critic=True, self_attention=False, own_embedding_shortcut=True,
maximum_entropy=True and delayed_policy_update=False. Dimensions remain 56/3/5.
Version, variant, environment/algorithm fingerprints and architecture are strict.

Checkpoints contain actor, target_actor, critic, target_critic, actor_optimizer,
critic_optimizer, sampled_steps, actor_updates, critic_updates, policy_updates,
target_updates and training seed. Plural actor/optimizer fields are removed.
MAAC v1/v2 and incompatible environment/algorithm checkpoints are rejected for
load, evaluation, recording and resume identity checks.

Exact resume is still unsupported because replay/environment/full RNG continuation
is not checkpointed. The training --resume option explicitly checks identity and
then rejects unsupported exact resume; it does not silently start a fresh run.

START/TRAIN/EVAL logs stay compact. JSONL retains gradients/loss, replay/cycle
counts, controller/firing diagnostics and actual-attempt fire_rate. The existing
MAPPO fire-window console meaning is not changed.

## Validation

Use WSL Ubuntu/uav, or Windows PowerShell/uav if WSL is unavailable. Training,
evaluation and checkpoint audits require CUDA; CPU is used only for unit tests.

```text
python -m pytest tests/test_maac.py -q
python -m pytest tests/test_mappo.py tests/test_madsac.py tests/test_madsac_v3.py tests/test_madsac_v326.py tests/test_combat_v329.py -q
python -u algorithm/train_maac.py --device cuda --seed 1 --num-envs 2 --total-sampled-steps 32 --smoke --output-dir outputs/maac_v329_validation/shared_v3/cuda_smoke
```

Training directories must be fresh. Smoke retains the existing reduced
32-hidden/16-batch/K4 profile. CUDA formal-spec unit validation independently
uses hidden=128, heads=4, batch=1024, K=16 and a real shared policy optimizer step.
It is not a long training experiment. No 7M training is started or prescribed
before the requested upload/review.
