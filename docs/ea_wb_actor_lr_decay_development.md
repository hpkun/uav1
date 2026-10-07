# EA-WB Actor-Only LR Decay Development Protocol

Actor-only delayed linear learning-rate decay is an optional stability technique for EA-WB-MAPPO. It is not part of the frozen primary-method definition and is not claimed as a core paper innovation.

The matched development experiment uses one common parent: the complete EA-WB checkpoint at `sampled_steps=502752`.

- Control: fixed Actor LR `3e-4`, from 502752 to 900000.
- Treatment: Actor LR `3e-4` through 600000, followed by a linear decay from `3e-4` at 600000 to `1e-4` at 900000.
- Critic LR: fixed at `3e-4` in both branches.

The intervention changes only Actor optimizer parameter-group LR. It does not enable PPO Stabilization, KL early stopping or Advantage Priority, and it does not change update ordering, minibatch permutations, PPO epochs, clipping, entropy, GAE, Wave Balance, Entity Attention or network architecture.

The primary purpose is to reduce late-stage best-to-latest regression, not merely to maximize peak validation performance. The primary comparisons are:

1. 900k latest performance;
2. average validation performance over 600k-900k and 700k-900k;
3. best-to-latest retention;
4. W1/W2/W3 clear probabilities;
5. average waves cleared;
6. return;
7. red loss;
8. kill/loss ratio.

Development validation uses 20 episodes and does not support statistical-significance claims.

## Branch and reproducibility boundary

`--branch-from` is an explicit, fresh-output operation. It preserves the parent checkpoint as read-only, starts new training/evaluation logs, records the parent path/SHA256/step and permits only `actor_lr_decay`, target total steps, output directory and branch metadata to differ. Ordinary `--resume` remains strict and same-directory only.

Future checkpoints store Python, NumPy, Torch CPU, available CUDA-device and trainer permutation-generator RNG states. Old checkpoints remain loadable and are marked as lacking restorable RNG state. Episode indices remain saved, but complete vector-environment state is intentionally not serialized. A matched branch controls model, optimizer, counters, config and available saved RNG state; an old checkpoint does not guarantee reproduction of the original future trajectory.

Ordinary resume now creates one content-addressed `resume_points/resume_start_<step>_<sha>.pt` copy before later saves can overwrite `latest.pt` or `final.pt`. Branch mode never writes into the parent run.
