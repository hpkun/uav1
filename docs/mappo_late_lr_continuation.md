# MAPPO Late-Training LR Matched Continuation

This development experiment branches both arms from the byte-identical
`checkpoint_1001472.pt` produced by the MAPPO-Attn seed-5303 run. The control
keeps Actor/Critic learning rates at `3e-4/3e-4`; the treatment changes only
the restored Actor optimizer learning rate to `1e-4` while retaining the
Critic learning rate at `3e-4`.

The checkpoint restores Actor and Critic parameters, both optimizer states,
PPO/optimizer/sample/vector counters, and the 24 environment episode indices.
It does not contain Python, global NumPy, Torch CPU, Torch CUDA, or trainer
permutation RNG state. Consequently this experiment does **not** claim
bitwise continuation of the original run or event-wise common-action-noise
coupling. The two branch processes instead use the same training seed, the
same episode-index continuation rule, the same architecture, and the same
47,000,000--47,000,049 deterministic evaluation scenarios.

Results are descriptive. The analyzer has no automatic winner or support
gate; exact 1.5M endpoints remain primary and best checkpoints are diagnostic.
