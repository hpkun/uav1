# Feed-forward MARC core factorial development experiment

All four configs inherit `dev_marc_credit_balance_1m.yaml`, use a 3M default
budget and differ only in factorial identity, continuation alpha and balancing
temperature. Retention is disabled, including elite ingestion and sampling.

| Cell | Alpha | Temperature | Eta W1/W2/W3 |
| --- | --- | --- | --- |
| none | 0 | 0 | 1/1/1 |
| credit_only | 1 | 0 | 1/3, 1/2, 1 |
| balance_only | 0 | .5 | 1/1/1 |
| full | 1 | .5 | 1/3, 1/2, 1 |

No architecture, reward, environment, PPO formula or LR schedule changes.
Actor is feed-forward 52D Gaussian MLP256-256; centralized attention Critic
is feed-forward, two heads. Environment is frozen 433, three waves, H3000.
Development validation remains deterministic50, 44000000..44000049.
45000000..45000199 is reserved and not executed by the tools.

`preflight_marc_factorial.py` validates the exact resolved source lineage,
checks fresh stage-1 output directories, matches Actor/Critic initialization
and post-initialization RNG for 5301/5302, and compares constructed complete
ten-epoch PPO updates. Entropy sampling RNG is reset between comparisons.
A0 is Plain-equivalent within floating-point precision, not promised bitwise
equivalent: local+(global-local) can round differently from global GAE.
A3 uses the unchanged full-MARC optimization path.

`smoke_marc_factorial.py` executes four tiny CUDA environment rollouts and
constructed mixed-wave updates; it does not run evaluation or long training.
The dedicated validator is separate from existing recurrent/formal guards.

Stage 1 prepares only fresh credit_only and balance_only seed5301 runs,
with distinct output directories. Existing Plain/full results are not rerun.
Their historical code/config/runtime provenance must also be matched before
interpreting cross-run contrasts as strictly causal. One seed is exploratory;
factorial interaction is (full-credit_only)-(balance_only-none), evaluated
at matched sampled-step endpoints, not selected peak checkpoints.
