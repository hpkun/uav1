# MADSAC for Combat Environment v3.26

This adaptation changes environment/config/checkpoint protocol and tests only.
networks.py, trainer.py, replay_buffer.py and all environment sources/configs
remain byte-identical. Shared local tanh-Gaussian actor, independent twin
centralized attention critics, target actor/critics, entropy, delayed actor
updates, Polyak updates and loss/reduction formulas are unchanged.

Formal config: configs/madsac_5v5_v326.yaml. It explicitly declares environment
3.26/5v5 and network observation/action/agents=56/3/5. Actor/critic hidden layers
remain 256/256 with two attention heads. Training retains actor/critic lr=1e-4,
gamma=.99, alpha=.1, tau=.001, replay=1M, minibatch=1024, 24 envs, 3M transitions
and CUDA. learning_starts=10000, policy_delay=2 and one gradient step per vector
step remain project implementation choices rather than claimed original paper
parameters. Existing evaluation and checkpoint intervals are preserved.

The validator adds v3.26 while retaining supported historical protocols. The
v3.26 branch requires exact integer 56/3/5 dimensions, environment declaration
and formal architecture. Checkpoints still require implementation identity,
environment version, both config hashes and training seed. v3.26 additionally
validates architecture/provenance fields. Legacy 4v4/52D checkpoints are rejected,
including a test that forges new metadata but retains actual old actor weights.

Checkpoint extra records environment_version=3.26, 56/3/5, shared_actor=true,
twin_independent_critics=true, attention_heads=2 and actual hidden_dim. Formal
hidden_dim=256; the unchanged smoke mode uses hidden_dim=32 and explicitly marks
training_smoke=true. formal_exact_resume_supported=false and
replay_buffer_in_checkpoint=false remain intact: formal exact resume is not
supported because replay is omitted, as before.

Unit tests now cover [B,5,56] observations, [B,5,3] actions, [B,5] masks/Q and
[B,2,5,5] attention. Dead-agent influence, twin parameter independence, initial
target copies/frozen targets, Polyak updates, own-action-only gradients, delayed
actor updates, replay terminal transitions, evaluation RNG and checkpoint reload
are verified. A separate CUDA test instantiates and reloads actual 256-wide
formal networks.

Validation:

```text
python -m pytest tests -k madsac -q
python -m pytest tests/test_combat_protocol.py::test_factory_and_public_api_are_combat_only tests/test_combat_v326.py -q
```

51 MADSAC-related tests and 19 environment/protocol regressions passed. Python
compilation and diff whitespace checks passed. Protected environment and core
algorithm hashes are unchanged.

CUDA smoke (not formal training):

```text
python -u algorithm/train_madsac.py --env-config configs/combat_environment_v326.yaml --algorithm-config configs/madsac_5v5_v326.yaml --device cuda --seed 1 --num-envs 4 --total-sampled-steps 1200 --smoke --output-dir outputs/madsac_v326_smoke_20261013_162234
```

Completed 1200 environment transitions / 300 vector steps. 297 critic updates,
148 actor updates, replay size 256; every recorded loss and gradient norm is
finite. The smoke's existing effective parameters are hidden=32, replay=256,
batch=16 and learning_starts=8. These do not change formal YAML parameters.
Evaluation completed; checkpoints best_eval, checkpoint_1200, latest and final
were saved. The actual saved checkpoint passed official protocol validation and
the official CUDA recording-policy loader.

Artifacts: outputs/madsac_v326_validation/validation.json and the smoke directory.
No unresolved integration failures were observed. This verifies sampling/update/
evaluation/checkpoint compatibility, not successful combat learning from only
1200 transitions. No formal long training or version-control mutation was run.
