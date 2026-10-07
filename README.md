# MAPPO UAV Combat Research Project

This repository contains the frozen Direct V2.3 4v4 UAV combat environment,
the Persistent-Wave V1/V2 mission variants, and the MAPPO baseline. The current
mainline is `persistent_wave_v2`; Direct V2.3 remains the single-round base
environment. Observation dimension (52), action dimension (3), environment
semantics, and MAPPO implementation version 2 are unchanged by the repository
layout.

## Project layout

```text
algorithm/  MAPPO implementation, shared RL utilities, train/evaluate entries
env/        Direct V2.3 and Persistent-Wave environments and all foundations
configs/    Flat environment and algorithm YAML files
outputs/    One direct child directory per experiment
tools/      Validation, audit, aggregation, and plotting utilities
tests/      Flat automated test suite
papers/     Reference paper PDFs
docs/       Active specifications and historical archives
```

## Install third-party dependencies

Use Python 3.10 or newer and install only the third-party dependencies:

```bash
pip install -r requirements.txt
```

The project itself does not need editable or wheel installation. `pytest` works
from the project root, while algorithm entries and tools bootstrap the project
root themselves and can be executed from any working directory.

```bash
pytest -q
python tools/validate_combat_environment.py
```

## Smoke test

```bash
python algorithm/train_mappo.py \
  --smoke \
  --device cpu \
  --num-envs 1 \
  --output-dir outputs/mappo_smoke
```

The default smoke and formal environment is Persistent-Wave V2, using
`configs/persistent_wave_v2_environment.yaml` and
`configs/mappo_persistent_wave.yaml`.

## Persistent-Wave V2 formal training

```bash
python -u algorithm/train_mappo.py \
  --device cuda \
  --seed 2023 \
  --num-envs 24 \
  --total-sampled-steps 8000000 \
  --env-config configs/persistent_wave_v2_environment.yaml \
  --algorithm-config configs/mappo_persistent_wave.yaml \
  --output-dir outputs/mappo_pw_v2_8m_seed2023
```

Persistent MAPPO uses `gamma=0.999`.

## Direct V2.3 training

```bash
python -u algorithm/train_mappo.py \
  --device cuda \
  --seed 2023 \
  --num-envs 24 \
  --total-sampled-steps 500000 \
  --env-config configs/combat_environment.yaml \
  --algorithm-config configs/mappo.yaml \
  --output-dir outputs/mappo_direct_v2_3_seed2023
```

Direct MAPPO uses `gamma=0.99`.

## Checkpoint evaluation

```bash
python algorithm/evaluate_mappo.py \
  --checkpoint outputs/mappo_pw_v2_8m_seed2023/best_eval.pt \
  --env-config configs/persistent_wave_v2_environment.yaml \
  --algorithm-config configs/mappo_persistent_wave.yaml \
  --seed-base 20000000 \
  --episodes 20 \
  --device cpu \
  --output outputs/mappo_pw_v2_8m_seed2023/holdout_evaluation.json
```

Checkpoint loading validates `environment_version`, `environment_variant`, and
MAPPO implementation compatibility before formal use. New checkpoints also
embed the effective training seed, gamma, environment count, sampled-step
target, smoke/formal mode, and SHA-256 fingerprints of both YAML configuration
objects. Configuration fingerprints use canonical sorted compact JSON, so YAML
formatting and mapping-key order do not alter experiment identity.

Cross-variant policy transfer is never implicit. To evaluate a Direct-trained
checkpoint in Persistent-Wave V2 (or the reverse), add the explicit flag:

```bash
python algorithm/evaluate_mappo.py \
  --checkpoint outputs/direct_g099_seed2023/best_eval.pt \
  --env-config configs/persistent_wave_v2_environment.yaml \
  --algorithm-config configs/mappo.yaml \
  --seed-base 20000000 \
  --episodes 200 \
  --device cpu \
  --allow-cross-variant \
  --output outputs/direct_g099_seed2023/holdout_on_pw_v2.json
```

Without `--allow-cross-variant`, an environment-variant mismatch is rejected.
Environment version, MAPPO implementation version, dimensions, and model weight
structure remain strict even in transfer evaluation.

## Validation and tools

Common direct commands include:

```bash
python tools/validate_combat_environment.py
python tools/validate_persistent_wave_environment.py
python tools/check_parallel_env.py --num-envs 2 --steps 10
python tools/stress_test_ground_guard.py --output outputs/ground_guard_check.json
```

All directly executable tools resolve project-relative configs, checkpoints,
inputs, and outputs against the repository root.

## Experiment output contract

One run equals one direct child directory under `outputs/`:

```text
outputs/
└── mappo_pw_v2_8m_seed2023/
    ├── algorithm_config.yaml
    ├── env_config.yaml
    ├── run_config.json
    ├── train.log
    ├── training_metrics.jsonl
    ├── optimization_metrics.jsonl
    ├── evaluation_history.csv
    ├── best_eval.pt
    ├── checkpoint_*.pt
    ├── latest.pt
    └── run_summary.json
```

The `--output-dir` value is the final run directory. No automatic
`run_seed_*` or algorithm/environment/seed hierarchy is added. If omitted, the
entry creates one timestamped directory directly under `outputs/`. Every run
stores snapshots of both YAML inputs plus a JSON record of effective command-line
settings.

Fresh training accepts only a missing or completely empty output directory. A
non-empty directory is rejected before the runner or multiprocessing workers are
created. With `--resume`, omission of `--output-dir` reuses the checkpoint's
parent directory; an explicitly different directory is rejected. Stored YAML
snapshots must match, and every continuation is appended to
`resume_history.jsonl` without rewriting the original `run_config.json`.

Resume is a safe continuation mechanism, not a claim of bitwise-exact replay.
The original seed, environment count, smoke/formal mode, and current training
target are inherited when their CLI options are omitted. Conflicting seed,
environment count, or mode is rejected; the target may only be extended, while
the device may be changed explicitly. Selecting an older checkpoint is rejected
when a newer regular checkpoint exists. Records written beyond a selected
checkpoint after an interruption are timestamp-backed-up and truncated before
continuation, and any future `best_eval.pt` is preserved under a
`best_eval.pre_resume_*.pt` name. MAPPO/environment multiprocessing RNG and
process state are not serialized, so resumed execution is scientifically
traceable but not bit-for-bit identical to an uninterrupted process.

## Formal experiment workflow

The following long commands are protocol examples for the user to run manually.
They are not launched as part of code-maintenance validation.

### A. Direct training

```bash
python -u algorithm/train_mappo.py \
  --device cuda \
  --seed 2023 \
  --num-envs 24 \
  --total-sampled-steps 3000000 \
  --env-config configs/combat_environment.yaml \
  --algorithm-config configs/mappo.yaml \
  --output-dir outputs/direct_g099_seed2023
```

Repeat with three to five independent training seeds and one run directory per
seed.

### B. Persistent-Wave V2 training

```bash
python -u algorithm/train_mappo.py \
  --device cuda \
  --seed 2023 \
  --num-envs 24 \
  --total-sampled-steps 3000000 \
  --env-config configs/persistent_wave_v2_environment.yaml \
  --algorithm-config configs/mappo_persistent_wave.yaml \
  --output-dir outputs/pw_v2_g0999_seed2023
```

### C. Four-cell policy-transfer evaluation

```bash
python tools/evaluate_policy_matrix.py \
  --direct-checkpoint outputs/direct_g099_seed2023/best_eval.pt \
  --persistent-checkpoint outputs/pw_v2_g0999_seed2023/best_eval.pt \
  --seed-base 20000000 \
  --episodes 200 \
  --device cpu \
  --output-dir outputs/transfer_matrix_seed2023
```

This evaluates D→D, D→PW, PW→D, and PW→PW with exactly the same holdout seeds.
It writes four complete JSON files, `matrix_summary.csv`,
`matrix_summary.json`, and `evaluation_manifest.json`. Before any cell runs, the
tool verifies that each checkpoint belongs to its declared Direct/Persistent
role and matches its source algorithm/environment fingerprints. Same-variant
cells remain strict; only the two transfer cells explicitly allow a variant
change.

### D. Final holdout evaluation

Use a range completely disjoint from training resets and training-time
evaluation. The recommended example is `seed-base=20000000` with 200 episodes.
Formal 200-episode evaluations are run manually, not during repository tests.

### E. Multi-seed aggregation

```bash
python tools/aggregate_training_runs.py \
  outputs/pw_v2_g0999_seed2023 \
  outputs/pw_v2_g0999_seed2024 \
  outputs/pw_v2_g0999_seed2025 \
  --output-dir outputs/pw_v2_training_summary
```

```bash
python tools/aggregate_holdout_results.py \
  outputs/pw_v2_g0999_seed2023/holdout_persistent.json \
  outputs/pw_v2_g0999_seed2024/holdout_persistent.json \
  outputs/pw_v2_g0999_seed2025/holdout_persistent.json \
  --output-dir outputs/pw_v2_holdout_summary
```

The training aggregator discovers all common numeric history metrics, but first
requires matching environment/algorithm fingerprints, gamma, environment count,
training budget, smoke mode, and effective hidden dimension; training seeds must
be unique. The holdout aggregator requires complete checkpoint protocol metadata,
matching source/target protocol and seed range, plus unique checkpoint training
seeds. Legacy checkpoints remain available for explicitly diagnostic evaluation,
where results carry `protocol_complete=false` and a warning, but such results are
rejected by formal holdout aggregation. Both aggregators report mean, sample
standard deviation, SEM, and a two-sided 95% Student-t interval, using a fixed t
table for df 1–30 and 1.96 only for df greater than 30.

## Discount/environment 2×2 protocol

Algorithm and environment YAML files are intentionally independent. Four
duplicate configs are unnecessary:

| Label | Environment config | Algorithm config | Gamma |
|---|---|---|---:|
| D-99 | `combat_environment.yaml` | `mappo.yaml` | 0.99 |
| D-999 | `combat_environment.yaml` | `mappo_persistent_wave.yaml` | 0.999 |
| PW-99 | `persistent_wave_v2_environment.yaml` | `mappo.yaml` | 0.99 |
| PW-999 | `persistent_wave_v2_environment.yaml` | `mappo_persistent_wave.yaml` | 0.999 |

The training entry does not bind either MAPPO config to one environment
variant. The two MAPPO YAML files remain identical except for gamma.

## Environment contract

- Four homogeneous learned Red UAVs fight four deterministic Blue UAVs.
- NED 3DOF point-mass dynamics use RK4 with `dt=0.1 s`.
- Actions are relative heading, pitch, and speed commands.
- The observation remains exactly 52 floats per Red agent.
- The weapon uses the frozen V2.3 3-D firing cone and probabilistic hit model.
- Rewards remain the frozen R1-R4 terms.
- Persistent-Wave V2 preserves the multi-wave contract and uses the existing
  ground-aware nearest-target Blue policy.

The normative Direct formulas and update order are in
`docs/environment_v2_spec.md`. Persistent mission behavior and variant identity
are documented in `docs/persistent_wave_environment_design.md`. Files under
`docs/archive/` are historical records and are not rewritten as active runtime
instructions.
# Modular Persistent-Wave formal experiments

Version protocol: modular v1 is prototype/diagnostic only; modular v2 is the
formal hardened implementation. A v1 checkpoint cannot resume into v2, and
the first 1.5M v2 experiments must start as fresh runs.

Modular MAPPO keeps raw environment outcomes separate from any algorithm-side
training reward.  Wave-balance weights are computed once from the complete
current rollout distribution, using alive-agent samples by default; they are
not re-estimated from shuffled PPO minibatches.

## Warm-start budget fairness

M6 Warm Start is not automatically fair to from-scratch Persistent-Wave
training in total data budget.  A Direct checkpoint trained with 1.505M
samples followed by 1.5M PW fine-tuning has about 3.005M total environment
interactions.  Every M6 report must therefore distinguish (A) equal PW
fine-tuning budget and (B) equal total environment-interaction budget.  A
1.5M M6 result must not be described as a fair 1.5M sample-efficiency
comparison without this qualification.

## Single-module screening after 1.5M

The first screen uses seed 2023 and the same new diagnostic evaluation seeds
for every single module.  Seeds `20000000–20000199` remain reserved for the
formal holdout and must not be used during screening.  Compare W3 completion,
average waves, raw return, Red/Blue loss, K/L, boundary and ground losses,
alive-agent wave sample fractions, best-checkpoint step, best-to-final gap and
optimization stability.  M4 is judged by raw environment outcomes, never its
shaped training return.  Only clearly promising modules proceed to 3M and
multi-seed experiments; module combinations are not part of the first screen.

## Matched all-off and M6+M8 coefficient screen

`pw_alloff_matched_1p5m.yaml` is the strict modular-v2 all-modules-off control
for the existing seed-2023 M5 run.  The two resolved configurations are equal
in every training-affecting field except `modules.wave_balancing`.

The 300k M6+M8 diagnostic uses actor-only Direct warm start and a constant
policy anchor tied to the same Direct checkpoint.  The coefficient labels are:
`c0001 = 0.001`, `c0003 = 0.003`, `c001 = 0.01`, `c003 = 0.03`, and
`c01 = 0.10`.  `pw_m6_screen_control_300k.yaml` is the matched coefficient-zero
control.  Run the complete single-GPU sequence with
`tools/run_next_stage_screening.sh`; the script performs all output-directory
and source-checkpoint checks before starting any training.

After all runs finish, `tools/analyze_next_stage_screening.py --mode all`
performs fresh paired evaluation.  It reserves seeds 35,000,000–35,000,049 for
All-Off versus M5, 35,100,000–35,100,029 for Persistent screening, and
35,200,000–35,200,029 for Direct screening.  Seeds 20,000,000–20,000,199 remain
untouched formal holdout seeds.  Anchor selection is based on the raw Pareto
table, with a descriptive Direct-win-drop reference line of 0.10; no composite
score is used.

## Three-training-seed confirmation

The frozen confirmation stage evaluates only M5 Wave Balance and M6+M8 with
constant policy-anchor coefficient 0.03.  `tools/run_multiseed_confirmation.sh`
adds training seeds 2024 and 2025; each warm-start pipeline automatically
discovers and validates the Direct source with the matching training seed.
No seed-specific YAML copies are used.

After the eight additional runs finish, use
`python -u tools/analyze_multiseed_confirmation.py --mode all --workers 4`.
M5 uses diagnostic seeds 36,000,000–36,000,099; M8 Persistent and Direct use
36,100,000–36,100,099 and 36,200,000–36,200,099.  The statistical unit is the
training seed (`n=3`), not the 100 evaluation episodes.  The 20M formal holdout
remains untouched, and M8 confirmation treats the 300k/latest checkpoint as
primary while reporting best only as a secondary stability diagnostic.

## Frozen M5 formal holdout

The final holdout compares only matched All-Off Modular MAPPO with M5 Wave
Balance for training seeds 2023, 2024, and 2025. Primary evaluation is locked
to `best_eval.pt`; `latest.pt` is a predeclared secondary best/final robustness
check. The only permitted formal episode seeds are
`20,000,000–20,000,199` (200 paired episodes per policy).

Create or verify the immutable protocol lock before evaluation:

```bash
python -u tools/prepare_formal_holdout.py
python -u tools/prepare_formal_holdout.py --verify
```

The lock is stored in `outputs/formal_holdout_protocol.json` with a companion
SHA-256 file. The CUDA-only evaluator rejects a missing or changed manifest,
changed run/config/checkpoint hashes, altered methods/seeds/checkpoint roles,
and any non-empty prior formal output. `--resume-exact` is only for continuing
an interrupted run whose manifest, task fingerprints, episode seeds, and
checkpoint hashes are identical. It cannot select individual methods or
checkpoints.

Use non-holdout seed 99,000,000 for a one-episode-per-policy CUDA smoke:

```bash
python -u tools/run_formal_holdout.py --smoke --workers 1
```

The actual formal command must be run exactly once after the smoke and final
protocol verification:

```bash
python -u tools/run_formal_holdout.py --workers 4
```

Never delete or selectively rerun formal episode caches. If an interrupted run
must be recovered, use the same command with `--resume-exact` appended.
