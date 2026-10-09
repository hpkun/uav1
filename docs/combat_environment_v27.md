# Combat environment v2.7: 5v5 with the unchanged Stern opponent

The protocol is 65 observations, 3 actions and 5 agents. Relative to v2.5, only
the version and Blue policy config change. All other config sections, including
the complete scenario and formation [-600, -300, 0, 300, 600], are identical.
Relative to v2.6, only the version, team size and formation change; observation
dimension follows the existing team-size formula.

v2.6 and v2.7 instantiate exactly the same SternConversionPolicy class. Its
source is unchanged, including outward candidate scores, parity ties, locks,
reacquisition, fixed entry bearing, reciprocal heading and Convert pursuit.
Both versions share the existing strict blue-policy schema validation.
Parameters remain 250 m/s, 15 degrees, 6500 m turn range, 600 m required lateral
displacement and 2500 m conversion range. Weapons, rewards, dynamics, termination
and automatic firing are unchanged. v2.7 uses RearAspectWeaponEnvelope and
5x5 PairFireState. All five Blue states clear at reset; external Blue overrides
still bypass the fixed policy. The default remains v2.3.

All four formal PPO algorithms reuse their current 5v5 YAML files. RMAPPO and
EA-MAPPO allow-lists include 2.7. MAPPO/STEA already use validated instance
dimensions and strict environment-version/config-SHA checkpoint matching.
An 8v8 config with v2.7 is rejected, and old checkpoints are not relabeled.
MADSAC and MADDPG support ranges are unchanged. No algorithm network or training
configuration changed.

## Files

New: `configs/combat_environment_v27.yaml`, `tests/test_combat_v27.py`,
`docs/combat_environment_v27.md`.

Modified: `env/config.py`, `env/combat_env.py`,
`algorithm/rmappo/runner.py`, `algorithm/rmappo/protocol.py`,
`algorithm/ea_mappo/runner.py`, `algorithm/ea_mappo/protocol.py`,
`tests/test_combat_protocol.py`, `tests/test_formal_8v8.py`,
`tools/validate_stern_conversion.py`, `tools/smoke_formal_8v8.py`.

The geometry tool only expands its supported version list. The existing smoke
tool selects the existing formal 5v5 configs for a 5-agent environment and checks
the corresponding dimensions; its default remains v2.4 8v8.

## Bounded validation

Run in WSL Ubuntu after `conda activate uav`:

```bash
python tools/validate_stern_conversion.py \
  --env-config configs/combat_environment_v27.yaml --trials 200 --seed-base 260000 \
  --output outputs/v27_validation/geometry.json
python tools/smoke_formal_8v8.py \
  --env-config configs/combat_environment_v27.yaml \
  --output outputs/v27_validation/cuda_smoke
python -m pytest -q
python -m compileall -q algorithm env tools tests
```

The 200-trial geometric validation is the original 1v1 test, with unchanged seeds
and logic. It checks policy integration, not whether 5v5 is easier than 8v8.
No parameters are tuned from these results. Reports are saved in
`outputs/v27_validation/`. CUDA smoke caps each algorithm at 512 sampled steps,
keeps the formal networks/settings, and evaluates two episodes after strict
checkpoint reload. No formal 3M run or second difficulty-reduction route starts.

## Acceptance results

- v2.6 + v2.7 targeted tests: 75 passed, including 14 new v2.7 cases.
- Full pytest: 690 passed in 125.13 s. Compileall passed.
- Original 200 seeds: fire-window success 100%, phase completion 100%,
  median/p90 first-window steps 268/271, boundary-before-window failures 0,
  Blue exits 0, ground losses 0. All 200 straight targets leave the arena
  after success. The result matches corrected v2.6 geometry exactly.
- MAPPO, RMAPPO, EA-MAPPO and STEA-MAPPO each completed 512 CUDA sampled steps
  with 16 spawn workers, one real PPO update and two deterministic episodes
  after checkpoint reload. Gradients, parameters and values were finite;
  recurrent pre-update ratios passed their existing checks.
- SHA-256 confirmed Stern policy source, all four previous environment YAMLs
  and all four formal 5v5 algorithm YAMLs unchanged.
- No formal 3M training, parameter tuning or second route was implemented.
