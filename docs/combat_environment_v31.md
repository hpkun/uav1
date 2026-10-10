# Combat Environment v3.1

v3.1 = v3.0 + diversified initialization protocol. The default environment remains v2.3.

Only the initialization distribution changes. The 5v5 observation/action contract remains 66/3/5 (self8, ally7, enemy6). Sensor, finite-ammunition weapon, 0.7 hit probability, Blue SensorLimitedPursuitPolicy, reward, outcome/timeout, aircraft dynamics, controller and all four PPO algorithms share the existing v3.0 implementations. Four `_5v5_v31.yaml` algorithm configurations are byte-identical copies of their v3.0 counterparts.

## Initialization mathematics

Each reset uses its own NumPy environment generator, seeded by `reset(seed)`:

\[
\alpha\sim U(-\pi,\pi),\quad e_L=(\cos\alpha,\sin\alpha),\quad e_T=(-\sin\alpha,\cos\alpha).
\]

The reference centers are \(c_R=-2500e_L\), \(c_B=2500e_L\), separated by exactly 5000 m. Sample centroids can have nonzero lateral components; they are not the reference centers.

Each team independently proposes a whole vector of five samples from \(U(-2500,2500)\) m. Reject the vector unless all same-team lateral separations are at least 400 m. This is joint uniform rejection sampling conditioned on spacing, so accepted individual marginals need not themselves be uniform. Sorting is used only to check separation. Accepted values are explicitly and independently permuted using the same environment generator, then assigned to aircraft indices. At most 10,000 proposals per team are allowed; exhaustion raises an error without a fixed-formation fallback.

\[
p_i^R=c_R+y_i^Re_T,\quad p_j^B=c_B+y_j^Be_T.
\]

Every aircraft independently samples altitude \(U(2900,3100)\) m, speed \(U(215,235)\) m/s, and heading perturbation \(U(-20^\circ,20^\circ)\). Pitch is zero. Red heading is `wrap_angle(alpha + epsilon)` and Blue heading is `wrap_angle(alpha + pi + epsilon)`. Red/Blue perturbations are not paired. Maximum initial horizontal radius is \(\sqrt{2500^2+2500^2}=3535.53\) m, within the unchanged 5000 m arena.

The ±20° heading perturbation and 400 m spacing are project adaptation parameters, not asserted to be exact parameters from a paper. Altitude and speed retain the project values.

## Isolation and compatibility

`env/v31_scenario.py` contains the only new state-generation path. `CombatEnvironmentV31` inherits the entire combat protocol from `CombatEnvironmentV30` and overrides reset only. The v3.0 constructor uses the subclass version for validation; v3.0 still validates 3.0. Neither `env/scenario.py` nor the v3.0 schema or YAML changes. A strict independent v3.1 scenario schema uses the v3.0 validator through a validation-only adapter; its temporary legacy offsets never generate v3.1 states.

Version dispatch, observation dimension dispatch, RMAPPO/EA version allowlists, evaluator family aggregation and the existing bounded CUDA acceptance tool recognize 3.1. No network, optimizer, PPO, Gaussian or recurrent parameter is modified. MADSAC/MADDPG support remains unchanged. Checkpoint environment version and config SHA validation remain strict.

## Descriptive audit

Run commands in WSL Ubuntu after `conda activate uav`; CUDA is required, with no CPU fallback. Simulation workers still execute NumPy dynamics as in ordinary training.

```bash
python tools/audit_combat_v31_initialization.py \
  --resets 10000 --episodes 1000 --seed-base 32000000 --workers 16 \
  --output outputs/v31_validation/audit
python tools/smoke_formal_8v8.py \
  --env-config configs/combat_environment_v31.yaml \
  --output outputs/v31_validation/cuda_smoke
python -m pytest -q
python -m compileall -q env algorithm tools tests
```

The paired audit uses the same diagnostic environment seeds for v3.0 and v3.1; their initialization consumes different RNG draws, so subsequent weapon coins are not forced to be identical. RANDOM policy sampling uses an independent generator seeded by `environment_seed + 100000000`, identically in both versions. ZERO uses zero actions. MIRROR calls the exact existing Blue policy with Red/Blue arguments exchanged.

Timing means exclude episodes without the corresponding event and include their sample counts. First-fire-to-end means first legal weapon window to episode termination, with separate side statistics and an earliest-window statistic. Center-funnel snapshots record both teams at reset and at first detection/first window by either side. Spread is the mean horizontal distance of living aircraft from their own team centroid. First-window location is the eligible pair midpoint, nearest pair within the first qualifying side (Red tie priority). Spatial bands at 500/1000/2000 m describe concentration; they are not pass/fail criteria.

Raw completed records are retained in `episodes.jsonl.gz`; `summary.json` records all requested outcomes, timing, loss causes and spatial diagnostics. No chance-win target or parameter tuning is performed.

## Known unchanged issues

Some formal training summaries still count timeout using the legacy `red_failure_timeout` reason string. The audit uses the authoritative `info['timeout']` instead. This submission does not fix the formal logger.

Terminal PBRS still uses the actual next-state potential in `gamma * Phi_next - Phi_current`; it does not force terminal potential to zero. This behavior is shared with v3.0 and remains unchanged.

Validation results and the complete comparison are saved in `outputs/v31_validation/report.md`. No formal training or parameter sweep is started; only four bounded 512 sampled-step CUDA update checks are run.
