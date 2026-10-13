# Combat Environment v3.29

v3.29 = v3.28 + paper-style high-level increment actions. The direct class
base is `CombatEnvironmentV328`; v3.28 and earlier implementations are retained.
The environment YAML differs from v3.28 only in `environment_version`.

## Action and controller contract

Both teams use `action_to_control_v329` in the same `_advance` method. Actor
dimensions remain 56 observations / 3 actions / 5 agents. Normalized actions
mean `[a_psi, a_theta, a_v]`, with increments `[pi*a_psi, pi/3*a_theta, 50*a_v]`.

Targets are constructed before tracking:

```text
psi_d   = wrap(psi + pi*a_psi)
theta_d = clip(theta + pi/3*a_theta, -pi/3, pi/3)
v_d     = clip(v + 50*a_v, 150, 300)
psi_dot_cmd   = wrap(psi_d - psi) / 2
theta_dot_cmd = (theta_d - theta) / 2
v_dot_cmd     = (v_d - v) / 2
A_raw = cos(theta) + v*theta_dot_cmd/g
B_raw = v*cos(theta)*psi_dot_cmd/g
A = max(A_raw, 0)
if hypot(A, B_raw) > 10:
    (A, B) = (A, B_raw) * 10/hypot(A, B_raw)
else:
    B = B_raw
nz = hypot(A, B)
phi = atan2(B, A)       # zero when nz=0
nx = sin(theta) + v_dot_cmd/g
```

An exact 180-degree heading error has two equally short paths. v3.29 chooses
the sign of the requested increment at that tie (within 1e-12 radians), avoiding
a positive full-scale command reversing at the half-open wrap seam. Heading
targets still use the existing `[-pi,pi)` representation; ordinary errors use
the shortest wrapped difference. No old global angle utility is modified.

`A>=0` ensures `phi` in `[-pi/2,pi/2]`; radial projection ensures `nz<=10`
within floating-point tolerance. Heading and pitch are projected simultaneously,
without the v3.20 heading-first load allocation.

PAPER positioning: the high-level action is heading/pitch/speed increments
followed by a point-mass controller. RECONSTRUCTION / CALIBRATION: the explicit
tracking formulas, 2-second constants, clipping, nonnegative A, radial 10G
projection and exact-pi tie convention are project implementation choices.
The paper does not publish these controller equations; this is not a claim
of complete reproduction of its controller.

## Frozen mechanics

Blue uses `PaperNearestTargetBluePolicy` with the same inherited nearest-live
3D selector, NED elevation `atan2(own.z-target.z, horizontal_distance)` and
desired speed 225 m/s. It encodes heading/pitch/speed error directly, with no
load-action inverse adapter. Its diagnostic counters do not affect decisions.
`CombatEnvironmentV326.step` still caches the step-start nearest Red as the
Blue weapon candidate; no persistent lock or off-target fire is introduced.

Red firing delegates to the inherited v3.26/v3.23 chain: all eligible Blue,
nearest eligible candidate, decremented cooldown zero, automatic one-shot
attempt. A read-only wrapper records geometry at this actual firing stage;
it does not sample hits or alter attempts, ammo, cooldown or RNG.

Reward, signed 56D Observation, shared Weapon, Gaussian hit sampling, ammo=6,
cooldown=1 second, symmetric noncombat deaths, outcome, scenario, RK4 and
PointMassDynamics remain inherited from v3.28. MAPPO/MADSAC network and loss
code remain unchanged; MADSAC implementation version remains 3. Formal
algorithm configurations copy v3.28, changing only MADSAC's environment identity.
MAPPO's environment identity is supplied separately by its environment config.
Existing version/hash checkpoint validation rejects v3.28/v3.29 cross-use for
resume, evaluation and recording. MADSAC still does not support exact resume.

## Diagnostics and saturation interpretation

`training_metrics.jsonl` records per-side live-agent-weighted `mean_nz`,
`nz_saturation_fraction` (`nz>=9.9`), `mean_abs_phi` (radians), action absolute
means and saturation fractions (`abs(action)>=.99`), including all three
dimensions. Red geometry funnel fields record live/range/range+az/full
geometry/eligible counts per environment step at the firing hook. Existing
attempt/hit/kill metrics remain available. MAPPO keeps its existing return,
loss, entropy, KL, clip fraction, explained variance and policy std metrics.
No optimization logic is changed.

Run `python tools/audit_combat_v329.py` in the uav environment for 9x9 and
81x81 action grids at speeds 150/225/300 and theta=0. `grid.csv` and
`fine_grid.csv` contain both commanded actions, actual yaw/pitch rates, nz
and phi. Summary JSON contains axial responses, four corner commands,
saturation fractions, flat adjacent-rate fractions and sign diagnostics.
Thresholds describe observations; they are not invented acceptance criteria.

Important physical limitations of the specified projection:

- A large heading command scales A downward as well as B. Even zero or small
  positive pitch commands can therefore produce a negative pitch rate.
- Negative pitch commands beyond A_raw=0 share the A=0 pitch response.
- Large action regions reach the load cap and can share identical rates.
- Passing finite/load/sign tests in the unsaturated region does not establish
  that action sensitivity or learning performance is adequate.

These effects must be reported before any formal experiment; this version
does not tune constants or change the requested projection to remove them.

## Validation commands

Use WSL Ubuntu/uav where available; otherwise Windows PowerShell/uav. Training
smoke and checkpoint checks require CUDA. CPU smoke is environment-only.

```text
python -m pytest tests/test_combat_v329.py tests/test_combat_v328.py tests/test_combat_v326.py tests/test_combat_v327.py -q
python -m pytest tests/test_mappo.py tests/test_madsac.py tests/test_madsac_v3.py tests/test_madsac_v326.py -q
python tools/audit_combat_v329.py
python algorithm/train_mappo.py --env-config configs/combat_environment_v329.yaml --algorithm-config configs/mappo_5v5_v329.yaml --device cuda --seed 1 --num-envs 2 --total-sampled-steps 32 --smoke --output-dir outputs/v329_validation/mappo_cuda_smoke
python algorithm/train_madsac.py --env-config configs/combat_environment_v329.yaml --algorithm-config configs/madsac_5v5_v329.yaml --device cuda --seed 1 --num-envs 2 --total-sampled-steps 32 --smoke --output-dir outputs/v329_validation/madsac_cuda_smoke
```

Training output directories must be fresh. Existing smoke mode reduces MAPPO
width to 64 and MADSAC width to 32; formal configs retain 256/256. Do not load
smoke checkpoints for formal training. These commands are short validation,
not the proposed 300k-500k comparison or a 3M formal run.
