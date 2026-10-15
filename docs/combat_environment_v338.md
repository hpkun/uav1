# Combat Environment v3.38

V3.38 only changes enemy observation angular semantics. It inherits
`CombatEnvironmentV337` and overrides only `_observations()` plus the version
identity. The inherited chain is v338 -> v337 (AA/ATA/HA reward) -> v336
(PaperATAHAWeapon) -> v335 -> v334 -> v332 (rate-limited kinematics) -> v330.
The previous environment and its observation builder remain unchanged.

## Exact 45D layout

| Slice | Meaning | Unchanged from v3.37 |
|---|---|---|
| 0:7 | Self: normalized x, y, altitude, speed, zero phi placeholder, psi, theta | Yes |
| 7:25 | Three allies, six values per live ally in ascending index order excluding own | Yes |
| 25:45 | Four enemies, five values per live enemy in fixed enemy index order | Only last two fields change |

For enemy j, `start = 25 + 5*j`:

```python
g = engagement_geometry(own, enemy)  # env.geometry, the shared authoritative source
slot = [
    g.distance / config['relative_position_scale'],
    (enemy.v - config['speed_center']) / config['speed_scale'],
    g.aa / np.pi,
    g.ata / np.pi,
    g.ha / (np.pi / 2),
]
```

V3.37 uses `[d, v, AA, bodyAz, bodyEl]`, with bodyAz/pi and bodyEl/(pi/2).
V3.38 uses `[d, v, AA, ATA, HA]` as requested for Li et al. Eq.(24).
Angles remain signed; the observation builder does not apply abs or clipping and
does not derive alternative ATA/HA formulas. With normal geometry the three
normalized angular values lie in [-1,1]. V3.36 Weapon uses the same geometry's
ATA/HA, and v3.37 Reward uses AA/ATA/HA; this revision removes their angular
semantic mismatch with observations.

Self remains `[x/horizontal_scale, y/horizontal_scale, altitude/altitude_scale,
(v-speed_center)/speed_scale, 0, psi/pi, theta/(pi/3)]`. Ally relative positions
still use the current flight-path frame and existing scale; ally speed, psi and
theta and slot order are unchanged. A dead own agent has an all-zero 45D row;
dead ally/enemy slots remain zero. Dimensions stay obs/action/agents = 45/3/4.

## Version and configuration isolation

`configs/combat_environment_v338.yaml` is copied from v337 with only
environment_version changed. Its strict validator substitutes v337 identity in a
deep copy and calls the unchanged v337 validation chain. All numeric values and
schema restrictions remain identical, including arena/altitude 0-6000m.

The three new algorithm configs are copied from their v337 counterparts. MADSAC
and MAAC only change their explicit environment identity. The v337 MAPPO config
has no `environment` section, so its v338 copy adds only that identity section
(`3.38`, team_size 4). No network or training hyperparameter changes.

Algorithm registries add only v3.38 to existing version gates/dimension mappings.
The MAPPO/common evaluation version gates are extended so v338 retains the same
reward/Blue-policy diagnostic fields and evaluation metadata. No optimizer,
policy, critic, target, action, loss, or scheduling implementation changes.
MADSAC remains implementation-v5 with scaled self-inclusive attention,
vector-step delay, 100/4/2 reconstruction and fixed alpha=.1.

The original `env/v330_observation.py`, CombatEnvironmentV337 and all existing
configs remain untouched. V3.30-v3.37 retain their historical body-frame enemy
observations. Official checkpoint protocols reject v337/v338 cross-version
evaluation even though the dimensions match. V338 formal models require a fresh
training run, not reused v337 weights. Historical outputs are not reinterpreted.

## Explicitly frozen mechanisms

Reward and reward aggregation, PaperATAHAWeapon and hit sampling, firing/cooldown,
initial formation, Blue policy, physical transitions, alive/ammo bookkeeping and
termination all use the inherited v337 methods. The new environment does not
override reset, step, kinematics, reward, weapon, firing or outcome methods.

Flight dynamics and action-to-motion remain project simplifications; this does
not restore PointMassDynamics/RK4 or a roll/controller model. The 6000m ceiling
remains a project extension. Existing 1km weapon range, 3km tactical reward range,
continuous distance weights, action limits and other paper differences are not
addressed here. No sensor noise or other mechanism was added.

## Validation

`tests/test_combat_v338.py` covers all 16 enemy slots against exact authoritative
geometry, signed angular ranges, and a nonzero-own-pitch / unequal-altitude state
where ATA differs from bodyAz and HA differs from bodyEl. It compares all self,
ally, enemy-distance/speed/AA values bit-for-bit, including dead slots.

Two hundred reset seeds and six full episodes (seeds 0/1/2, fixed and pitch-probe
Red actions; normal automatic Blue policy) compare v337/v338 physical states,
RNG states, all non-version info fields, rewards and components, ammo, alive masks,
fire counts/hits/kills, terminated/truncated and outcomes. Only enemy columns
start+3/start+4 may differ. The weapon test also checks eligibility and 100 paired
hit draws at an eligible, nontrivial geometry, with equal RNG states after each.

`tools/validate_combat_v338.py` uses CUDA and the original formal network sizes for
MAPPO, MADSAC-v5 and MAAC. It performs actor/critic forward passes, reset/16 steps,
checkpoint save/protocol/load, crossed observation-identity rejection, and one
full deterministic evaluation episode per algorithm. It performs zero optimizer
steps and never calls a training runner's run method. It fails if CUDA is absent.

```powershell
conda activate uav
python -m pytest -q tests/test_combat_v338.py tests/test_combat_v335.py tests/test_combat_v336.py tests/test_combat_v337.py
python -u tools/validate_combat_v338.py --output-dir outputs/v338_validation/new_smoke
```

Historical mechanisms and algorithm math files are protected by before/after
SHA256 manifests. Changes in existing registry/metrics files are verified to be
only the new v338 dispatch or whitelist entries. Results and the ordered report
are saved in `outputs/v338_validation/`. No formal training is started.
