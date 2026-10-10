# Combat Environment v3.5: paper-structured 3D Blue

V35 inherits V33, not V34. Its two intentional Blue changes are automatic decision generation and post-dynamics forced boundary return. The paper source is the text supplied for this task; the unavailable original paper has not been independently verified.

## Rules supplied as the paper's description

After detection, close range plus enemy numerical advantage causes escape away from the enemy cluster; otherwise pursue the nearest detected target. Without detection, choose random-target guerrilla pursuit or cluster-centroid centripetal pursuit based on enemy dispersion. Blue boundary violations force return rather than noncombat death. Red boundary violations still kill the Red aircraft.

## Implementation supplements, not original paper parameters

Close range is nearest visible true-3D distance <=1800 m. Numerical advantage is total living Red > total living Blue. Dispersion is the 3D RMS Euclidean distance of all living Red positions to their 3D centroid; its threshold is 1500 m. Speed is 250 m/s. These values are frozen and are not fitted to audit outcomes.

Each living Blue decision follows this tree:

```
visible Red exists?
  yes: nearest visible distance <=1800 AND alive Red > alive Blue?
         yes -> DETECTED_ESCAPE: away from ALL alive Red 3D centroid
         no  -> DETECTED_PURSUIT: nearest visible alive Red, index tie-break
  no:  all-alive-Red 3D RMS spread >1500?
         yes -> NO_DETECTION_GUERRILLA: uniform alive target, resampled each step
         no  -> NO_DETECTION_CENTRIPETAL: toward all-alive-Red 3D centroid
```

Dead Blue gets zero action and contributes no live mode steps. Empty living-Red sets safely return zero action and NO_TARGETS_SAFE. There is no hysteresis, pre-guard, center search, target lock, memory, cooldown, lead pursuit, aspect strategy or weapon-aware threat judgment.

All pursuit vectors use heading `atan2(dy,dx)` and elevation `atan2(-dz,hypot(dx,dy))` in NED coordinates. Escape vector is own position minus global living-Red centroid: a centroid above Blue therefore requests descending escape, and a centroid below requests climbing escape. Exactly degenerate escape (norm <=1e-12 m) falls back to away from nearest visible Red, then own heading + pi and zero elevation. Existing action_toward controls and clipping are unchanged.

Blue guerrilla sampling uses `default_rng(SeedSequence(episode_seed, spawn_key=(35,)))`. It never consumes the environment initialization/hit RNG. Episode reset reconstructs this domain-separated stream and clears modes, targets, live-step counters, flags and boundary counts. Selected target is diagnostic only, never a lock. A missing episode seed remains nondeterministic, consistent with ordinary unseeded reset; explicit seeds fully reproduce Blue sampling.

## Paper wording and information inconsistencies

Both original no-detection branches repeat wording about dispersion. The complementary interpretation here is spread > threshold for guerrilla, spread <= threshold for centripetal; this is an implementation interpretation and is not claimed to be an explicit paper statement of a concentrated formation.

The paper's no-detection branch still needs true enemy distribution. V35 intentionally permits the fixed Blue script to access all living Red positions there, and uses the global centroid/total counts for specified escape. Red learned-agent observations remain partial and unchanged. This information asymmetry is intentional; V34's sensor-only isolation test does not apply to V35's global branches. No future state or hit RNG is read.

## Forced 3D boundary return

After dynamics and before any fire-window/weapon evaluation, living Blue outside radius 5000 m is projected to radius 5000-1e-6 m and its heading becomes wrap(psi+pi). Ground contact (altitude <=0) sets altitude to 1e-6 m and theta to abs(theta). Ceiling overshoot (>6000) sets altitude to 6000-1e-6 m and theta to -abs(theta). Simultaneous horizontal/vertical violations apply both corrections. Dead Blue is skipped and never resurrected.

The fixed epsilon is numerical only and is not in YAML. Total boundary_returns counts one aircraft per transition, with separate horizontal/ground/ceiling reason counts. These are not blue_boundary_exits/ground_losses/ceiling_losses. No direct reward, casualty, kill or ammo usage is assigned to return; changed corrected geometry may naturally change inherited potential/safety and subsequent combat. Red loss precedence and all reward rules remain V33.

The shared combat step is not copied. V35 overrides noncombat resolution, reset/metric reset, and a small step wrapper for automatic actions and diagnostic fields. Without Blue boundary contact, identical explicit Red/Blue actions reproduce V33 states, observations, combat, rewards and RNG exactly. Boundary cases explicitly test the specified Blue-only divergence.

## Frozen mechanisms and verification

Initialization, sensor, weapon/ammo/hit probability/strict entry protocol, dynamics/controller, observation 66/action 3/agents 5, termination/timeout survivor comparison and V33 reward are unchanged. Four PPO V35 YAML files are byte-identical to their V33/V34 counterparts. Existing version/config SHA checkpoint contracts remain strict. All historical versions remain available; default stays V2.3.

Audit compares the same 1000 seeds in each V33/V35 ZERO and sensor-limited PURSUIT Red group, with the Red script always using the old V33 policy/config. Mode fractions use living Blue agent steps. First-event time statistics exclude missing events and include occurrence counts. Two absent first-window events do not count as a same-step first window. Results are descriptive; thresholds 1800/1500 are never tuned. See `outputs/v35_validation/report.md`.
