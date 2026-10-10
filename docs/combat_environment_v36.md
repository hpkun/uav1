# Combat Environment v3.6

V36 inherits V35 and changes shared radar range, weapon eligibility and the advantage reward. Blue decisions, close_distance=1800 m, dispersion_threshold=1500 m, its independent RNG and forced boundary returns are inherited. The 1800 m value is a policy threshold, not hardware; it remains below the new 2000 m fire range intentionally. No threshold is tuned against audit outcomes.

## Source versus adaptation

The supplied MADSAC paper description provides ATA/HA <=30 degrees, original maximum fire range 4 km, the R3 guide structure and R4 offense +0.01/+0.02/+0.10 and threat -0.015/-0.025/-0.15. It identifies ATA, AA and HA as tactical geometry. The original paper has not been independently retrieved; attribution here refers to the user-supplied description.

Our scale adaptation uses weapon range 2 km (0.4 times current nominal 5 km center separation), radar range 4 km, and radar FOV 90 degrees. A nominal 450 m/s closure leaves about 4.4 s between radar and weapon ranges versus a 2 s controller time constant. These are adapted parameters, not original MADSAC values.

Precise horizontal/vertical geometry, radar numerical range/FOV, strongest-target aggregation, fixed p_hit=0.7 and the existing strict pair-entry protocol are explicit implementation supplements. No unreported D_hit/c4/c5 hit model is invented.

## Geometry and symmetric hardware

For attacker A and target T, dx=T.x-A.x, dy=T.y-A.y, h=hypot(dx,dy), heading_LOS=atan2(dy,dx), elevation_LOS=atan2(A.z-T.z,h). ATA is absolute wrapped heading_LOS minus A.psi; HA is absolute wrapped elevation_LOS minus A.theta; AA is absolute wrapped heading_LOS minus T.psi. Angles are in [0,pi]. Rear same-heading pursuit has AA=0; head-on AA=pi; crossing AA=pi/2. NED target-above means positive LOS elevation.

At zero horizontal separation the undefined horizontal LOS uses A.psi; at full 3D coincidence elevation uses A.theta. This deterministic finite convention gives ATA=HA=0 for exact coincidence. A 1e-12 rad comparison tolerance handles floating point angular boundaries only.

Eligibility is living attacker/target, ammo>0, 0<=distance<=2000 m, ATA<=pi/6, HA<=pi/6. AA does not gate fire. One weapon instance implements both sides, same limits/p_hit=.7/ammo=6. Both colors share the same aircraft spec, dynamics, controller, action limits and 4000 m/pi/2 sensor config. Policy and inherited color-specific boundary treatment differ, not hardware performance. No geometry features are added to the 66D observation; visibility simply follows the shared new radar.

Strict pair-entry remains unchanged: consume all false-to-true entries, fire once at the nearest newly eligible target (index tie-break), maximum one attempt per attacker per step, no repeat while continuously eligible, leave/reenter for another attempt, finite ammo, simultaneous hit/kill resolution. Unselected simultaneous entries are consumed as in V35.

## Reward, without PBRS

Main reward is event + outcome + adv + safe. Event still includes individual_event + team_casualty; do not add casualty twice. Kill/loss/boundary credit remains +10/-10/-10 and casualty -2 per unique new Red loss for transition-start living slots. Outcome remains +20/-20/0 on all fixed slots. Safety remains true-3D nearest-aircraft distance 200 m, scale .2.

Adv now equals guide + tactical_offense + tactical_defense, not potential shaping. The formal V36 reward schema has no potential_gamma/potential_scale/distance_weight/angle_weight. Old tools receive zero phi diagnostics and pbrs_enabled=false; no distance/angle potential is calculated or added. Aliases stay r1=event, r2=outcome, r3=adv, r4=safe.

Guide is +.001 once if any currently radar-visible living Blue lies at 2000<d<=4000 with ATA/HA<=30 degrees. Offense considers radar-visible targets within 2000 and AA<=30 degrees, then takes the maximum of matching joint ATA/HA levels: <=5 degrees -> .10, <=15 -> .02, <=30 -> .01. Defense evaluates true Blue-attacker/Red-target geometry, requires living aircraft, d<=2000 and AA<=30, then takes the most negative joint level: <=5 -> -.15, <=15 -> -.025, <=30 -> -.015. Defensive geometry may be invisible to Red; this evaluates real danger in reward and does not expose hidden positions in actor observation. Dead slots/targets do not contribute.

Each surviving Red's tactical value is bounded [-.15,.101]. Guide is not summed over targets; offense uses max, defense min. This aggregation is our supplement for unspecified multi-target behavior, not an asserted paper formula.

Tactical values are cached after dynamics and noncombat boundary resolution, before eligibility, firing and simultaneous kills. They remain credited even when the attacker/target dies in that step. There is no terminal-zero PBRS transformation of this direct state reward. The shared step has a default empty dense hook and the original advantage method for V30–V35; older versions do not perform extra reward arithmetic or emit new diagnostics.

Fields: guide_rewards, tactical_offense_rewards, tactical_defense_rewards, tactical_total_rewards; episode_guide_total, episode_tactical_offense_total, episode_tactical_defense_total, episode_tactical_total; corresponding average_episode_*_total in evaluator and PPO training summary. Episode arrays remain available as diagnostics. No r5 is created.

## Validation conventions

Four V36 PPO config files are byte-identical to V35. Checkpoint version/config hashes remain strict; old checkpoints are not relabeled. MADDPG/MADSAC version allowlists accept 3.6 without network or hyperparameter changes.

Fixed-policy audit uses 1000 common seeds for each V35/V36 ZERO/PURSUIT group. Red PURSUIT is the same memoryless sensor-limited controller with the same old center/guard thresholds; its radar input naturally uses each environment's shared hardware. Nonmutating observers measure post-dynamics/pre-fire radar, actual eligibility entries and actual first-step fire attempts. First radar/window/attempt/kill times are conditional on occurrence with counts; first-fire angle means are weighted across actual attacker-target attempts on each episode's first firing step. Entry fractions count every new eligible pair, including unselected entries. V35 entries need not meet the new 30/30 constraints; V36 entries must.

See outputs/v36_validation/report.md for test, symmetry, historical regression, smoke and audit evidence. No formal training or parameter sweep is run.
