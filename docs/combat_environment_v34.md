# Combat Environment v3.4

v3.4 inherits v3.3 and changes only automatic Blue action generation. The source design is the paper description supplied for this task: nearest pursuit, escape when locally outnumbered at close range, and battlefield boundary protection. The original paper was unavailable; this implementation does not claim to reproduce unspecified details.

The description's no-detection enemy-distribution analysis, random guerrilla pursuit and omniscient enemy centroid are deliberately omitted. They would violate the current partial-observation protocol. No current visible target means center search. There is no last-seen memory, target prediction, target lock, weapon-aware evasion or random policy.

Decision priority for each Blue aircraft is:

1. DEAD: zero action, clear evade; excluded from live mode counts.
2. BOUNDARY_GUARD: radius >=4500 m or altitude <=500/>=5500 m; clear evade and reuse v3.3 center/3000 m return action.
3. SEARCH: no currently sensor-visible living Red; clear evade and reuse v3.3 center search.
4. Existing EVADE: continue while nearest visible Red distance <1600 m, regardless of changing local counts. At >=1600 m return to PURSUIT.
5. New EVADE: nearest visible Red <=1200 m and local enemy count > local friendly count.
6. PURSUIT: reuse v3.3 LOS/elevation toward nearest visible living Red, selected by true 3D sensor distance (index breaks ties).

Enemy count uses only current visible living Red within 1500 m in 3D. Friendly count uses living Blue within 1500 m in 3D and includes self; friendly positions may be shared. Consequently 1v1 and 2v2 do not initiate evade, whereas 2v1 and 3v2 can. Enter/exit/threat/support thresholds are frozen at 1200/1600/1500/1500 m.

EVADE heads horizontally away from the horizontal centroid of current visible Red within 1600 m. Desired elevation is zero and speed remains 250 m/s in every live mode. If the escape vector is within 1e-6 m of zero, use the nearest visible target's opposite bearing; if that also degenerates, use own heading + pi. No random tie-break exists.

Every episode reset clears per-aircraft evading flags, last modes and counters. Spawn workers call the same environment reset for vector auto-reset. Live-agent steps accumulate SEARCH/PURSUIT/EVADE/GUARD counters and an episode-ever-evaded flag. Diagnostics are added only when automatic Blue actions execute. Explicit Blue action overrides bypass the policy and its diagnostics, so the complete v3.3/v3.4 info can be compared without extra fields.

Reward is inherited without overrides: individual event + team casualty, terminal +20/-20/0 fixed-slot outcome, terminal-safe tactical PBRS and existing true-3D safety. Observation 66, action 3, agents 5, sensors, initialization, ammo, hit probability, strict pair-entry fire, dynamics, controller and outcome rules are unchanged. Four v3.4 PPO configs are byte copies of v3.3. Checkpoints retain strict version/config identity checks.

The fixed-seed audit runs the same 1000 environment seeds for v3.3/v3.4 under ZERO Red and the same memoryless sensor-limited pursuit Red. It reports mode fractions over live Blue agent steps and conditional first-event step means/medians with occurrence counts. Same-step first-fire-window rate requires both events to exist; two missing events do not count. Noncombat aircraft fraction includes horizontal/ceiling exits and ground losses, without double-counting ceiling as a separate boundary loss. Results are descriptive and never trigger threshold tuning.

See `outputs/v34_validation/report.md` for results and evidence. No formal RL experiment is started.
