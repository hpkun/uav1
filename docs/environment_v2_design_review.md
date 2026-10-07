# V2.1 Design Review (Historical Baseline for V2.2)

V2.2 supersedes only this document's separable ATA/HA launch gate with the true
3-D off-boresight cone defined in `environment_v2_spec.md`. The remaining V2.1
choices below are retained.

V2.1 removes the custom V2.0 scenario mixture, absolute command decoder,
rate-limited controller, 5-step deterministic lock, generic shaping rewards,
altitude ceiling and timeout draw. These were internally coherent, but they were
not the direct paper-constrained reconstruction requested for this benchmark.

The paper explicitly supplies the NED equations, action increments, combat
geometry variables, Eq. (7)-(8) attack form, Eq. (24) observation content,
Eq. (25) reward tiers, speed/pitch limits, 4 km range, 30 degree fire angles and
random-diameter initialization. It does not publish a complete controller, arena
radius, initial diameter length, normalization layout, attack event cadence or
all noise scales. V2.1 therefore labels the following as reconstruction choices:

- a 2 s first-order desired-state response;
- proportional A/B projection at `nz=8`;
- a hard 5 km radius and ground-only vertical destruction;
- an 8 km center diameter with disclosed formation perturbations;
- `D_hit=4000/ln(6)`, `c4=c5=1`, and independent Eq. (8) noise;
- one attempt on entry into the union of legal firing windows;
- the explicit 52-index normalization contract;
- timeout as a distinct Red mission failure.

These choices are intentionally simple, fully public and testable. They are not
presented as recovered hidden author parameters. Optimizer, network and
training-loop behavior are outside this environment design review.

The release gate is: full pytest, controller case validation, Eq. (8) Monte Carlo,
1000 reset statistics and 100-200 rule-based episodes. A fresh training sanity run
may follow only after those checks pass; V2.0 checkpoints are never resumed.
