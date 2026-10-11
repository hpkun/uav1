# Combat Environment v3.8: paper-scale initialization

v3.8 directly inherits v3.7. Its only environment semantic change is horizontal
spawn geometry. Arena radius remains 5000 m (10 km diameter), altitude bounds
0–6000 m, team size 5, and observation/action dimensions 66/3.

The supplied MADSAC paper description specifies a 10 km airspace diameter and
opposite-end initialization along a randomly selected diameter. It does not
specify how a five-aircraft formation is distributed near each endpoint.
The following chord placement is therefore a project 5v5 adaptation, not a
formula attributed to the paper.

At each reset, alpha is sampled uniformly from [-pi,pi]. Define
u=(cos(alpha),sin(alpha)) and t=(-sin(alpha),cos(alpha)). Nominal anchors are
Red=-5000*u and Blue=+5000*u, with nominal separation exactly 10000 m.

Both teams independently retain the v3.7 lateral rejection/permutation sampling
in [-2500,2500] m, with minimum same-team lateral separation 400 m. For each
lateral offset l:

```
q = sqrt((R - eps)^2 - l^2)
Red position  = -q*u + l*t
Blue position = +q*u + l*t
R = arena.radius = 5000 m
eps = 1e-7 m, numerical floating-point safety only
```

Each horizontal radius is R-eps. eps is not a physical parameter and is not
configurable. Simply using center_radius=5000 in the old linear formula would
produce radius sqrt(5000^2+l^2)>5000 whenever l is nonzero, placing aircraft
outside the arena. v3.8 removes scenario.center_radius entirely and rejects it
in its independent validator. The validation-only v3.7 adapter does not generate
runtime states.

Nominal anchor separation is 10 km; actual aircraft pair distances vary with
lateral offsets and altitudes. Small 3D distances above 10000 m are possible
because the airspace diameter describes the horizontal circle, not an upper
bound on 3D pair distances. Reset info exposes true minimum/mean/maximum
inter-team distance, both nominal anchors and their separation. It also reports
initial visibility and weapon eligibility counts for both colors: all zero.

Altitude remains 3000±100 m, speed 225±10 m/s, pitch zero, heading alpha for Red
and alpha+pi for Blue, each with the existing ±20 degree perturbation. The RNG
draw order exactly retains v3.7: alpha, independent Red lateral proposal and
permutation, Red altitude/speed/heading draws per aircraft, then the same Blue
sequence. Thus same-seed scalar flight states and RNG state match v3.7 at reset;
horizontal positions differ by design. Same-version same-seed resets reproduce
all states exactly. Blue keeps its separate original policy RNG.

Reward R1/R2/R3/R4/safety, zero team casualty/outcome, PBRS off, shared Gaussian
Equation (8) weapon calibration/ammo=6, 2 km fire range, ATA/HA 30 degree gates,
strict pair-entry, 4 km / 90 degree radar, Blue policy and boundary return,
Red boundary losses, dynamics/controller and observation/action semantics are
inherited unchanged. No climb, altitude, pitch, survival or other reward is added.

Four *_5v5_v38 algorithm configs are byte-identical to v3.7. Version allowlists
and diagnostic dispatch include v3.8; no model, PPO, GRU/attention, optimizer,
training/evaluation/checkpoint schedule or compact logging protocol is altered.

Small fixed-policy comparison uses seeds 1,31,10000000 for v3.7 and v3.8:
ZERO means zero Red actions; PURSUIT means the existing nearest-living-target Red
script. Blue always uses the unchanged paper-style policy. First detection is
the first state with at least one visible pair in either direction, including
reset step 0. First fire-window/attempt/kill are the earliest recorded event
across both colors. Missing events are null, never fabricated. The comparison
tests relative delay, not hardcoded seconds or improved win rates.

Validation is limited to focused/full tests, compileall, vector reset, twelve
fixed-policy episodes, and four bounded CUDA smokes of 512 transitions each,
one real update, checkpoint save/reload and two deterministic evaluation episodes
per algorithm. No sweep or 1M/3M training is run.
