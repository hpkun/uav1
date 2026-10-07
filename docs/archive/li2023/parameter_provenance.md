# Archived parameter provenance for the discontinued Li et al. reconstruction

## Source hierarchy and retrieval record

The 2023 target paper is always normative. The 2022 predecessor is used only
where the 2023 paper is silent and the text identifies a reusable simulator
convention. No value below is promoted merely because the author list overlaps.

Sources inspected:

1. Shaowei Li et al., *Multi-UAV Cooperative Air Combat Decision-Making Based
   on Multi-Agent Double-Soft Actor-Critic*, Aerospace 10 (2023) 574,
   DOI `10.3390/aerospace10070574`: repository PDF and publisher HTML.
2. Shaowei Li et al., *Collaborative Decision-Making Method for Multi-UAV Based
   on Multiagent Reinforcement Learning*, IEEE Access 10 (2022) 91385-91396,
   DOI `10.1109/ACCESS.2022.3199070`: complete 12-page IEEE text, equations,
   captions and references recovered from the official IEEE document endpoint;
   the author-posted ResearchGate full-text rendering was used to cross-check
   page text.
3. OpenAlex, Crossref, DOAJ, Semantic Scholar, DBLP, author/DOI web searches,
   Beihang-domain searches, GitHub and Gitee searches. These established the
   publication lineage but yielded no attributable simulator code,
   supplementary file, PID specification, or author dissertation containing
   the missing parameters.

Important retrieval limitation: IEEE exposes 2022 Table 1 and Table 2 as GIF
assets. Their captions and URLs were recovered, but the IEEE media endpoint and
the author-copy CDN returned HTTP 403 to both command-line and browser attempts.
Consequently, the table image values were **not independently visually
verified** and are not used as parameter evidence. This is preferable to
silently treating OCR guesses as author values.

## Unknown parameter resolution table

| Item | 2023 target paper | 2022 predecessor | Other author source | Final decision | Confidence |
|---|---|---|---|---|---|
| Solution interval `dt` | 0.1 s for all models, Section 2 | Same simulator family uses the same point-mass model | None needed | 2023-PAPER | High |
| Battlefield diameter | 10 km, Table 1 | 20 km; explicitly different experiment | None | 2023-PAPER | High |
| Aircraft speed limits | 150-300 m/s, Table 1 | Table image could not be visually verified | None | 2023-PAPER | High |
| Pitch limits | `[-pi/3, pi/3]`, Table 1 | Table image could not be visually verified | None | 2023-PAPER | High |
| Roll limits | `[-pi/2, pi/2]`, Table 1 | Table image could not be visually verified | None | 2023-PAPER | High |
| Action increments | `delta psi in [-pi,pi]`, `delta theta in [-pi/3,pi/3]`, `delta v in [-50,50]`, Table 2 | Discrete action table; superseded | None | 2023-PAPER | High |
| Controller interface | Desired `[psi_d,theta_d,v_d]` becomes `[phi,nz,nx]` | Eq. (16) says the same conversion is performed by a designed PID | No code or supplement found | 2023-PAPER | High |
| Controller law | Not published | Says PID but gives no formula in body text | No attributable source found | STILL-UNSPECIFIED | High impact / low epistemic confidence |
| Yaw PID `Kp,Ki,Kd` | Not published | Not published in recoverable text | No attributable source found | STILL-UNSPECIFIED | High impact / low |
| Pitch PID `Kp,Ki,Kd` | Not published | Not published in recoverable text | No attributable source found | STILL-UNSPECIFIED | High impact / low |
| Speed PID `Kp,Ki,Kd` | Not published | Not published in recoverable text | No attributable source found | STILL-UNSPECIFIED | High impact / low |
| `nx` limits | Not published | Not published in recoverable text | No attributable source found | STILL-UNSPECIFIED | High impact / low |
| `nz` limits | Not published | Not published in recoverable text | No attributable source found | STILL-UNSPECIFIED | High impact / low |
| Controller rate limits | Not published | Not published in recoverable text | No attributable source found | STILL-UNSPECIFIED | High impact / low |
| Controller acceleration limit | Not published | Not published in recoverable text | No attributable source found | STILL-UNSPECIFIED | High impact / low |
| Sensor `c1` | Formula only, Eq. (3) | Sensor-noise model is absent | No matching author source found | STILL-UNSPECIFIED | High impact / low |
| Sensor `c2` | Formula only, Eq. (4) | Sensor-noise model is absent | No matching author source found | STILL-UNSPECIFIED | High impact / low |
| Sensor `c3` | Formula only, Eq. (5) | Sensor-noise model is absent | No matching author source found | STILL-UNSPECIFIED | High impact / low |
| Sensor `b1` | Clipping symbol only, Eq. (3) | Sensor-noise model is absent | No matching author source found | STILL-UNSPECIFIED | High impact / low |
| Sensor `b2` | Clipping symbol only, Eq. (4) | Sensor-noise model is absent | No matching author source found | STILL-UNSPECIFIED | High impact / low |
| Sensor `b3` | Clipping symbol only, Eq. (5) | Sensor-noise model is absent | No matching author source found | STILL-UNSPECIFIED | High impact / low |
| `D_firemin` | Symbol appears in Eq. (7); no value in 2023 Table 1 | Eq. (4) has only a maximum range, so zero cannot be inferred | No source found | STILL-UNSPECIFIED | High impact / low |
| `D_firemax` | 4000 m, Table 1 | Eq. (4) uses the same symbol; image value unverified | None needed | 2023-PAPER | High |
| Fire ATA maximum | 30 degrees, Table 1 | Uses yaw/pitch launch gates; image value unverified | None needed | 2023-PAPER | High |
| Fire HA maximum | 30 degrees, Table 1 | Uses yaw/pitch launch gates; image value unverified | None needed | 2023-PAPER | High |
| `D_hit` | Symbol only in Eq. (8) | Symbol only in recoverable Eq. (5); table image value unverified | No source found | STILL-UNSPECIFIED | High impact / low |
| Weapon `c4` | Symbol only in Eq. (8) | One standard Gaussian is added directly to both angles, with no coefficient | No source found; coefficient `1` is not inherited | STILL-UNSPECIFIED | High impact / low |
| Weapon `c5` | Symbol only in Eq. (8) | One standard Gaussian is added directly to both angles, with no coefficient | No source found; coefficient `1` is not inherited | STILL-UNSPECIFIED | High impact / low |
| Weapon noise correlation | One printed `epsilon_fire` is shared by the two inequalities | Same single-noise convention in Eq. (5) | None | 2023-DERIVED | Medium-high |
| Own position frame | Eq. (24) uses `p_e`; own state includes position/velocity/attitude | Eq. (17) explicitly defines own `p_e` in Earth frame `F_g` | Same model lineage | SUPPORTED-BY-AUTHOR-WORK | High |
| Teammate position frame | Final Eq. (24) paragraph says information is transformed under UAV-i body coordinates, but tuple label is `p_e` | Eq. (17) explicitly defines relative `p_b` in full observer body frame `F_b` | No contrary source | SUPPORTED-BY-AUTHOR-WORK | Medium-high |
| `F_g -> F_b` transform | No matrix printed | Fig. 1 defines full body frame; Eq. (17) names `p_b`, not a horizontal frame | Standard 3-2-1 DCM is the direct mathematical realization | 2023-DERIVED | High |
| Teammate yaw/pitch frame | Eq. (24) lists nose direction but does not define scalar transform | 2022 teammate tuple used AA/ATA/HA instead, so it does not resolve this | None | STILL-UNSPECIFIED | Medium impact / low |
| Observation dimension | Eq. (24) field expansion gives `7 + 3*6 + 4*5` | Different field tuple | None | 2023-DERIVED | High |
| Observation normalization | Not published | Not published | No source found | STILL-UNSPECIFIED | High impact / low |
| Dead-slot representation | Not published | Not published | No source found | STILL-UNSPECIFIED | Medium / low |
| Geometry tail truth | AA/ATA/HA and Fig. 2 | Author text explicitly says tail pursuit has AA=ATA=HA=0 | None | SUPPORTED-BY-AUTHOR-WORK | High |
| Blue target rule | Select nearest Red continuously and switch when nearest changes | Same nearest-target fixed rule | None | 2023-PAPER | High |
| Blue desired speed | Not published | Not published | No source found | STILL-UNSPECIFIED | Medium / low |
| Initial layout rule | Opposite ends of a random diameter | Random left/right halves; explicitly changed | None | 2023-PAPER | High |
| Formation spacing | Not published | Table image/text do not establish reusable spacing | No source found | STILL-UNSPECIFIED | High impact / low |
| Initial altitude | Not published | Table image/text do not establish reusable altitude | No source found | STILL-UNSPECIFIED | High impact / low |
| Initial speed | Not published | Table image/text do not establish reusable speed | No source found | STILL-UNSPECIFIED | High impact / low |
| Episode horizon | Not published | Not published | No source found | STILL-UNSPECIFIED | High impact / low |
| Reward formula | Eq. (25) | Different reward, explicitly superseded | None | 2023-PAPER | High |
| Reward target selection | Not stated for multiple enemies | Different reward does not resolve it | No source found | STILL-UNSPECIFIED | High impact / low |
| Critic reward semantics | Eq. (18) targets `Q_i` with `r_i`; Eq. (25) is evaluated for a specific Red UAV | Different reward does not resolve it | Markov-game vector-return definition | 2023-DERIVED | Medium-high |
| Figure-8 team return | Paper plots one return curve but does not state the multi-agent reduction | Different experiment | Current value is sum of four per-agent episode returns; mean is logged separately | 2023-DERIVED | Medium |
| Eq. (21) action gradient | Printed gradient is `nabla_{a_i} Q_i` times the agent-i policy derivative | Different discrete policy | Other joint actions are constants in each agent-i loss term | 2023-PAPER | High |
| Eq. (19)/(20) agent reduction | Both objectives print `sum_i` outside the replay expectation | Different objective | Mean over batch of the alive-agent sum | 2023-PAPER | High |
| Shared-actor gradient aggregation | Parameter sharing is explicit; exact implementation is not | Shared actor also used | Batch-mean of summed alive own-action terms into one parameter set | 2023-DERIVED | High |
| Target update timing | Algorithm 1 places target actor/critic updates inside the delayed actor branch | Different MATAC schedule | Critic-only trigger no longer updates targets | 2023-PAPER | High |
| Actor hidden layers | Two 256-unit layers, Section 4.1 | Actor hidden size 256 in predecessor | None | 2023-PAPER | High |
| Critic hidden layers | Two 256-unit layers, Section 4.1 | Different Transformer critic | None | 2023-PAPER | High |
| Attention heads | Two, Section 4.1 | Two-head Transformer encoder | None | 2023-PAPER | High |
| Actor activation | Not published | Section 4.1 states ReLU activation | No code found | SUPPORTED-BY-AUTHOR-WORK | Medium |
| Critic activation | Not published | Section 4.1 states ReLU, but critic architecture differs | No code found | SUPPORTED-BY-AUTHOR-WORK | Medium-low |
| Actor log-std bounds | Not published | Discrete actor, not applicable | No source found | STILL-UNSPECIFIED | Medium / low |
| Attention embedding wiring | Figure-level only | Different Transformer critic | No source found | STILL-UNSPECIFIED | Medium / low |
| `steps_per_update` | Algorithm 1 names threshold `T`, no value | Different MATAC schedule | No source/code found | STILL-UNSPECIFIED | High impact / low |
| `update_steps_n` | Algorithm 1 names `n`, no value | Different MATAC schedule | No source/code found | STILL-UNSPECIFIED | High impact / low |
| `policy_delay_d` | Algorithm 1 names `d`, no value | Different MATAC schedule | No source/code found | STILL-UNSPECIFIED | High impact / low |
| Algorithm-1 variable `t` mapping | Pseudocode places `t` inside episode; parallel early-terminal behavior is absent | Does not resolve 24-env behavior | Current implementation uses global synchronous vector-step counter | STILL-UNSPECIFIED | High impact / low |
| Parallel environments | 24, Section 4.1 | Not a substitute for scheduler values | None | 2023-PAPER | High |
| Evaluation action mode | Deterministic versus sampled policy is not stated | Different discrete evaluation | Current evaluator uses deterministic mean/tanh action | STILL-UNSPECIFIED | Medium / low |
| Training-cycle sampled-step mapping | Tests occur every two training cycles; cycle size is absent | Does not resolve MADSAC cycle size | Current assumption is 50,000 sampled steps per cycle, hence evaluation every 100,000 | STILL-UNSPECIFIED | Medium / low |

The current numeric values for every `STILL-UNSPECIFIED` row remain executable
placeholders, not recovered paper parameters. Candidate-only alternatives are
in `configs/sensitivity_candidates.yaml`. They are one-factor profiles, are
never combined automatically, and are explicitly labelled `CANDIDATE ONLY -
NOT PAPER VALUE`.

## 2022 versus 2023 model evolution

| Topic | 2022 predecessor | 2023 target | Consequence |
|---|---|---|---|
| Area/initialization | 20 km; Red left half, Blue right half | 10 km; opposite ends of random diameter | Use 2023 only |
| Action | Discretized desired-state increments, Table 2 | Continuous three-dimensional increments, Table 2 | Use 2023 only |
| Controller | Desired state passed through a designed PID | Same desired-to-control interface, controller unnamed | Interface confirmed; law still missing |
| Weapon launch | Yaw/pitch body-frame angles plus maximum range, no minimum | ATA/HA plus min/max range | Do not invent `D_firemin` from 2022 |
| Weapon hit | `abs(angle + epsilon_fire)` for both angles | `abs(angle + c4/c5 * epsilon_fire)` | Shared noise is supported; coefficients remain unknown |
| Observation | Own Earth-frame `p_e`; relative full-body `p_b` for others | Own, teammate and enemy tuples changed; final text names observer body coordinates | Keep 2023 fields; use full body transform for teammate relative position |
| Reward | Kill/loss/boundary plus small launch-position shaping | Eq. (25) multi-tier geometric reward | Use 2023 only |
| Algorithm | MATAC, discrete actor and Transformer critic | MADSAC, continuous actor and double attention critics | Do not inherit optimizer scheduling details |

## Final 45-dimensional observation source map

Slots remain fixed by aircraft ID; dead slots are zero placeholders. `pos_scale`
and `speed_scale` are still-unpublished normalization assumptions.

| Index | Physical quantity | Frame | Normalization | Source |
|---:|---|---|---|---|
| 0 | own north position | `F_g` NED | `/pos_scale` | Eq. (24) + 2022 Eq. (17) convention |
| 1 | own east position | `F_g` NED | `/pos_scale` | Eq. (24) + 2022 Eq. (17) convention |
| 2 | own down position | `F_g` NED | `/pos_scale` | Eq. (24) + 2022 Eq. (17) convention |
| 3 | own speed | scalar | `/speed_scale` | 2023 Eq. (24) |
| 4 | own roll `phi` | `F_g` attitude | radians, unscaled | 2023 Eq. (24) |
| 5 | own yaw `psi` | `F_g` attitude | radians, unscaled | 2023 Eq. (24) |
| 6 | own pitch `theta` | `F_g` attitude | radians, unscaled | 2023 Eq. (24) |
| 7 | teammate 1 relative body x | observer `F_b` | `/pos_scale` | 2023 body sentence + 2022 `p_b` |
| 8 | teammate 1 relative body y | observer `F_b` | `/pos_scale` | same |
| 9 | teammate 1 relative body z | observer `F_b` | `/pos_scale` | same |
| 10 | teammate 1 speed | scalar | `/speed_scale` | 2023 Eq. (24) |
| 11 | teammate 1 yaw | unresolved scalar convention | radians, unscaled | 2023 Eq. (24) |
| 12 | teammate 1 pitch | unresolved scalar convention | radians, unscaled | 2023 Eq. (24) |
| 13 | teammate 2 relative body x | observer `F_b` | `/pos_scale` | 2023 body sentence + 2022 `p_b` |
| 14 | teammate 2 relative body y | observer `F_b` | `/pos_scale` | same |
| 15 | teammate 2 relative body z | observer `F_b` | `/pos_scale` | same |
| 16 | teammate 2 speed | scalar | `/speed_scale` | 2023 Eq. (24) |
| 17 | teammate 2 yaw | unresolved scalar convention | radians, unscaled | 2023 Eq. (24) |
| 18 | teammate 2 pitch | unresolved scalar convention | radians, unscaled | 2023 Eq. (24) |
| 19 | teammate 3 relative body x | observer `F_b` | `/pos_scale` | 2023 body sentence + 2022 `p_b` |
| 20 | teammate 3 relative body y | observer `F_b` | `/pos_scale` | same |
| 21 | teammate 3 relative body z | observer `F_b` | `/pos_scale` | same |
| 22 | teammate 3 speed | scalar | `/speed_scale` | 2023 Eq. (24) |
| 23 | teammate 3 yaw | unresolved scalar convention | radians, unscaled | 2023 Eq. (24) |
| 24 | teammate 3 pitch | unresolved scalar convention | radians, unscaled | 2023 Eq. (24) |
| 25 | enemy 1 range | relative scalar | `/pos_scale` | 2023 Eq. (24) |
| 26 | enemy 1 speed | scalar | `/speed_scale` | 2023 Eq. (24) |
| 27 | enemy 1 AA | relative geometry | radians, unscaled | 2023 Eq. (24), author truth text |
| 28 | enemy 1 ATA | relative geometry | radians, unscaled | 2023 Eq. (24), author truth text |
| 29 | enemy 1 HA | relative geometry | radians, unscaled | 2023 Eq. (24), author truth text |
| 30 | enemy 2 range | relative scalar | `/pos_scale` | 2023 Eq. (24) |
| 31 | enemy 2 speed | scalar | `/speed_scale` | 2023 Eq. (24) |
| 32 | enemy 2 AA | relative geometry | radians, unscaled | 2023 Eq. (24) |
| 33 | enemy 2 ATA | relative geometry | radians, unscaled | 2023 Eq. (24) |
| 34 | enemy 2 HA | relative geometry | radians, unscaled | 2023 Eq. (24) |
| 35 | enemy 3 range | relative scalar | `/pos_scale` | 2023 Eq. (24) |
| 36 | enemy 3 speed | scalar | `/speed_scale` | 2023 Eq. (24) |
| 37 | enemy 3 AA | relative geometry | radians, unscaled | 2023 Eq. (24) |
| 38 | enemy 3 ATA | relative geometry | radians, unscaled | 2023 Eq. (24) |
| 39 | enemy 3 HA | relative geometry | radians, unscaled | 2023 Eq. (24) |
| 40 | enemy 4 range | relative scalar | `/pos_scale` | 2023 Eq. (24) |
| 41 | enemy 4 speed | scalar | `/speed_scale` | 2023 Eq. (24) |
| 42 | enemy 4 AA | relative geometry | radians, unscaled | 2023 Eq. (24) |
| 43 | enemy 4 ATA | relative geometry | radians, unscaled | 2023 Eq. (24) |
| 44 | enemy 4 HA | relative geometry | radians, unscaled | 2023 Eq. (24) |

## High-impact unresolved set

The environment should not be frozen for a 0.5M pilot until at least bounded
sensitivity checks have been run for: weapon hit scale/noise, sensor scale,
controller response, scheduler frequency and `t` mapping, episode horizon,
initialization, observation normalization, reward target selection, and
evaluation action mode. The candidate
file intentionally contains only three settings per requested subsystem and is
not a claim about ground truth.
