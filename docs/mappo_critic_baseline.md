# MAPPO Critic Baseline

This matched development comparison isolates the centralized critic structure.
Both arms use the unchanged shared local-observation MLP Gaussian Actor and the
same PPO/GAE implementation. `MAPPO` uses the shared centralized MLP value
function `V_phi(masked_global_state, masked_focal_observation)`. `MAPPO-Attn`
uses the historical two-head centralized attention critic.

Both arms are paired by training seed 5303 and deterministic evaluation
scenarios 46,000,000–46,000,049. They are not common-action-noise trajectories
after initialization because the different critic architectures consume
different amounts of initialization randomness. No wave index, mission context,
GRU, PopArt, reward modification, or modular research mechanism is included.
