# DAWE-MAPPO V1

Deployment-Aligned Wave Exploration (DAWE) V1 is a development-only MAPPO intervention motivated by the completed DDD/DDS/SSD/SSS Fixed10 diagnostic. That diagnostic showed that stochastic W1/W2 execution changes W3 reach and entry conditions, so this screen tests whether lower training-time exploration in W1/W2 better aligns the visited state distribution with deterministic deployment.

V1 changes only the standard deviation of the training behavior distribution. For environment wave `w`, the base Actor produces `mu_theta(o)` and `sigma_theta(o)`, while the effective behavior distribution is `Normal(mu_theta(o), rho_w sigma_theta(o))`, followed by the existing `tanh` transform. The fixed multipliers are `rho_1=0.25`, `rho_2=0.25`, and `rho_3=1.0`. The value 0.25 is a fixed development parameter, not a claimed optimum.

The Actor mean, Actor topology, state-dependent log-standard-deviation head, Critic, reward, GAE, PPO clipping, rollout schema, environment, Blue policy, weapons, and termination logic are unchanged. Both rollout `old_log_prob` and PPO-update `new_log_prob` use the same effective distribution and the existing squashed-policy Jacobian correction. Entropy is estimated from that effective distribution. Deterministic evaluation remains exactly `tanh(mu_theta(o))`.

V1 contains no learnable variance controller, schedule, entry-state reward, wave-specific Actor, replay, recurrent memory, or new random-number generator. The base Actor may learn to increase its own sigma; the logged base/effective sigma diagnostics measure this possible compensation rather than preventing it.

The matched screen continues the same three Plain MAPPO checkpoints from 1,505,280 to 1,805,280 sampled steps. Control and DAWE differ only in method identity, branch intervention identity, and whether DAWE is enabled. Exact endpoint deterministic evaluation on the registered 44M development scenarios is primary. A single 300k screen is development evidence and is not final paper-level validation.
