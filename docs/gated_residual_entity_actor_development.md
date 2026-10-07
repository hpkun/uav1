# Gated Residual Entity-Aware Actor Development

This is a development-only representation study; it is not EA-WB Formal V1 evidence and does not claim that the new architecture is effective.

- Legacy EA (`replacement`): the entity-attention fusion fully replaces the two-layer MLP actor representation.
- Residual EA (`residual`): the stable MLP feature is retained and a zero-initialized entity adapter is added: `h = h_base + adapter(h_entity)`.
- Gated Residual EA (`gated_residual`): the same zero-initialized residual is multiplied by a learned per-agent scalar gate: `h = h_base + sigmoid(gate([h_base,h_entity])) * adapter(h_entity)`.

The adapter weight and bias start at zero. The gated variant starts at 0.05 using a zero gate weight and logit bias. Consequently, both new variants initially reproduce the ordinary MAPPO mean and state-dependent standard deviation exactly, while allowing relational information to enter gradually.

The 400k screen uses training seeds 4101-4103 and common deterministic development-validation scenarios 32000000-32000019. Scenarios 33000000-33000199 are reserved and untouched for a possible future final test. The 29M monitoring, 30M formal holdout, and 31M mechanism-diagnostic ranges are excluded from architecture selection.

Selection, after training is run separately, uses latest-at-400k and prioritizes absence of catastrophic seeds, cross-seed dispersion, safety, Average Waves, then W3. Best checkpoints and stochastic deployment are diagnostics only.
