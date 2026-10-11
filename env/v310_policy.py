"""Deterministic Section 2.5 pursuit; diagnostics never affect control."""
from .fixed_policy import NearestTargetPursuitPolicy


class PaperNearestTargetBluePolicy(NearestTargetPursuitPolicy):
    mode = 'GLOBAL_NEAREST_PURSUIT'

    def __init__(self, config, action_config):
        super().__init__(config, action_config)
        self.reset()

    def reset(self, seed=None):
        # Signature compatibility only: no RNG, target lock or seed-dependent state.
        self.last_targets = []
        self.pursuit_agent_steps = 0
        self.target_switches = 0

    def team_actions(self, team, targets):
        # The inherited action/nearest_target_index/action_toward are authoritative.
        actions = super().team_actions(team, targets)
        selected = [self.nearest_target_index(own, targets) if own.alive else None for own in team]
        for i, target in enumerate(selected):
            if target is not None:
                self.pursuit_agent_steps += 1
                previous = self.last_targets[i] if i < len(self.last_targets) else None
                if previous is not None and previous != target:
                    self.target_switches += 1
        self.last_targets = selected
        return actions

    def diagnostics(self):
        return dict(blue_paper_nearest_agent_steps=self.pursuit_agent_steps,
            blue_target_switches=self.target_switches,
            blue_last_target_indices=list(self.last_targets))
