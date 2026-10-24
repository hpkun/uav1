"""Explicit reward/task contract for v3.50 artifacts; old identities stay exact."""


def reward_outcome_identity(version):
    if version != '3.50':
        return {}
    from .combat_v350 import REWARD_SEMANTICS, OUTCOME_SEMANTICS
    return dict(reward_semantics=REWARD_SEMANTICS, outcome_semantics=OUTCOME_SEMANTICS,
                pbrs_enabled=False, full_paper_environment_replication=False)
