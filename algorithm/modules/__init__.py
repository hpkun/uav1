from .base import CapabilityModule,enabled_module_names
from .wave_context import WaveContextModule
from .recurrent_memory import RecurrentMemoryModule
from .popart import PopArtValueNormalizer
from .multi_wave_reward import MultiWaveRewardAdapter
from .wave_balancing import WaveBalancingModule
from .warm_start import WarmStartInitializer
from .curriculum import CurriculumController
from .wave_entry_curriculum import (WAVE_ENTRY_CURRICULUM_VERSION,
 WaveEntryCurriculumModule)
from .policy_anchor import PolicyAnchorRegularizer
from .advantage_priority import ADVANTAGE_PRIORITY_VERSION,AdvantagePriorityModule,capped_mean_preserving
from .ppo_stabilization import PPO_STABILIZATION_VERSION,PPOStabilizationModule
from .actor_lr_decay import ACTOR_LR_DECAY_VERSION,ActorLRDecayModule
from .wave_survival_pbrs import WaveSurvivalPotentialShapingModule
from .mission_film import MISSION_FILM_VERSION,MissionFiLMModule
from .actor_kl_guard import ACTOR_KL_GUARD_VERSION,ActorKLEpochGuardModule
from .inter_wave_credit import IWSC_MAPPO_VERSION,InterWaveCreditModule
from .counterfactual_inter_wave_credit import (CAIW_MAPPO_VERSION,CounterfactualInterWaveCreditModule,
 W1_TO_W2,W1_TO_W3,W2_TO_W3,CAIW_TASKS,TASK_SOURCE_WAVE,TASK_HEAD,binary_auroc,
 prior_corrected_probability,freshness_mask)
from .boundary_redistributed_segment_credit import (BRSC_MAPPO_VERSION,
 BoundaryRedistributedSegmentCreditModule,W1_BOUNDARY_TO_W2,W1_BOUNDARY_TO_W3,
 W2_BOUNDARY_TO_W3,BRSC_TASKS,BRSC_TASK_SOURCE_WAVE,BRSC_TASK_HEAD,
 redistribute_boundary_credit)
from .hierarchical_temporal_abstraction import (HTA_MAPPO_VERSION,
 HierarchicalTemporalAbstractionModule,ManagerTransitionBatch,
 discounted_macro_reward,smdp_boundary_masks,compute_smdp_gae)
from .hta_worker_consolidation import (HTA_WORKER_CONSOLIDATION_VERSION,
 HTAWorkerConsolidationModule)
from .sequential_wave_gradient_projection import (SWGP_MAPPO_VERSION,
 SequentialWaveGradientProjectionModule,gradient_dot,gradient_norm,gradient_cosine,
 project_nonconflicting,ordered_upstream_pairwise,natural_wave_fractions,weighted_gradient_sum)
from .team_mean_credit import TEAM_MEAN_CREDIT_VERSION,TeamMeanCreditModule
from .persistent_wave_trajectory_replay import (PWTR_MAPPO_VERSION,PWTR_SEQUENCE_LENGTH,
 PWTR_BRIDGE_HALF_LENGTH,PWTR_MIN_SEGMENT_LENGTH,PWTR_PARTITION_CAPACITY,
 PWTR_ACTOR_MAX_AGE_UPDATES,W2_INTERNAL,W3_INTERNAL,BRIDGE_12,BRIDGE_23,
 PersistentWaveTrajectoryReplayModule,wave_stratified_permutation,replay_batch_budget,replay_budget_fill_fraction,
 valid_collection_generations,transition_actor_age_mask,clipped_importance_weights,normalized_ess_and_freshness,
 vtrace_targets_and_advantages)
from .wave1_sensitivity_gating import (W1SG_MAPPO_VERSION,Wave1SensitivityGatingModule,
 normalize_importance,inverse_sqrt_gate)
from .wave_specific_actor_isolation import (WSAI_MAPPO_VERSION,EXPECTED_WSAI_CONFIG,
 WaveSpecificActorIsolationModule,pairwise_actor_l2_distances)
from .wave_specific_mean_heads import (WSMH_MAPPO_VERSION,EXPECTED_WSMH_CONFIG,
 WaveSpecificMeanHeadsModule,pairwise_mean_l2_distances)
from .actor_gradient_clipping import (ACTOR_GRAD_CLIP_VERSION,ALLOWED_ACTOR_LIMITS,
 ActorGradientClippingModule)
from .deployment_aligned_wave_exploration import (DAWE_MAPPO_VERSION,
 EXPECTED_DAWE_MULTIPLIERS,DeploymentAlignedWaveExplorationModule)
from .reference_variance import (REFERENCE_VARIANCE_VERSION,
 EXPECTED_REFERENCE_VARIANCE_CONFIG,ReferenceVarianceModule)
from .milestone_aware_retention_credit import (
 MARC_MAPPO_VERSION,MARC_MAPPO_V2_VERSION,MilestoneAwareRetentionCreditModule,compute_local_gae,
 continuation_coefficients,successful_wave_from_transition,tempered_wave_weights)

__all__=["CapabilityModule","enabled_module_names","WaveContextModule","RecurrentMemoryModule","PopArtValueNormalizer","MultiWaveRewardAdapter","WaveBalancingModule","WarmStartInitializer","CurriculumController","WAVE_ENTRY_CURRICULUM_VERSION","WaveEntryCurriculumModule","PolicyAnchorRegularizer","ADVANTAGE_PRIORITY_VERSION","AdvantagePriorityModule","capped_mean_preserving","PPO_STABILIZATION_VERSION","PPOStabilizationModule","ACTOR_LR_DECAY_VERSION","ActorLRDecayModule","WaveSurvivalPotentialShapingModule","MISSION_FILM_VERSION","MissionFiLMModule","ACTOR_KL_GUARD_VERSION","ActorKLEpochGuardModule","IWSC_MAPPO_VERSION","InterWaveCreditModule","CAIW_MAPPO_VERSION","CounterfactualInterWaveCreditModule","W1_TO_W2","W1_TO_W3","W2_TO_W3","CAIW_TASKS","TASK_SOURCE_WAVE","TASK_HEAD","binary_auroc","prior_corrected_probability","freshness_mask","BRSC_MAPPO_VERSION","BoundaryRedistributedSegmentCreditModule","W1_BOUNDARY_TO_W2","W1_BOUNDARY_TO_W3","W2_BOUNDARY_TO_W3","BRSC_TASKS","BRSC_TASK_SOURCE_WAVE","BRSC_TASK_HEAD","redistribute_boundary_credit","HTA_MAPPO_VERSION","HierarchicalTemporalAbstractionModule","ManagerTransitionBatch","discounted_macro_reward","smdp_boundary_masks","compute_smdp_gae","HTA_WORKER_CONSOLIDATION_VERSION","HTAWorkerConsolidationModule","SWGP_MAPPO_VERSION","SequentialWaveGradientProjectionModule","gradient_dot","gradient_norm","gradient_cosine","project_nonconflicting","ordered_upstream_pairwise","natural_wave_fractions","weighted_gradient_sum","TEAM_MEAN_CREDIT_VERSION","TeamMeanCreditModule","PWTR_MAPPO_VERSION","PWTR_SEQUENCE_LENGTH","PWTR_BRIDGE_HALF_LENGTH","PWTR_MIN_SEGMENT_LENGTH","PWTR_PARTITION_CAPACITY","PWTR_ACTOR_MAX_AGE_UPDATES","W2_INTERNAL","W3_INTERNAL","BRIDGE_12","BRIDGE_23","PersistentWaveTrajectoryReplayModule","wave_stratified_permutation","replay_batch_budget","clipped_importance_weights","normalized_ess_and_freshness","vtrace_targets_and_advantages","W1SG_MAPPO_VERSION","Wave1SensitivityGatingModule","normalize_importance","inverse_sqrt_gate","WSAI_MAPPO_VERSION","EXPECTED_WSAI_CONFIG","WaveSpecificActorIsolationModule","pairwise_actor_l2_distances","DAWE_MAPPO_VERSION","EXPECTED_DAWE_MULTIPLIERS","DeploymentAlignedWaveExplorationModule","MARC_MAPPO_VERSION","MilestoneAwareRetentionCreditModule","compute_local_gae","continuation_coefficients","successful_wave_from_transition","tempered_wave_weights"]
__all__.append("MARC_MAPPO_V2_VERSION")
