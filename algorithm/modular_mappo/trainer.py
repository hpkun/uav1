"""MAPPO with opt-in capability modules and a genuine contiguous-sequence GRU path."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any
import hashlib,json,random
import numpy as np
import torch
from torch import nn
from torch.distributions import kl_divergence
from algorithm.mappo.trainer import compute_gae,masked_mean,MAPPO_IMPL_VERSION
from algorithm.modules import (WaveContextModule,RecurrentMemoryModule,PopArtValueNormalizer,
 MultiWaveRewardAdapter,WaveBalancingModule,WarmStartInitializer,CurriculumController,
 WaveEntryCurriculumModule,WAVE_ENTRY_CURRICULUM_VERSION,PolicyAnchorRegularizer,enabled_module_names)
from algorithm.modules import (AdvantagePriorityModule,PPOStabilizationModule,
 ADVANTAGE_PRIORITY_VERSION,PPO_STABILIZATION_VERSION)
from algorithm.modules import ActorLRDecayModule,ACTOR_LR_DECAY_VERSION
from algorithm.modules import WaveSurvivalPotentialShapingModule
from algorithm.modules import MissionFiLMModule,MISSION_FILM_VERSION
from algorithm.modules import ActorKLEpochGuardModule,ACTOR_KL_GUARD_VERSION
from algorithm.modules import (InterWaveCreditModule,IWSC_MAPPO_VERSION,CounterfactualInterWaveCreditModule,
 CAIW_MAPPO_VERSION,CAIW_TASKS,TASK_SOURCE_WAVE,TASK_HEAD,W1_TO_W2,W1_TO_W3,W2_TO_W3,
 binary_auroc,prior_corrected_probability,freshness_mask)
from algorithm.modules import (BoundaryRedistributedSegmentCreditModule,BRSC_MAPPO_VERSION,BRSC_TASKS,
 BRSC_TASK_SOURCE_WAVE,BRSC_TASK_HEAD,W1_BOUNDARY_TO_W2,W1_BOUNDARY_TO_W3,W2_BOUNDARY_TO_W3,
 redistribute_boundary_credit)
from algorithm.modules import (HierarchicalTemporalAbstractionModule,HTA_MAPPO_VERSION,
 compute_smdp_gae)
from algorithm.modules import (HTAWorkerConsolidationModule,
 HTA_WORKER_CONSOLIDATION_VERSION)
from algorithm.modules import (SequentialWaveGradientProjectionModule,SWGP_MAPPO_VERSION,
 gradient_dot,gradient_norm,gradient_cosine,ordered_upstream_pairwise,
 natural_wave_fractions,weighted_gradient_sum)
from algorithm.modules.hierarchical_temporal_abstraction import (HTA_MANAGER_TORCH_SEED_XOR,
 HTA_MANAGER_NUMPY_SEED_XOR)
from algorithm.modules import TeamMeanCreditModule,TEAM_MEAN_CREDIT_VERSION
from algorithm.modules import (PersistentWaveTrajectoryReplayModule,PWTR_MAPPO_VERSION,
 replay_batch_budget,replay_budget_fill_fraction,clipped_importance_weights,normalized_ess_and_freshness,
 vtrace_targets_and_advantages,W2_INTERNAL,W3_INTERNAL,BRIDGE_12,BRIDGE_23,
 transition_actor_age_mask)
from algorithm.modules import Wave1SensitivityGatingModule,W1SG_MAPPO_VERSION
from algorithm.modules import (WaveSpecificActorIsolationModule,WSAI_MAPPO_VERSION,
 pairwise_actor_l2_distances)
from algorithm.modules import (WaveSpecificMeanHeadsModule,WSMH_MAPPO_VERSION,
 pairwise_mean_l2_distances)
from algorithm.modules import (ActorGradientClippingModule,ACTOR_GRAD_CLIP_VERSION)
from algorithm.modules import (DeploymentAlignedWaveExplorationModule,DAWE_MAPPO_VERSION)
from algorithm.modules import (ReferenceVarianceModule,REFERENCE_VARIANCE_VERSION)
from algorithm.modules import (MARC_MAPPO_VERSION,MilestoneAwareRetentionCreditModule,
 compute_local_gae)
from .networks import (ModularMAPPOActor,StateMemoryMAPPOActor,ModularCentralizedCritic,InterWaveStateQualityCritic,
 InterWaveActionOutcomeCritic,BoundaryStateOutcomeCritic,HierarchicalManagerActor)
from .buffer import ModularRolloutBatch,contiguous_chunks,recurrent_alive_mean,wave_segmented_chunks,episode_contiguous_chunks

# Version 2 is the formal hardened implementation. Version 1 checkpoints use
# prototype recurrent/weighting semantics and are diagnostic-only artifacts.
MODULAR_MAPPO_IMPL_VERSION=2

def stable_ratio_terms(new_log_prob,old_log_prob):
 """Return PPO log-ratio/ratio without reconstructing log-ratio from exp()."""
 if not torch.all(torch.isfinite(new_log_prob)):raise FloatingPointError("non-finite new_log_prob")
 if not torch.all(torch.isfinite(old_log_prob)):raise FloatingPointError("non-finite old_log_prob")
 log_ratio=new_log_prob-old_log_prob
 if not torch.all(torch.isfinite(log_ratio)):raise FloatingPointError("non-finite log_ratio")
 ratio=log_ratio.exp()
 if not torch.all(torch.isfinite(ratio)):raise FloatingPointError("non-finite ratio")
 return log_ratio,ratio

def aggregate_update_rows(rows,clip_ratio):
 """Aggregate sample diagnostics over every alive sample in the PPO update."""
 if not rows:raise RuntimeError("no modular PPO metric rows")
 counts=np.asarray([row["_valid_count"] for row in rows],dtype=np.float64);total=float(counts.sum())
 if total<=0:raise FloatingPointError("no valid alive PPO samples")
 private={"_valid_count","_ratio_values","_log_ratio_values"}
 weighted={"actor_loss","weighted_actor_loss","value_loss","weighted_value_loss","entropy","approx_kl","clip_fraction"}
 result={}
 for key in rows[0]:
  if key in private or key.startswith("ratio_") or key.startswith("log_ratio_") or key=="max_abs_log_ratio":continue
  values=np.asarray([row[key] for row in rows],dtype=np.float64)
  result[key]=float(np.sum(values*counts)/total) if key in weighted else float(values.mean())
 ratio=np.concatenate([row["_ratio_values"] for row in rows]).astype(np.float64,copy=False)
 log_ratio=np.concatenate([row["_log_ratio_values"] for row in rows]).astype(np.float64,copy=False)
 if ratio.size!=int(total) or log_ratio.size!=int(total):raise RuntimeError("ratio diagnostic sample-count mismatch")
 if not np.all(np.isfinite(log_ratio)):raise FloatingPointError("non-finite log_ratio diagnostics")
 if not np.all(np.isfinite(ratio)):raise FloatingPointError("non-finite ratio diagnostics")
 kl=(ratio-1.0)-log_ratio
 if not np.all(np.isfinite(kl)):raise FloatingPointError("non-finite approx_kl samples")
 underflow=(ratio==0.0)&np.isfinite(log_ratio)
 result.update({
  "approx_kl":float(kl.mean()),
  "clip_fraction":float((np.abs(ratio-1.0)>clip_ratio).mean()),
  "ratio_mean":float(ratio.mean()),"ratio_std":float(ratio.std()),
  "ratio_p1":float(np.quantile(ratio,.01)),"ratio_p50":float(np.quantile(ratio,.5)),"ratio_p99":float(np.quantile(ratio,.99)),
  "ratio_min":float(ratio.min()),"ratio_max":float(ratio.max()),
  "log_ratio_min":float(log_ratio.min()),"log_ratio_max":float(log_ratio.max()),"max_abs_log_ratio":float(np.abs(log_ratio).max()),
  "ratio_underflow_count":int(underflow.sum()),"ratio_underflow_fraction":float(underflow.mean()),"ratio_sample_count":int(total),
 })
 return result

def asymmetric_tactical_projection(tactical,auxiliary,epsilon=1e-12):
 """Project only the auxiliary gradient when it conflicts with tactical PPO."""
 dot=sum((a*b).sum() for a,b in zip(tactical,auxiliary));norm_sq=sum(a.square().sum() for a in tactical)
 return ([b-dot/(norm_sq+epsilon)*a for a,b in zip(tactical,auxiliary)] if bool(dot.detach()<0) else auxiliary),dot

def normalize_iw_deltas(delta,waves,ready):
 """Normalize team-state deltas independently per ready source wave."""
 normalized=delta.clone();active=torch.zeros_like(waves,dtype=torch.bool)
 for wave in (1,2):
  mask=waves==wave;values=delta[mask]
  if ready.get(wave,False) and values.numel() and float(values.std(unbiased=False))>=1e-8:
   normalized[mask]=(values-values.mean())/(values.std(unbiased=False)+1e-8);active[mask]=True
 return normalized,active

def balanced_iw_losses(losses):
 return torch.stack(list(losses)).mean()

def combine_iw_actor_loss(surrogate,alive_mask,active_states,source_waves,wave_balanced):
 """Combine active IW PPO samples using the configured minibatch semantics."""
 active_mask=alive_mask*active_states[:,None].to(alive_mask.dtype)
 if not bool(active_states.any()):return surrogate.sum()*0
 if not wave_balanced:return -masked_mean(surrogate,active_mask)
 losses=[]
 for wave in (1,2):
  wave_states=active_states & (source_waves==wave)
  if wave_states.any():losses.append(-masked_mean(surrogate,alive_mask*wave_states[:,None].to(alive_mask.dtype)))
 return balanced_iw_losses(losses)

def caiw_trust_cap(tactical,auxiliary,ratio_cap=.25,epsilon=1e-12):
 """Bound the projected auxiliary norm relative to the untouched tactical norm."""
 nt=torch.sqrt(sum(g.square().sum() for g in tactical));na=torch.sqrt(sum(g.square().sum() for g in auxiliary))
 scale=torch.zeros((),device=nt.device) if float(nt)<epsilon else torch.minimum(torch.ones((),device=nt.device),ratio_cap*nt/(na+epsilon))
 return [scale*g for g in auxiliary],scale,nt,na

def antithetic_latent_actions(mean,std,caiw_rng,samples=4):
 """Generate K=4 antithetic latent alternatives without advancing torch RNG."""
 if samples!=4:raise ValueError("CAIW V2 requires exactly four counterfactual samples")
 eps=torch.as_tensor(caiw_rng.standard_normal((2,*mean.shape)),dtype=mean.dtype,device=mean.device)
 return torch.stack((mean+std*eps[0],mean-std*eps[0],mean+std*eps[1],mean-std*eps[1]),1)

class ModularMAPPOTrainer:
 def __init__(self,observation_dim=52,action_dim=3,num_agents=4,hidden_dim=256,attention_heads=2,
  actor_learning_rate=3e-4,critic_learning_rate=3e-4,gamma=.99,gae_lambda=.95,clip_ratio=.2,
  value_loss_coefficient=.5,entropy_coefficient=.01,max_grad_norm=.5,ppo_epochs=10,minibatch_size=512,
  normalize_advantages=True,clip_value_loss=True,device="cpu",seed=0,actor_activation="relu",
  critic_activation="relu",log_std_min=-5.,log_std_max=2.,modules_config=None,
  total_sampled_steps=1,development_method=None):
  self.development_method=development_method
  self.device=torch.device(device)
  if self.device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
  self.num_agents=int(num_agents);self.gamma=float(gamma);self.gae_lambda=float(gae_lambda);self.clip_ratio=float(clip_ratio)
  self.value_loss_coefficient=float(value_loss_coefficient);self.entropy_coefficient=float(entropy_coefficient);self.max_grad_norm=float(max_grad_norm)
  self.ppo_epochs=int(ppo_epochs);self.minibatch_size=int(minibatch_size);self.normalize_advantages=bool(normalize_advantages);self.clip_value_loss=bool(clip_value_loss)
  self.rng=np.random.default_rng(seed);torch.manual_seed(seed)
  if self.device.type=="cuda":torch.cuda.manual_seed_all(seed)
  self.modules_config=deepcopy(modules_config or {})
  self.wave_context=WaveContextModule(self.modules_config.get("wave_context"));self.recurrent=RecurrentMemoryModule(self.modules_config.get("recurrent_memory"))
  self.popart=PopArtValueNormalizer(self.modules_config.get("popart")).to(self.device);self.reward_adapter=MultiWaveRewardAdapter(self.modules_config.get("multi_wave_reward"))
  self.wave_survival_pbrs=WaveSurvivalPotentialShapingModule(self.modules_config.get("wave_survival_pbrs"),self.gamma)
  self.wave_balance=WaveBalancingModule(self.modules_config.get("wave_balancing"));self.warm_start=WarmStartInitializer(self.modules_config.get("warm_start"))
  self.curriculum=CurriculumController(self.modules_config.get("curriculum"));self.anchor=PolicyAnchorRegularizer(self.modules_config.get("policy_anchor"))
  self.wave_entry_curriculum=WaveEntryCurriculumModule(self.modules_config.get("wave_entry_curriculum"),seed)
  self.advantage_priority=AdvantagePriorityModule(self.modules_config.get("advantage_priority"))
  self.ppo_stabilization=PPOStabilizationModule(self.modules_config.get("ppo_stabilization"))
  self.actor_lr_decay=ActorLRDecayModule(self.modules_config.get("actor_lr_decay"))
  self.mission_film=MissionFiLMModule(self.modules_config.get("mission_film"))
  self.actor_kl_guard=ActorKLEpochGuardModule(self.modules_config.get("actor_kl_guard"))
  self.inter_wave_credit=InterWaveCreditModule(self.modules_config.get("inter_wave_credit"))
  self.counterfactual_inter_wave_credit=CounterfactualInterWaveCreditModule(self.modules_config.get("counterfactual_inter_wave_credit"))
  self.boundary_redistributed_segment_credit=BoundaryRedistributedSegmentCreditModule(self.modules_config.get("boundary_redistributed_segment_credit"))
  self.hierarchical_temporal_abstraction=HierarchicalTemporalAbstractionModule(self.modules_config.get("hierarchical_temporal_abstraction"))
  self.hta_worker_consolidation=HTAWorkerConsolidationModule(self.modules_config.get("hta_worker_consolidation"))
  self.sequential_wave_gradient_projection=SequentialWaveGradientProjectionModule(self.modules_config.get("sequential_wave_gradient_projection"))
  self.team_mean_credit=TeamMeanCreditModule(self.modules_config.get("team_mean_credit"))
  self.persistent_wave_trajectory_replay=PersistentWaveTrajectoryReplayModule(self.modules_config.get("persistent_wave_trajectory_replay"),seed)
  self.wave1_sensitivity_gating=Wave1SensitivityGatingModule(self.modules_config.get("wave1_sensitivity_gating"))
  self.wave_specific_actor_isolation=WaveSpecificActorIsolationModule(self.modules_config.get("wave_specific_actor_isolation"))
  self.wave_specific_mean_heads=WaveSpecificMeanHeadsModule(self.modules_config.get("wave_specific_mean_heads"))
  self.actor_gradient_clipping=ActorGradientClippingModule(self.modules_config.get("actor_gradient_clipping"))
  self.deployment_aligned_wave_exploration=DeploymentAlignedWaveExplorationModule(
   self.modules_config.get("deployment_aligned_wave_exploration"))
  self.reference_variance=ReferenceVarianceModule(self.modules_config.get("reference_variance"))
  self.milestone_aware_retention_credit=MilestoneAwareRetentionCreditModule(
   self.modules_config.get("milestone_aware_retention_credit"),seed)
  self.reference_variance_actor=None
  self.reference_variance_initial_sha256=None
  self.reference_variance_source_checkpoint_sha256=None
  self.reference_variance_source_sampled_steps=None
  self.reference_variance_source_training_seed=None
  self._pending_reference_variance_branch=None
  self.entity_attention_config=deepcopy(self.modules_config.get("entity_attention",{}))
  self.entity_attention_enabled=bool(self.entity_attention_config.get("enabled",False))
  self.frozen_base_policy_entity_enabled=self.entity_attention_enabled and self.entity_attention_config.get("mode","replacement") in {"frozen_base_mean_residual","frozen_base_dual_bounded_mean_residual"}
  # Backward-compatible public flag used by existing FBMR V1 code and audits.
  self.fbmr_enabled=self.frozen_base_policy_entity_enabled
  self.total_sampled_steps=int(total_sampled_steps)
  if self.total_sampled_steps<=0:raise ValueError("total_sampled_steps must be positive")
  if self.milestone_aware_retention_credit.enabled:
   enabled=set(enabled_module_names(self.modules_config));required={"actor_lr_decay","milestone_aware_retention_credit"}
   segmented=(self.recurrent.wave_segmented and development_method=="marc_mappo_wsgru_v1") or (self.recurrent.state_memory and development_method=="marc_mappo_state_memory_v1")
   if segmented:
    required.add("recurrent_memory")
    if self.milestone_aware_retention_credit.version!=2 or self.milestone_aware_retention_credit.deployment_distill_coefficient!=0 or self.recurrent.hidden_dim!=128 or self.recurrent.sequence_length!=32:
     raise ValueError("experimental wave-segmented MARC requires V2, zero retention, 128 hidden and 32 BPTT")
   if enabled!=required:raise ValueError(f"MARC V{self.milestone_aware_retention_credit.version} requires exact enabled modules: {sorted(required)}")
   if not self.actor_lr_decay.enabled:raise ValueError("MARC requires actor_lr_decay")
   if any((self.recurrent.enabled and not segmented,self.hierarchical_temporal_abstraction.enabled,self.wave_balance.enabled,
           self.wave_entry_curriculum.enabled,self.actor_kl_guard.enabled,self.inter_wave_credit.enabled,
           self.counterfactual_inter_wave_credit.enabled,self.boundary_redistributed_segment_credit.enabled,
           self.mission_film.enabled,self.wave_context.enabled,self.entity_attention_enabled,
           self.advantage_priority.enabled,self.ppo_stabilization.enabled,self.popart.enabled,
           self.curriculum.enabled,self.anchor.enabled,self.warm_start.enabled,self.reward_adapter.enabled,
           self.wave_survival_pbrs.enabled,self.team_mean_credit.enabled,
           self.persistent_wave_trajectory_replay.enabled,self.sequential_wave_gradient_projection.enabled,
           self.wave1_sensitivity_gating.enabled,self.wave_specific_actor_isolation.enabled,
           self.wave_specific_mean_heads.enabled,self.deployment_aligned_wave_exploration.enabled,
           self.reference_variance.enabled)):
    raise ValueError("MARC requires an otherwise Plain feed-forward Actor/Critic")
  if self.recurrent.wave_segmented and (development_method!="marc_mappo_wsgru_v1" or not self.milestone_aware_retention_credit.enabled):
   raise ValueError("wave_segmented_actor_gru requires isolated marc_mappo_wsgru_v1 method")
  if self.recurrent.state_memory and (development_method!="marc_mappo_state_memory_v1" or not self.milestone_aware_retention_credit.enabled):
   raise ValueError("State Memory requires isolated marc_mappo_state_memory_v1 method")
  if self.entity_attention_enabled and (self.recurrent.enabled or self.wave_context.enabled):raise ValueError("entity attention v1 is incompatible with recurrent memory and wave context")
  if self.ppo_stabilization.enabled and self.recurrent.enabled:raise ValueError("PPO stabilization v1 requires the feed-forward update path")
  if self.actor_kl_guard.enabled and self.recurrent.enabled:raise ValueError("actor_kl_guard requires the feed-forward update path")
  if self.actor_kl_guard.enabled and self.ppo_stabilization.enabled:raise ValueError("actor_kl_guard and ppo_stabilization are mutually exclusive")
  if self.actor_lr_decay.enabled and self.ppo_stabilization.enabled:raise ValueError("actor_lr_decay and ppo_stabilization are mutually exclusive")
  if self.actor_lr_decay.enabled and abs(self.actor_lr_decay.start_lr-float(actor_learning_rate))>1e-15:raise ValueError("actor_lr_decay start_lr must match the configured base actor learning rate")
  if self.sequential_wave_gradient_projection.enabled:
   enabled=set(enabled_module_names(self.modules_config));required={"actor_lr_decay","sequential_wave_gradient_projection"}
   if enabled!=required:raise ValueError(f"SWGP V1 requires exact enabled modules: {sorted(required)}")
   if not self.actor_lr_decay.enabled:raise ValueError("SWGP V1 requires actor_lr_decay")
   if any((self.recurrent.enabled,self.hierarchical_temporal_abstraction.enabled,self.wave_balance.enabled,
           self.wave_entry_curriculum.enabled,self.actor_kl_guard.enabled,self.inter_wave_credit.enabled,
           self.counterfactual_inter_wave_credit.enabled,self.boundary_redistributed_segment_credit.enabled,
           self.mission_film.enabled,self.wave_context.enabled,self.entity_attention_enabled,
           self.advantage_priority.enabled,self.ppo_stabilization.enabled,self.popart.enabled,
           self.curriculum.enabled,self.anchor.enabled,self.warm_start.enabled,
           self.reward_adapter.enabled,self.wave_survival_pbrs.enabled)):
    raise ValueError("SWGP V1 requires an otherwise Plain feed-forward Actor/Critic")
  if self.curriculum.enabled and self.wave_entry_curriculum.enabled:raise ValueError("curriculum and wave_entry_curriculum are mutually exclusive")
  if self.team_mean_credit.enabled:
   enabled=set(enabled_module_names(self.modules_config));required={"actor_lr_decay","team_mean_credit"}
   if enabled!=required:raise ValueError(f"team_mean_credit requires exact enabled modules: {sorted(required)}")
   if not self.actor_lr_decay.enabled:raise ValueError("team_mean_credit requires actor_lr_decay")
  if self.persistent_wave_trajectory_replay.enabled:
   enabled=set(enabled_module_names(self.modules_config));required={"actor_lr_decay","persistent_wave_trajectory_replay"}
   if self.wave1_sensitivity_gating.enabled:required.add("wave1_sensitivity_gating")
   if enabled!=required:raise ValueError(f"PWTR V1 requires exact enabled modules: {sorted(required)}")
   if not self.actor_lr_decay.enabled:raise ValueError("PWTR V1 requires actor_lr_decay")
   if any((self.recurrent.enabled,self.hierarchical_temporal_abstraction.enabled,self.wave_balance.enabled,
           self.wave_entry_curriculum.enabled,self.actor_kl_guard.enabled,self.inter_wave_credit.enabled,
           self.counterfactual_inter_wave_credit.enabled,self.boundary_redistributed_segment_credit.enabled,
           self.mission_film.enabled,self.wave_context.enabled,self.entity_attention_enabled,
           self.advantage_priority.enabled,self.ppo_stabilization.enabled,self.popart.enabled,
           self.curriculum.enabled,self.anchor.enabled,self.warm_start.enabled,
           self.reward_adapter.enabled,self.wave_survival_pbrs.enabled,self.team_mean_credit.enabled)):
    raise ValueError("PWTR V1 requires an otherwise Plain feed-forward Actor/Critic")
  if self.wave1_sensitivity_gating.enabled:
   p=self.persistent_wave_trajectory_replay
   if not p.enabled or (p.fresh_wave_stratification,p.replay_enabled,p.replay_source,p.priority_enabled,p.bridge_enabled,p.actor_replay,p.critic_replay)!=(True,True,"current",False,False,True,False):raise ValueError("W1SG V1 requires exact PWTR CurrentActor mode")
  if self.wave_specific_actor_isolation.enabled:
   enabled=set(enabled_module_names(self.modules_config));required={"actor_lr_decay","wave_specific_actor_isolation"}
   if enabled!=required:raise ValueError(f"WSAI V1 requires exact enabled modules: {sorted(required)}")
   if not self.actor_lr_decay.enabled:raise ValueError("WSAI V1 requires actor_lr_decay")
   if any((self.recurrent.enabled,self.hierarchical_temporal_abstraction.enabled,self.wave_balance.enabled,
           self.wave_entry_curriculum.enabled,self.actor_kl_guard.enabled,self.inter_wave_credit.enabled,
           self.counterfactual_inter_wave_credit.enabled,self.boundary_redistributed_segment_credit.enabled,
           self.mission_film.enabled,self.wave_context.enabled,self.entity_attention_enabled,
           self.advantage_priority.enabled,self.ppo_stabilization.enabled,self.popart.enabled,
           self.curriculum.enabled,self.anchor.enabled,self.warm_start.enabled,self.reward_adapter.enabled,
           self.wave_survival_pbrs.enabled,self.team_mean_credit.enabled,self.persistent_wave_trajectory_replay.enabled,
           self.sequential_wave_gradient_projection.enabled,self.wave1_sensitivity_gating.enabled)):
    raise ValueError("WSAI V1 requires an otherwise Plain feed-forward Actor/Critic")
  if self.wave_specific_mean_heads.enabled:
   enabled=set(enabled_module_names(self.modules_config));required={"actor_lr_decay","wave_specific_mean_heads"}
   if enabled!=required:raise ValueError(f"WSMH V1 requires exact enabled modules: {sorted(required)}")
   if not self.actor_lr_decay.enabled:raise ValueError("WSMH V1 requires actor_lr_decay")
   if self.wave_specific_actor_isolation.enabled:raise ValueError("WSMH and WSAI are mutually exclusive")
   forbidden=(self.recurrent.enabled,self.hierarchical_temporal_abstraction.enabled,self.wave_balance.enabled,
    self.wave_entry_curriculum.enabled,self.actor_kl_guard.enabled,self.inter_wave_credit.enabled,
    self.counterfactual_inter_wave_credit.enabled,self.boundary_redistributed_segment_credit.enabled,
    self.mission_film.enabled,self.wave_context.enabled,self.entity_attention_enabled,self.advantage_priority.enabled,
    self.ppo_stabilization.enabled,self.popart.enabled,self.curriculum.enabled,self.anchor.enabled,self.warm_start.enabled,
    self.reward_adapter.enabled,self.wave_survival_pbrs.enabled,self.team_mean_credit.enabled,
    self.persistent_wave_trajectory_replay.enabled,self.sequential_wave_gradient_projection.enabled,
    self.wave1_sensitivity_gating.enabled)
   if any(forbidden):raise ValueError("WSMH V1 requires an otherwise Plain feed-forward Actor/Critic")
  if self.actor_gradient_clipping.enabled:
   enabled=set(enabled_module_names(self.modules_config));required={"actor_lr_decay","actor_gradient_clipping"}
   if self.deployment_aligned_wave_exploration.enabled:required.add("deployment_aligned_wave_exploration")
   if self.reference_variance.enabled:required.add("reference_variance")
   if enabled!=required:raise ValueError(f"Actor gradient clipping V1 requires exact enabled modules: {sorted(required)}")
   if not self.actor_lr_decay.enabled:raise ValueError("Actor gradient clipping V1 requires actor_lr_decay")
   forbidden=(self.recurrent.enabled,self.hierarchical_temporal_abstraction.enabled,self.wave_balance.enabled,
    self.wave_entry_curriculum.enabled,self.actor_kl_guard.enabled,self.inter_wave_credit.enabled,
    self.counterfactual_inter_wave_credit.enabled,self.boundary_redistributed_segment_credit.enabled,
    self.mission_film.enabled,self.wave_context.enabled,self.entity_attention_enabled,self.advantage_priority.enabled,
    self.ppo_stabilization.enabled,self.popart.enabled,self.curriculum.enabled,self.anchor.enabled,self.warm_start.enabled,
    self.reward_adapter.enabled,self.wave_survival_pbrs.enabled,self.team_mean_credit.enabled,
    self.persistent_wave_trajectory_replay.enabled,self.sequential_wave_gradient_projection.enabled,
    self.wave1_sensitivity_gating.enabled,self.wave_specific_actor_isolation.enabled,self.wave_specific_mean_heads.enabled)
   if any(forbidden):raise ValueError("Actor gradient clipping V1 requires an otherwise Plain feed-forward Actor/Critic")
   if abs(self.max_grad_norm-.5)>1e-15 or abs(self.actor_gradient_clipping.critic_max_grad_norm-self.max_grad_norm)>1e-15:raise ValueError("Actor gradient clipping V1 requires unchanged critic max_grad_norm=0.5")
  if self.deployment_aligned_wave_exploration.enabled:
   enabled=set(enabled_module_names(self.modules_config));required={"actor_lr_decay","actor_gradient_clipping","deployment_aligned_wave_exploration"}
   if enabled!=required:raise ValueError(f"DAWE V1 requires exact enabled modules: {sorted(required)}")
   if not self.actor_gradient_clipping.enabled or abs(self.actor_gradient_clipping.actor_max_grad_norm-.5)>1e-15:
    raise ValueError("DAWE V1 requires Fixed10 Control05 actor clipping")
   forbidden=(self.recurrent.enabled,self.hierarchical_temporal_abstraction.enabled,self.wave_balance.enabled,
    self.wave_entry_curriculum.enabled,self.actor_kl_guard.enabled,self.inter_wave_credit.enabled,
    self.counterfactual_inter_wave_credit.enabled,self.boundary_redistributed_segment_credit.enabled,
    self.mission_film.enabled,self.wave_context.enabled,self.entity_attention_enabled,self.advantage_priority.enabled,
    self.ppo_stabilization.enabled,self.popart.enabled,self.curriculum.enabled,self.anchor.enabled,self.warm_start.enabled,
    self.reward_adapter.enabled,self.wave_survival_pbrs.enabled,self.team_mean_credit.enabled,
    self.persistent_wave_trajectory_replay.enabled,self.sequential_wave_gradient_projection.enabled,
    self.wave1_sensitivity_gating.enabled,self.wave_specific_actor_isolation.enabled,self.wave_specific_mean_heads.enabled)
   if any(forbidden):raise ValueError("DAWE V1 requires an otherwise Plain feed-forward Fixed10 Actor/Critic")
  if self.reference_variance.enabled:
   enabled=set(enabled_module_names(self.modules_config));required={"actor_lr_decay","actor_gradient_clipping","reference_variance"}
   if enabled!=required:raise ValueError(f"RV V1 requires exact enabled modules: {sorted(required)}")
   if self.deployment_aligned_wave_exploration.enabled:raise ValueError("RV and DAWE are mutually exclusive")
   if self.actor_kl_guard.enabled or self.ppo_stabilization.enabled:raise ValueError("RV V1 forbids actor KL guard and PPO stabilization")
   if not self.actor_gradient_clipping.enabled or abs(self.actor_gradient_clipping.actor_max_grad_norm-.5)>1e-15:
    raise ValueError("RV V1 requires Fixed10 actor clipping at 0.5")
  if self.wave_entry_curriculum.enabled:
   enabled=set(enabled_module_names(self.modules_config));allowed={"actor_lr_decay","wave_balancing","wave_entry_curriculum"}
   if not enabled.issubset(allowed):raise ValueError(f"wave_entry_curriculum incompatible enabled modules: {sorted(enabled-allowed)}")
  if self.inter_wave_credit.enabled:
   enabled=set(enabled_module_names(self.modules_config));allowed={"actor_lr_decay","inter_wave_credit"}
   if not enabled.issubset(allowed):raise ValueError(f"IWSC v1 incompatible enabled modules: {sorted(enabled-allowed)}")
   if self.recurrent.enabled:raise ValueError("IWSC v1 supports feed-forward Actor/Critic only")
  if self.counterfactual_inter_wave_credit.enabled:
   enabled=set(enabled_module_names(self.modules_config));allowed={"actor_lr_decay","counterfactual_inter_wave_credit"}
   if not enabled.issubset(allowed):raise ValueError(f"CAIW V2 incompatible enabled modules: {sorted(enabled-allowed)}")
   if self.inter_wave_credit.enabled:raise ValueError("IWSC V1 and CAIW V2 are mutually exclusive")
   if self.recurrent.enabled or self.wave_context.enabled or self.entity_attention_enabled:raise ValueError("CAIW V2 requires the feed-forward 52D Plain actor")
  if self.boundary_redistributed_segment_credit.enabled:
   enabled=set(enabled_module_names(self.modules_config));allowed={"actor_lr_decay","boundary_redistributed_segment_credit"}
   if not enabled.issubset(allowed):raise ValueError(f"BRSC V1 incompatible enabled modules: {sorted(enabled-allowed)}")
   if self.inter_wave_credit.enabled or self.counterfactual_inter_wave_credit.enabled:raise ValueError("BRSC, IWSC, and CAIW are mutually exclusive")
   if self.recurrent.enabled or self.wave_context.enabled or self.entity_attention_enabled:raise ValueError("BRSC V1 requires the feed-forward 52D Plain actor")
  if self.hierarchical_temporal_abstraction.enabled:
   enabled=set(enabled_module_names(self.modules_config));allowed={"actor_lr_decay","hierarchical_temporal_abstraction"}
   if self.hta_worker_consolidation.enabled:allowed.add("hta_worker_consolidation")
   if not enabled.issubset(allowed):raise ValueError(f"HTA V1 incompatible enabled modules: {sorted(enabled-allowed)}")
   if any((self.inter_wave_credit.enabled,self.counterfactual_inter_wave_credit.enabled,
           self.boundary_redistributed_segment_credit.enabled,self.recurrent.enabled,
           self.wave_context.enabled,self.entity_attention_enabled,self.mission_film.enabled)):
    raise ValueError("HTA V1 requires an otherwise Plain feed-forward Actor/Critic")
  if self.hta_worker_consolidation.enabled:
   if not self.hierarchical_temporal_abstraction.enabled:raise ValueError("hta_worker_consolidation requires hierarchical_temporal_abstraction")
   if not self.actor_lr_decay.enabled:raise ValueError("hta_worker_consolidation requires actor_lr_decay")
   enabled=set(enabled_module_names(self.modules_config))
   required={"actor_lr_decay","hierarchical_temporal_abstraction","hta_worker_consolidation"}
   if enabled!=required:raise ValueError(f"HTA Worker consolidation requires exact enabled modules: {sorted(required)}")
  if self.mission_film.enabled:
   if not (self.wave_context.enabled and self.wave_context.actor_enabled and not self.wave_context.critic_enabled
           and self.wave_context.target=="actor_only" and self.wave_context.encoding=="mission_markov"
           and self.wave_context.context_dim==5):
    raise ValueError("mission_film requires actor_only 5D mission_markov wave context and no critic context")
   allowed={"actor_lr_decay","wave_context","mission_film","actor_kl_guard"};enabled=set(enabled_module_names(self.modules_config))
   if not enabled.issubset(allowed):raise ValueError(f"mission_film incompatible enabled modules: {sorted(enabled-allowed)}")
  ac=self.wave_context.context_dim if self.wave_context.actor_enabled else 0;cc=(self.hierarchical_temporal_abstraction.num_options if self.hierarchical_temporal_abstraction.enabled else (self.wave_context.context_dim if self.wave_context.critic_enabled else 0))
  ar=self.recurrent.hidden_dim if self.recurrent.actor_enabled else 0;cr=self.recurrent.hidden_dim if self.recurrent.critic_enabled else 0
  critic_context_injection=("additive_zero" if self.hierarchical_temporal_abstraction.enabled or (self.wave_context.enabled and
   self.wave_context.encoding=="mission_markov" and self.wave_context.critic_enabled and
   not self.wave_context.actor_enabled) else "concat")
  if self.recurrent.state_memory:
   self.actor=StateMemoryMAPPOActor(observation_dim,action_dim,hidden_dim,log_std_min,log_std_max,actor_activation).to(self.device)
  else:
   self.actor=ModularMAPPOActor(observation_dim,action_dim,hidden_dim,log_std_min,log_std_max,actor_activation,ac,ar,self.entity_attention_config,self.mission_film.config,self.hierarchical_temporal_abstraction.config,self.recurrent.wave_segmented).to(self.device)
  self.critic=ModularCentralizedCritic(observation_dim,hidden_dim,attention_heads,critic_activation,cc,cr,critic_context_injection).to(self.device)
  trainable_actor_parameters=self.actor.trainable_policy_parameters()
  if not trainable_actor_parameters:raise RuntimeError("actor has no trainable policy parameters")
  self.actor_optimizer=torch.optim.Adam(trainable_actor_parameters,lr=actor_learning_rate);self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_learning_rate)
  if self.reference_variance.enabled:
   for parameter in self.actor.log_std.parameters():parameter.requires_grad_(False)
  self.wave2_actor=self.wave3_actor=None;self.wave2_actor_optimizer=self.wave3_actor_optimizer=None
  if self.wave_specific_actor_isolation.enabled:
   # deepcopy does not advance any RNG and produces exact source-policy clones.
   self.wave2_actor=deepcopy(self.actor).to(self.device);self.wave3_actor=deepcopy(self.actor).to(self.device)
   self.wave2_actor_optimizer=torch.optim.Adam(self.wave2_actor.trainable_policy_parameters(),lr=actor_learning_rate)
   self.wave3_actor_optimizer=torch.optim.Adam(self.wave3_actor.trainable_policy_parameters(),lr=actor_learning_rate)
  self.wave2_mean=self.wave3_mean=None;self.wave2_mean_optimizer=self.wave3_mean_optimizer=None
  if self.wave_specific_mean_heads.enabled:
   # deepcopy preserves the exact source mean without consuming any RNG.
   self.wave2_mean=deepcopy(self.actor.mean).to(self.device);self.wave3_mean=deepcopy(self.actor.mean).to(self.device)
   self.wave2_mean_optimizer=torch.optim.Adam(self.wave2_mean.parameters(),lr=actor_learning_rate)
   self.wave3_mean_optimizer=torch.optim.Adam(self.wave3_mean.parameters(),lr=actor_learning_rate)
  self.manager_actor=None;self.manager_critic=None;self.manager_actor_optimizer=None;self.manager_critic_optimizer=None
  self.hta_rng=np.random.default_rng(int(seed)^HTA_MANAGER_NUMPY_SEED_XOR)
  generator_device=self.device if self.device.type=="cuda" else torch.device("cpu")
  self.hta_manager_generator=torch.Generator(device=generator_device)
  self.hta_manager_generator.manual_seed(int(seed)^HTA_MANAGER_TORCH_SEED_XOR)
  self.manager_actor_update_count=0;self.manager_critic_update_count=0;self.manager_optimizer_step_count=0
  self.hta_option_usage_counts=np.zeros(4,dtype=np.int64)
  self.hta_decision_reason_counts={key:0 for key in ("rollout_start","periodic","wave_transition","episode_reset","rollout_truncation","episode_terminal")}
  if self.hierarchical_temporal_abstraction.enabled:
   with torch.random.fork_rng(devices=[]):
    self.manager_actor=HierarchicalManagerActor(observation_dim,hidden_dim,self.hierarchical_temporal_abstraction.num_options,actor_activation).to(self.device)
    self.manager_critic=ModularCentralizedCritic(observation_dim,hidden_dim,attention_heads,critic_activation,0,0,"concat").to(self.device)
   self.manager_actor_optimizer=torch.optim.Adam(self.manager_actor.parameters(),lr=actor_learning_rate)
   self.manager_critic_optimizer=torch.optim.Adam(self.manager_critic.parameters(),lr=critic_learning_rate)
  self.iw_critic=None;self.iw_critic_optimizer=None
  self.iw_rng=np.random.default_rng(int(seed)^0x49575343)
  self.iw_conflict_count=0;self.iw_gradient_step_count=0
  if self.inter_wave_credit.enabled:
   # The extra network must not perturb any RNG visible to the matched Plain run.
   with torch.random.fork_rng(devices=[]):
    self.iw_critic=InterWaveStateQualityCritic(observation_dim,hidden_dim,attention_heads,critic_activation,self.inter_wave_credit.max_waves).to(self.device)
   self.iw_critic_optimizer=torch.optim.Adam(self.iw_critic.parameters(),lr=self.inter_wave_credit.quality_critic_learning_rate)
  self.caiw_critic=None;self.caiw_critic_optimizer=None;self.caiw_rng=np.random.default_rng(int(seed)^0x43414957)
  self.caiw_conflict_count=0;self.caiw_gradient_step_count=0;self.caiw_trust_cap_count=0;self.caiw_aux_induced_clip_count=0
  if self.counterfactual_inter_wave_credit.enabled:
   with torch.random.fork_rng(devices=[]):
    self.caiw_critic=InterWaveActionOutcomeCritic(observation_dim,action_dim,hidden_dim,attention_heads,critic_activation,self.counterfactual_inter_wave_credit.max_waves).to(self.device)
   self.caiw_critic_optimizer=torch.optim.Adam(self.caiw_critic.parameters(),lr=self.counterfactual_inter_wave_credit.outcome_critic_learning_rate)
  self.brsc_critic=None;self.brsc_critic_optimizer=None;self.brsc_rng=np.random.default_rng(int(seed)^0x42525343)
  self.brsc_conflict_count=0;self.brsc_gradient_step_count=0;self.brsc_trust_cap_count=0;self.brsc_aux_induced_clip_count=0
  if self.boundary_redistributed_segment_credit.enabled:
   with torch.random.fork_rng(devices=[]):
    self.brsc_critic=BoundaryStateOutcomeCritic(observation_dim,hidden_dim,attention_heads,critic_activation,self.boundary_redistributed_segment_credit.max_waves).to(self.device)
   self.brsc_critic_optimizer=torch.optim.Adam(self.brsc_critic.parameters(),lr=self.boundary_redistributed_segment_credit.outcome_critic_learning_rate)
  self.base_actor_learning_rate=float(actor_learning_rate);self.base_critic_learning_rate=float(critic_learning_rate)
  self.ppo_update_count=self.actor_update_count=self.critic_update_count=self.sampled_steps=self.vector_steps=0
  self.kl_hard_stop_count=0
  self.actor_kl_guard_hard_stop_count=0
  self.actor_kl_guard_actor_epochs_total=0
  self.actor_kl_guard_actor_epochs_min=None
  self.warm_start_provenance={};self.anchor_provenance={}
  self.fbmr_branch_metadata={};self._frozen_actor_reference=None
  self.rng_restore_metadata={"rng_state_available":False,"rng_state_restored":False,"cuda_rng_state_restored":False}

 def context_numpy(self,wave,total,**state):return self.wave_context.encode_numpy(wave,total,**state) if self.wave_context.enabled else np.zeros((*np.asarray(wave).shape,0),np.float32)
 def initial_hidden(self,num_envs):
  actor=self.recurrent.zeros(num_envs,self.num_agents,True)
  if self.recurrent.wave_segmented:
   actor=self.prepare_actor_hidden(actor,np.ones((num_envs,self.num_agents),np.float32),np.ones(num_envs,np.int64),np.ones(num_envs,bool))
  return actor,self.recurrent.zeros(num_envs,self.num_agents,False)
 @torch.no_grad()
 def prepare_actor_hidden(self,hidden,alive,waves,phase_reset_flags):
  if not self.recurrent.wave_boundary_reset:return hidden
  if phase_reset_flags is None or waves is None:raise RuntimeError("wave-segmented Actor requires explicit pre-action phase reset flags and waves")
  if self.recurrent.state_memory:
   self.actor.wave_one_hot(waves,alive,self.device) # validate the pre-action wave provenance
   live=np.zeros((*np.asarray(alive).shape,self.recurrent.hidden_dim),np.float32)
  else:live=self.actor.phase_hidden(waves,alive).cpu().numpy()
  flags=np.asarray(phase_reset_flags,dtype=bool)
  if flags.shape!=live.shape[:-2]:raise RuntimeError("actor phase reset flags shape mismatch")
  if hidden is None:hidden=np.zeros_like(live)
  return np.where(flags[...,None,None],live,hidden)*np.asarray(alive)[...,None]
 @torch.no_grad()
 def actor_context_numpy(self,waves,alive,context):
  if not self.recurrent.state_memory:return context
  return self.actor.wave_one_hot(waves,alive,self.device).cpu().numpy()
 def _ctx(self,c,actor):
  if self.recurrent.state_memory:return c if actor else None
  active=self.wave_context.actor_enabled if actor else self.wave_context.critic_enabled
  return c if active else None
 def _actor_grad_clip_limit(self):return self.actor_gradient_clipping.actor_max_grad_norm if self.actor_gradient_clipping.enabled else self.max_grad_norm
 def _wsai_actors(self):return [self.actor,self.wave2_actor,self.wave3_actor]
 def _wsai_actor_optimizers(self):return [self.actor_optimizer,self.wave2_actor_optimizer,self.wave3_actor_optimizer]
 def _wsmh_means(self):return [self.actor.mean,self.wave2_mean,self.wave3_mean]
 def _wsmh_mean_optimizers(self):return [self.actor_optimizer,self.wave2_mean_optimizer,self.wave3_mean_optimizer]
 def _clone_source_mean_optimizer_state(self,optimizer):
  """Clone actor.mean Adam state by live Parameter identity, never integer IDs."""
  source=list(self.actor.mean.parameters());destination=list(optimizer.param_groups[0]["params"])
  if len(source)!=len(destination):raise RuntimeError("WSMH mean optimizer parameter mismatch")
  source_group=self.actor_optimizer.param_groups[0];target_group=optimizer.param_groups[0]
  for key,value in source_group.items():
   if key!="params":target_group[key]=deepcopy(value)
  optimizer.state.clear()
  for source_parameter,destination_parameter in zip(source,destination):
   if source_parameter not in self.actor_optimizer.state:raise RuntimeError("source actor mean Adam state missing")
   optimizer.state[destination_parameter]=deepcopy(self.actor_optimizer.state[source_parameter])
 def _wsmh_routed_distribution(self,obs,mask,wave_indices):
  if not self.wave_specific_mean_heads.enabled:return self.actor.distribution_step(obs,None,None,None,mask)[0]
  if wave_indices is None:raise ValueError("WSMH action routing requires explicit environment wave indices")
  waves=torch.as_tensor(wave_indices,dtype=torch.long,device=self.device).reshape(-1)
  if obs.shape[0]!=waves.shape[0] or bool(((waves<1)|(waves>3)).any()):raise ValueError("WSMH wave indices must align with batch and lie in 1..3")
  self.wave_specific_mean_heads.record_routing(waves.detach().cpu().tolist())
  encoded=self.actor.backbone(obs);means=torch.stack([head(encoded) for head in self._wsmh_means()],0)
  batch=torch.arange(obs.shape[0],device=self.device);loc=means[waves-1,batch]
  scale=self.actor.log_std(encoded).clamp(self.actor.log_std_min,self.actor.log_std_max).exp()
  return torch.distributions.Normal(loc,scale)
 def _routed_actor_distribution(self,obs,mask,wave_indices):
  if not self.wave_specific_actor_isolation.enabled:return self.actor.distribution_step(obs,None,None,None,mask)[0]
  if wave_indices is None:raise ValueError("WSAI action routing requires explicit environment wave indices")
  waves=torch.as_tensor(wave_indices,dtype=torch.long,device=self.device).reshape(-1)
  if obs.shape[0]!=waves.shape[0] or bool(((waves<1)|(waves>3)).any()):raise ValueError("WSAI wave indices must align with batch and lie in 1..3")
  self.wave_specific_actor_isolation.record_routing(waves.detach().cpu().tolist())
  distributions=[actor.distribution_step(obs,None,None,None,mask)[0] for actor in self._wsai_actors()]
  loc=torch.stack([dist.loc for dist in distributions],0);scale=torch.stack([dist.scale for dist in distributions],0)
  batch=torch.arange(obs.shape[0],device=self.device);return torch.distributions.Normal(loc[waves-1,batch],scale[waves-1,batch])
 def _effective_actor_distribution(self,base_distribution,wave_indices):
  """Single DAWE distribution transform shared by rollout, PPO, and entropy."""
  return self.deployment_aligned_wave_exploration.effective_distribution(base_distribution,wave_indices)
 def _behavior_actor_distribution(self,base_distribution,observations,alive_mask,wave_indices=None):
  """The only stochastic behavior-distribution transform used by rollout/PPO."""
  if self.reference_variance.enabled:
   if self.reference_variance_actor is None:raise RuntimeError("RV reference actor is not attached")
   with torch.no_grad():
    reference,_=self.reference_variance_actor.distribution_step(observations,None,None,None,alive_mask)
   return torch.distributions.Normal(base_distribution.loc,reference.scale.detach())
  return self._effective_actor_distribution(base_distribution,wave_indices)
 @torch.no_grad()
 def act(self,observations,alive_mask=None,deterministic=False,return_policy_data=False,context=None,hidden=None,episode_mask=None,option_ids=None,wave_indices=None,actor_phase_reset_flags=None):
  if self.recurrent.wave_boundary_reset:hidden=self.prepare_actor_hidden(hidden,alive_mask,wave_indices,actor_phase_reset_flags)
  if self.recurrent.state_memory:context=self.actor_context_numpy(wave_indices,alive_mask,context)
  obs=torch.as_tensor(observations,dtype=torch.float32,device=self.device);mask=torch.as_tensor(alive_mask,dtype=torch.float32,device=self.device) if alive_mask is not None else None
  ctx=torch.as_tensor(context,dtype=torch.float32,device=self.device) if context is not None else None;hid=torch.as_tensor(hidden,dtype=torch.float32,device=self.device) if hidden is not None else None
  ep=torch.as_tensor(episode_mask,dtype=torch.float32,device=self.device) if episode_mask is not None else None
  options=None if option_ids is None else torch.as_tensor(option_ids,dtype=torch.long,device=self.device)
  if self.wave_specific_mean_heads.enabled:dist=self._wsmh_routed_distribution(obs,mask,wave_indices);new_h=None
  elif self.wave_specific_actor_isolation.enabled:dist=self._routed_actor_distribution(obs,mask,wave_indices);new_h=None
  else:dist,new_h=self.actor.distribution_step(obs,self._ctx(ctx,True),hid,ep,mask,option_ids=options)
  effective=dist if deterministic else self._behavior_actor_distribution(dist,obs,mask,wave_indices)
  raw=dist.mean if deterministic else effective.rsample();actions=torch.tanh(raw);log=self.actor._squashed_log_prob(effective,raw,actions)
  if mask is not None:actions*=mask[...,None];raw*=mask[...,None];log*=mask
  out=(actions.cpu().numpy(),raw.cpu().numpy(),log.cpu().numpy(),None if new_h is None else new_h.cpu().numpy())
  return out if return_policy_data else (out[0],out[3])
 @torch.no_grad()
 def _tactical_option_context(self,option_ids):
  if not self.hierarchical_temporal_abstraction.enabled:return None
  if option_ids is None:raise ValueError("HTA tactical critic requires option_ids")
  value=torch.as_tensor(option_ids,dtype=torch.long,device=self.device)
  return torch.nn.functional.one_hot(value,self.hierarchical_temporal_abstraction.num_options).to(torch.float32)
 @torch.no_grad()
 def manager_act(self,observations,alive_mask,deterministic=False,return_policy_data=False):
  if not self.hierarchical_temporal_abstraction.enabled:raise RuntimeError("HTA manager is disabled")
  obs=torch.as_tensor(observations,dtype=torch.float32,device=self.device);mask=torch.as_tensor(alive_mask,dtype=torch.float32,device=self.device)
  logits=self.manager_actor.logits(obs);probs=torch.softmax(logits,-1)
  if deterministic:options=probs.argmax(-1)
  else:options=torch.multinomial(probs.reshape(-1,probs.shape[-1]),1,generator=self.hta_manager_generator).reshape(probs.shape[:-1])
  log_probs=torch.log_softmax(logits,-1).gather(-1,options[...,None]).squeeze(-1)
  options=torch.where(mask>.5,options,torch.zeros_like(options));log_probs=log_probs*mask
  out=(options.cpu().numpy(),log_probs.cpu().numpy(),logits.cpu().numpy(),probs.cpu().numpy())
  return out if return_policy_data else out[0]
 @torch.no_grad()
 def manager_values_step(self,obs,alive):
  if not self.hierarchical_temporal_abstraction.enabled:raise RuntimeError("HTA manager is disabled")
  value,_=self.manager_critic.forward_step(torch.as_tensor(obs,dtype=torch.float32,device=self.device),torch.as_tensor(alive,dtype=torch.float32,device=self.device))
  return value.cpu().numpy()
 @torch.no_grad()
 def values_step(self,obs,alive,context=None,hidden=None,episode_mask=None,raw=True,option_ids=None):
  conv=lambda x:None if x is None else torch.as_tensor(x,dtype=torch.float32,device=self.device)
  critic_context=self._tactical_option_context(option_ids) if self.hierarchical_temporal_abstraction.enabled else self._ctx(conv(context),False)
  value,new_h=self.critic.forward_step(conv(obs),conv(alive),critic_context,conv(hidden),conv(episode_mask))
  if raw and self.popart.enabled:value=self.popart.denormalize_values(value)
  return value.cpu().numpy(),None if new_h is None else new_h.cpu().numpy()
 def _value_rollout(self,r,obs,next_obs):
  T,E=obs.shape[:2];values=[];next_values=[]
  for t in range(T):
   h=None if r.critic_hidden_before_step is None else r.critic_hidden_before_step[t]
   current_options=None if r.hta_options is None else r.hta_options[t]
   next_options=None if r.hta_next_options is None else r.hta_next_options[t]
   v,nh=self.values_step(obs[t],r.alive_masks[t],r.contexts[t],h,r.episode_masks[t] if r.episode_masks is not None else None,option_ids=current_options)
   nv,_=self.values_step(next_obs[t],r.next_alive_masks[t],r.next_contexts[t],nh,1-r.dones[t],option_ids=next_options)
   values.append(v);next_values.append(nv)
  return torch.as_tensor(np.asarray(values),device=self.device),torch.as_tensor(np.asarray(next_values),device=self.device)
 def update(self,r:ModularRolloutBatch):
  if self.recurrent.state_memory:
   expected=self.actor_context_numpy(r.wave_indices,r.alive_masks,None)
   if not np.array_equal(r.contexts,expected):raise RuntimeError("State Memory rollout one-hot context/pre-action wave mismatch")
  tt=lambda x:torch.as_tensor(x,dtype=torch.float32,device=self.device)
  obs,act,raw,oldlog,rewards,dones,alive,nobs,nalive=map(tt,(r.observations,r.actions,r.raw_actions,r.old_log_probs,r.rewards,r.dones,r.alive_masks,r.next_observations,r.next_alive_masks))
  waves=torch.as_tensor(r.wave_indices,dtype=torch.long,device=self.device);ctx=tt(r.contexts);T,E=obs.shape[:2]
  dawe_metrics=self._dawe_diagnostics(r,obs,raw,alive,ctx,waves)
  rv_metrics=self._reference_variance_diagnostics(obs,alive,waves)
  if self.team_mean_credit.enabled:
   credit_rewards,team_credit_metrics=self.team_mean_credit.transform(rewards,alive,waves)
  else:
   credit_rewards=rewards;team_credit_metrics=self.team_mean_credit.disabled_metrics()
  marc_metrics={};marc_weights=None
  with torch.no_grad():
   values,next_values=self._value_rollout(r,obs,nobs)
   global_adv,returns=compute_gae(credit_rewards,values,next_values,dones,alive,nalive,self.gamma,self.gae_lambda)
   adv=global_adv;raw_adv=global_adv.clone()
   if self.milestone_aware_retention_credit.enabled:
    if r.wave_transition_flags is None:raise RuntimeError("MARC rollout lacks wave_transition_flags")
    transitions=torch.as_tensor(r.wave_transition_flags,dtype=torch.float32,device=self.device)
    local_adv,local_returns=compute_local_gae(credit_rewards,values,next_values,dones,alive,nalive,
     transitions,self.gamma,self.gae_lambda)
    continuation_adv=global_adv-local_adv
    eta=self.milestone_aware_retention_credit.coefficients(waves).unsqueeze(-1)
    adv=local_adv+eta*continuation_adv
    # The local return is diagnostic only.  Critic targets remain full-horizon.
    del local_returns
    marc_weights,wave_metrics=self.milestone_aware_retention_credit.wave_weights(waves,alive)
    def marc_stats(name,tensor):
     live=tensor[alive>.5]
     return {f"marc_{name}_mean":float(live.mean()),f"marc_{name}_std":float(live.std(unbiased=False))}
    marc_metrics.update(marc_stats("global_adv",global_adv))
    marc_metrics.update(marc_stats("local_adv",local_adv))
    marc_metrics.update(marc_stats("cont_adv",continuation_adv))
    marc_metrics.update(marc_stats("actor_adv",adv))
    marc_metrics.update(wave_metrics)
   wave_w,wmetrics=self.wave_balance.compute_tensor(waves,alive)
   _,actor_w,pmetrics=self.advantage_priority.compute_tensor(raw_adv,waves,alive,wave_w if self.wave_balance.actor_enabled else torch.ones_like(wave_w))
   if self.normalize_advantages:
    live=adv[alive>.5];adv=((adv-live.mean())/live.std(unbiased=False).clamp_min(1e-8))*alive
   if self.popart.enabled:
    self.popart.update(returns[alive>.5],self.critic.output_layer); old_values=self.popart.normalize_targets(values);target_returns=self.popart.normalize_targets(returns)
   else:old_values=values;target_returns=returns
  iw_metrics=self._iw_default_metrics()
  iw_adv=None;iw_active=None
  if self.inter_wave_credit.enabled:
   self.inter_wave_credit.ingest(r.iw_supervision_segments)
   iw_metrics.update(self._train_iw_critic())
   iw_adv,iw_active,credit_metrics=self._iw_rollout_credit(r,obs,nobs,alive,nalive,dones,waves)
   iw_metrics.update(credit_metrics)
  caiw_metrics=self._caiw_default_metrics();caiw_adv=None;caiw_active=None
  if self.counterfactual_inter_wave_credit.enabled:
   self.counterfactual_inter_wave_credit.ingest(r.caiw_supervision_segments)
   caiw_metrics.update(self._train_caiw_critic())
   caiw_metrics.update(self._validate_caiw_tasks())
   caiw_adv,caiw_active,credit_metrics=self._caiw_rollout_credit(r,obs,act,raw,oldlog,alive,waves)
   caiw_metrics.update(credit_metrics)
  # BRSC deliberately scores the current rollout with the critic/readiness/prior
  # frozen at update entry. Current-rollout labels are ingested only after PPO.
  brsc_metrics=self._brsc_default_metrics();brsc_adv=None;brsc_active=None
  if self.boundary_redistributed_segment_credit.enabled:
   brsc_adv,brsc_active,credit_metrics=self._brsc_rollout_credit(r,nobs,alive,nalive,dones,waves)
   brsc_metrics.update(credit_metrics)
  hta_prepared=None
  if self.hierarchical_temporal_abstraction.enabled:
   if r.hta_options is None or r.hta_next_options is None or r.hta_manager_transitions is None:
    raise RuntimeError("HTA rollout lacks option or manager-transition data")
   hta_prepared=self._prepare_hta_manager_batch(r.hta_manager_transitions)
  # All actor priorities are frozen from the complete raw-GAE rollout.
  consolidation_lr={"effective_lr":float(self.actor_optimizer.param_groups[0]["lr"]),"multiplier":1.0}
  if self.actor_lr_decay.enabled:
   worker_baseline_lr=self.actor_lr_decay.apply(self.actor_optimizer,self.sampled_steps,self.base_actor_learning_rate)
   if self.wave_specific_mean_heads.enabled:
    self.actor_lr_decay.apply(self.wave2_mean_optimizer,self.sampled_steps,self.base_actor_learning_rate)
    self.actor_lr_decay.apply(self.wave3_mean_optimizer,self.sampled_steps,self.base_actor_learning_rate)
   if self.wave_specific_actor_isolation.enabled:
    self.actor_lr_decay.apply(self.wave2_actor_optimizer,self.sampled_steps,self.base_actor_learning_rate)
    self.actor_lr_decay.apply(self.wave3_actor_optimizer,self.sampled_steps,self.base_actor_learning_rate)
   if self.hierarchical_temporal_abstraction.enabled:
    self.actor_lr_decay.apply(self.manager_actor_optimizer,self.sampled_steps,self.base_actor_learning_rate)
   if self.hta_worker_consolidation.enabled:
    consolidation_lr=self.hta_worker_consolidation.apply(self.actor_optimizer,self.sampled_steps,worker_baseline_lr)
  elif self.ppo_stabilization.enabled:
   lr=self.ppo_stabilization.actor_learning_rate(self.sampled_steps,self.total_sampled_steps)
   for group in self.actor_optimizer.param_groups:group["lr"]=lr
  actor_before,critic_before=self.actor_update_count,self.critic_update_count
  worker_parameters=self.actor.trainable_policy_parameters() if self.hta_worker_consolidation.enabled else []
  worker_before=[parameter.detach().clone() for parameter in worker_parameters]
  if self.milestone_aware_retention_credit.enabled:
   self.milestone_aware_retention_credit.ingest_success_segments(
    r.marc_success_segments,self.actor,self.device)
   if self.recurrent.wave_boundary_reset:
    metrics=self._update_actor_recurrent_critic_flat(r,obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx,marc_weights)
   else:
    metrics=self._update_flat_marc(obs,act,raw,oldlog,alive,adv,old_values,target_returns,
     wave_w,ctx,waves,marc_weights)
  elif self.wave_specific_mean_heads.enabled:
   metrics=self._update_flat_wsmh(obs,act,raw,oldlog,alive,adv,old_values,target_returns,ctx,waves)
  elif self.wave_specific_actor_isolation.enabled:
   metrics=self._update_flat_wsai(obs,act,raw,oldlog,alive,adv,old_values,target_returns,ctx,waves)
  elif self.hierarchical_temporal_abstraction.enabled:
   options=torch.as_tensor(r.hta_options,dtype=torch.long,device=self.device)
   metrics=self._update_flat_hta(obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx,options)
  elif self.recurrent.actor_enabled and not self.recurrent.critic_enabled:
   metrics=self._update_actor_recurrent_critic_flat(r,obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx)
  elif self.recurrent.actor_enabled or self.recurrent.critic_enabled:
   metrics=self._update_recurrent(r,obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx)
  elif self.ppo_stabilization.enabled:
   metrics=self._update_flat_stabilized(obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx,actor_w)
  elif self.actor_kl_guard.enabled:
   metrics=self._update_flat_actor_kl_guard(obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx,actor_w if self.advantage_priority.enabled else None)
  elif self.sequential_wave_gradient_projection.enabled:
   swgp_active,swgp_first_activation=self.sequential_wave_gradient_projection.record_update_start(self.sampled_steps)
   if swgp_active:
    metrics=self._update_flat_swgp(obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx,waves)
   else:
    # Delayed-SWGP must be literally Plain before activation: do not enter the
    # SWGP routine, compute per-wave gradients, or consume any extra RNG.
    metrics=self._update_flat(obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx)
  elif self.inter_wave_credit.enabled and iw_active is not None and bool(iw_active.any()):
   self._iw_source_waves=waves.reshape(-1)
   metrics=self._update_flat_iwsc(obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx,iw_adv,iw_active)
  elif self.counterfactual_inter_wave_credit.enabled and caiw_active is not None and bool(caiw_active.any()):
   self._caiw_source_waves=waves.reshape(-1)
   metrics=self._update_flat_caiw(obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx,caiw_adv,caiw_active)
  elif self.boundary_redistributed_segment_credit.enabled and brsc_active is not None and bool(brsc_active.any()):
   self._brsc_source_waves=waves.reshape(-1)
   metrics=self._update_flat_brsc(obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx,brsc_adv,brsc_active)
  else:
   metrics=self._update_flat(obs,act,raw,oldlog,alive,adv,old_values,target_returns,wave_w,ctx,
    actor_w if self.advantage_priority.enabled else None,waves if self.persistent_wave_trajectory_replay.enabled else None,
    waves if (self.deployment_aligned_wave_exploration.enabled or self.reference_variance.enabled) else None)
  if self.hta_worker_consolidation.enabled:
   with torch.no_grad():
    norm_before=torch.sqrt(sum(value.double().square().sum() for value in worker_before))
    norm_after=torch.sqrt(sum(parameter.detach().double().square().sum() for parameter in worker_parameters))
    delta=torch.sqrt(sum((parameter.detach().double()-before.double()).square().sum() for parameter,before in zip(worker_parameters,worker_before)))
   metrics.update({"hta_worker_consolidation_multiplier":float(consolidation_lr["multiplier"]),
    "hta_worker_actor_lr":float(self.actor_optimizer.param_groups[0]["lr"]),
    "hta_manager_actor_lr":float(self.manager_actor_optimizer.param_groups[0]["lr"]),
    "hta_manager_worker_lr_ratio":float(self.manager_actor_optimizer.param_groups[0]["lr"]/(self.actor_optimizer.param_groups[0]["lr"]+1e-30)),
    "hta_worker_parameter_norm_before":float(norm_before),"hta_worker_parameter_norm_after":float(norm_after),
    "hta_worker_parameter_delta_l2":float(delta),
    "hta_worker_relative_parameter_update":float(delta/(norm_before+1e-12))})
  if self.boundary_redistributed_segment_credit.enabled:
   self.boundary_redistributed_segment_credit.ingest(r.brsc_supervision_boundaries)
   brsc_metrics.update(self._train_brsc_critic())
   brsc_metrics.update(self._validate_brsc_tasks())
  if self.hierarchical_temporal_abstraction.enabled:
   metrics.update(self._update_hta_manager(hta_prepared))
   metrics.update(self._hta_worker_diagnostics(obs,torch.as_tensor(r.hta_options,dtype=torch.long,device=self.device)))
  fresh_actor_steps=self.actor_update_count-actor_before;fresh_critic_steps=self.critic_update_count-critic_before
  w1sg_metrics=self.wave1_sensitivity_gating.compute_sensitivity(self.actor,obs,raw,oldlog,alive,waves,adv,self.clip_ratio)
  pwtr_metrics=self.persistent_wave_trajectory_replay.default_metrics(r.wave_indices)
  if self.persistent_wave_trajectory_replay.enabled and self.persistent_wave_trajectory_replay.replay_enabled:
   pwtr_metrics.update(self._pwtr_replay_phase())
  metrics.update(pwtr_metrics)
  metrics.update(self.wave1_sensitivity_gating.gradient_metrics() if self.wave1_sensitivity_gating.enabled else w1sg_metrics)
  metrics["pwtr_fresh_actor_optimizer_steps"]=float(fresh_actor_steps)
  metrics["pwtr_fresh_critic_optimizer_steps"]=float(fresh_critic_steps)
  live_mask=alive>.5;rv=returns[live_mask];vv=values[live_mask];variance=torch.var(rv,unbiased=False)
  metrics["explained_variance"]=float((1-torch.var(rv-vv,unbiased=False)/variance.clamp_min(1e-8)).detach())
  metrics["actor_optimizer_steps_this_update"]=float(self.actor_update_count-actor_before);metrics["critic_optimizer_steps_this_update"]=float(self.critic_update_count-critic_before)
  recurrent_steps=(self.actor_update_count-actor_before) if (self.recurrent.actor_enabled or self.recurrent.critic_enabled) else 0
  metrics["recurrent_optimizer_steps_this_update"]=float(recurrent_steps)
  metrics.update(self._policy_diagnostics(r,obs,act,alive,ctx))
  metrics.update(dawe_metrics)
  metrics.update(rv_metrics)
  self.ppo_update_count+=1;metrics.update(wmetrics);metrics.update(pmetrics)
  for key,value in iw_metrics.items():metrics.setdefault(key,value)
  for key,value in caiw_metrics.items():metrics.setdefault(key,value)
  for key,value in brsc_metrics.items():metrics.setdefault(key,value)
  metrics.update(team_credit_metrics)
  metrics.update(marc_metrics)
  if self.sequential_wave_gradient_projection.enabled:
   module=self.sequential_wave_gradient_projection
   for key in ("swgp_active","swgp_wave1_alive_fraction","swgp_wave2_alive_fraction",
               "swgp_wave3_alive_fraction","swgp_wave1_grad_norm","swgp_wave2_grad_norm",
               "swgp_wave3_grad_norm","swgp_raw_cos_w1_w2","swgp_raw_cos_w1_w3",
               "swgp_raw_cos_w2_w3","swgp_w2_projection_applied_fraction",
               "swgp_w2_dot_before","swgp_w2_dot_after","swgp_w2_cos_before",
               "swgp_w2_cos_after","swgp_w3_w1_projection_applied_fraction",
               "swgp_w3_w1_dot_before","swgp_w3_w1_dot_after",
               "swgp_w3_w2_projection_applied_fraction","swgp_w3_w2_dot_before",
               "swgp_w3_w2_dot_after","swgp_surrogate_grad_norm_plain",
               "swgp_surrogate_grad_norm_projected","swgp_entropy_grad_norm",
               "swgp_actor_grad_norm_pre_clip","swgp_actor_grad_norm_post_clip",
               "swgp_no_projection_fraction","swgp_single_wave_minibatch_fraction"):
    metrics.setdefault(key,0.0)
   metrics.update({"swgp_total_minibatches":float(module.total_swgp_minibatches),
    "swgp_w2_projection_count":float(module.w2_projection_count),
    "swgp_w3_w1_projection_count":float(module.w3_w1_projection_count),
    "swgp_w3_w2_projection_count":float(module.w3_w2_projection_count),
    "swgp_no_projection_count":float(module.no_projection_count),
    "swgp_single_wave_minibatch_count":float(module.single_wave_minibatch_count)})
   metrics.update(module.activation_diagnostics(swgp_active,swgp_first_activation))
  metrics.update({"popart_mean":float(self.popart.mean),"popart_std":float(self.popart.std),"popart_count":float(self.popart.count),"actor_learning_rate":float(self.actor_optimizer.param_groups[0]["lr"]),"critic_learning_rate":float(self.critic_optimizer.param_groups[0]["lr"]),"kl_hard_stop_count":float(self.kl_hard_stop_count),"cumulative_kl_hard_stop_count":float(self.kl_hard_stop_count),"actor_kl_guard_hard_stop_count":float(self.actor_kl_guard_hard_stop_count),"actor_kl_guard_hard_stop_fraction":float(self.actor_kl_guard_hard_stop_count/self.ppo_update_count)})
  numeric_metrics=[value for value in metrics.values() if isinstance(value,(int,float,np.integer,np.floating))]
  if not np.all(np.isfinite(numeric_metrics)):raise FloatingPointError(f"non-finite modular update: {metrics}")
  return metrics

 def _prepare_hta_manager_batch(self,batch):
  tt=lambda x,d=torch.float32:torch.as_tensor(x,dtype=d,device=self.device)
  obs=tt(batch.observations);alive=tt(batch.alive_masks);nobs=tt(batch.next_observations);nalive=tt(batch.next_alive_masks)
  with torch.no_grad():
   values,_=self.manager_critic.forward_step(obs,alive)
   next_values,_=self.manager_critic.forward_step(nobs,nalive)
   advantages,returns,deltas=compute_smdp_gae(tt(batch.rewards),values,next_values,tt(batch.durations),
    tt(batch.bootstrap_masks),tt(batch.trace_masks),self.gamma,self.gae_lambda,batch.env_ids)
   if self.normalize_advantages:
    live=advantages[alive>.5];advantages=((advantages-live.mean())/live.std(unbiased=False).clamp_min(1e-8))*alive
  return {"obs":obs,"alive":alive,"options":tt(batch.options,torch.long),"oldlog":tt(batch.old_log_probs),
          "oldvalue":values,"returns":returns,"advantages":advantages,"durations":tt(batch.durations),
          "env_ids":np.asarray(batch.env_ids),"end_reasons":np.asarray(batch.end_reasons),
          "decision_reasons":np.asarray(batch.decision_reasons) if batch.decision_reasons is not None else None,"deltas":deltas}

 def _update_hta_manager(self,data):
  O,M,Z,L,OV,TG,ADV=(data[k] for k in ("obs","alive","options","oldlog","oldvalue","returns","advantages"))
  states=O.shape[0];states_per_minibatch=max(1,self.minibatch_size//self.num_agents)
  actor_losses=[];value_losses=[];entropies=[];kls=[];clips=[]
  for _ in range(self.ppo_epochs):
   permutation=self.hta_rng.permutation(states)
   for start in range(0,states,states_per_minibatch):
    ix=torch.as_tensor(permutation[start:start+states_per_minibatch],dtype=torch.long,device=self.device)
    mask=M[ix];dist=self.manager_actor.distribution(O[ix]);newlog=dist.log_prob(Z[ix]);logratio,ratio=stable_ratio_terms(newlog,L[ix])
    surrogate=torch.minimum(ratio*ADV[ix],ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*ADV[ix])
    actor_loss=-masked_mean(surrogate,mask);entropy=masked_mean(dist.entropy(),mask)
    self.manager_actor_optimizer.zero_grad();(actor_loss-self.entropy_coefficient*entropy).backward()
    nn.utils.clip_grad_norm_(self.manager_actor.parameters(),self.max_grad_norm);self.manager_actor_optimizer.step();self.manager_actor_update_count+=1
    value,_=self.manager_critic.forward_step(O[ix],mask);clipped=OV[ix]+(value-OV[ix]).clamp(-self.clip_ratio,self.clip_ratio)
    error=torch.maximum((value-TG[ix]).square(),(clipped-TG[ix]).square()) if self.clip_value_loss else (value-TG[ix]).square()
    value_loss=.5*masked_mean(error,mask)
    self.manager_critic_optimizer.zero_grad();(self.value_loss_coefficient*value_loss).backward()
    nn.utils.clip_grad_norm_(self.manager_critic.parameters(),self.max_grad_norm);self.manager_critic_optimizer.step();self.manager_critic_update_count+=1
    live=mask>.5;kl=((ratio-1)-logratio)[live]
    actor_losses.append(float(actor_loss.detach()));value_losses.append(float(value_loss.detach()));entropies.append(float(entropy.detach()))
    kls.append(float(kl.mean().detach()));clips.append(float((torch.abs(ratio[live]-1)>self.clip_ratio).float().mean().detach()))
  self.manager_optimizer_step_count+=1
  alive_np=M.detach().cpu().numpy()>0.5;options_np=Z.detach().cpu().numpy();counts=np.asarray([(options_np[alive_np]==i).sum() for i in range(4)],dtype=np.int64)
  self.hta_option_usage_counts+=counts;total=max(1,int(counts.sum()));fractions=counts/total
  reasons=np.asarray(data["end_reasons"],dtype=str)
  for key in self.hta_decision_reason_counts:self.hta_decision_reason_counts[key]+=int((reasons==key).sum())
  if data["decision_reasons"] is not None:
   starts=np.asarray(data["decision_reasons"],dtype=str)
   for key in self.hta_decision_reason_counts:self.hta_decision_reason_counts[key]+=int((starts==key).sum())
  switches=[];last={}
  for env,options,mask in zip(data["env_ids"],options_np,alive_np):
   env=int(env)
   if env in last:
    valid=mask & last[env][1]
    if valid.any():switches.extend((options[valid]!=last[env][0][valid]).tolist())
   last[env]=(options,mask)
  rv=TG[M>.5];vv=OV[M>.5];variance=torch.var(rv,unbiased=False)
  durations=data["durations"].detach().cpu().numpy()
  result={"hta_manager_actor_loss":float(np.mean(actor_losses)),"hta_manager_value_loss":float(np.mean(value_losses)),
   "hta_manager_entropy":float(np.mean(entropies)),"hta_manager_approx_kl":float(np.mean(kls)),
   "hta_manager_clip_fraction":float(np.mean(clips)),"hta_manager_explained_variance":float(1-torch.var(rv-vv,unbiased=False)/variance.clamp_min(1e-8)),
   "hta_manager_actor_lr":float(self.manager_actor_optimizer.param_groups[0]["lr"]),"hta_macro_count":float(states),
   "hta_macro_duration_mean":float(np.mean(durations)),"hta_macro_duration_p50":float(np.quantile(durations,.5)),"hta_macro_duration_p90":float(np.quantile(durations,.9)),
   "hta_option_switch_fraction":float(np.mean(switches)) if switches else 0.,"hta_manager_max_option_fraction":float(fractions.max())}
  for i in range(4):result[f"hta_option_{i}_fraction"]=float(fractions[i])
  reason_names={"periodic":"hta_periodic_end_fraction","wave_transition":"hta_wave_end_fraction","rollout_truncation":"hta_rollout_truncation_fraction","episode_terminal":"hta_episode_terminal_fraction"}
  for reason,key in reason_names.items():result[key]=float(np.mean(reasons==reason))
  return result

 @torch.no_grad()
 def _hta_worker_diagnostics(self,obs,options):
  flat=obs.reshape(-1,obs.shape[-1]);h=self.actor.backbone(flat)
  residuals=torch.stack([head(h) for head in self.actor.option_mean_residuals],1)
  selected=torch.gather(residuals,1,options.reshape(-1)[:,None,None].expand(-1,1,residuals.shape[-1])).squeeze(1)
  return {"hta_option_residual_abs_mean":float(selected.abs().mean()),"hta_option_residual_abs_max":float(selected.abs().max()),
          **{f"hta_option_{i}_residual_norm":float(torch.linalg.vector_norm(residuals[:,i],dim=-1).mean()) for i in range(4)}}

 def _iw_default_metrics(self):
  keys=("iw_actor_active","iw_active_wave1","iw_active_wave2","iw_q_current_mean","iw_q_next_mean",
        "iw_delta_mean_wave1","iw_delta_std_wave1","iw_delta_mean_wave2","iw_delta_std_wave2",
        "iw_actor_loss","iw_tactical_grad_norm","iw_aux_grad_norm","iw_combined_grad_norm","iw_gradient_dot",
        "iw_gradient_cosine_pre_projection","iw_gradient_conflict","iw_projection_applied","iw_cumulative_conflict_fraction",
        "iw_critic_loss","iw_replay_segments_wave1","iw_replay_segments_wave2","iw_ready_wave1","iw_ready_wave2",
        "iw_critic_update_count","iw_mae_wave1","iw_pred_mean_wave1","iw_target_mean_wave1","iw_target_std_wave1",
        "iw_mae_wave2","iw_pred_mean_wave2","iw_target_mean_wave2","iw_target_std_wave2")
  return {key:0. for key in keys}

 def _train_iw_critic(self):
  module=self.inter_wave_credit;metrics={}
  metrics.update({f"iw_replay_segments_wave{w}":float(len(module.replay[w])) for w in (1,2)})
  losses=[];stats={1:[],2:[]}
  for _ in range(module.critic_updates_per_rollout):
   available=[w for w in (1,2) if module.replay[w]]
   if not available:break
   per=max(1,module.critic_minibatch_size//len(available));wave_losses=[]
   for wave in available:
    sample=module.sample_states(wave,per,self.iw_rng);tt=lambda x,dtype=torch.float32:torch.as_tensor(x,dtype=dtype,device=self.device)
    pred=self.iw_critic(tt(sample["observations"]),tt(sample["alive_masks"]),tt(sample["credit_waves"],torch.long),tt(sample["horizons"]))
    target=tt(sample["targets"]);loss=torch.nn.functional.huber_loss(pred,target,delta=module.huber_delta)
    wave_losses.append(loss);stats[wave].append((float((pred-target).abs().mean().detach()),float(pred.mean().detach()),float(target.mean().detach())))
   total=torch.stack(wave_losses).mean() if module.wave_balanced_supervision else torch.stack(wave_losses).sum()
   self.iw_critic_optimizer.zero_grad();total.backward();nn.utils.clip_grad_norm_(self.iw_critic.parameters(),self.max_grad_norm);self.iw_critic_optimizer.step()
   module.valid_update_count+=1;losses.append(float(total.detach()))
  metrics["iw_critic_loss"]=float(np.mean(losses)) if losses else 0.
  metrics["iw_critic_update_count"]=float(module.valid_update_count)
  for wave in (1,2):
   rows=stats[wave];metrics[f"iw_mae_wave{wave}"]=float(np.mean([x[0] for x in rows])) if rows else 0.
   metrics[f"iw_pred_mean_wave{wave}"]=float(np.mean([x[1] for x in rows])) if rows else 0.
   metrics[f"iw_target_mean_wave{wave}"]=float(np.mean([x[2] for x in rows])) if rows else 0.
   metrics[f"iw_target_std_wave{wave}"]=module.target_std(wave)
   metrics[f"iw_ready_wave{wave}"]=float(module.ready(wave))
  return metrics

 @torch.no_grad()
 def _iw_rollout_credit(self,r,obs,nobs,alive,nalive,dones,waves):
  if r.remaining_horizons is None or r.next_remaining_horizons is None:return None,None,{}
  T,E=waves.shape;flat=lambda x:x.reshape(T*E,*x.shape[2:]);source=waves.reshape(-1)
  h=torch.as_tensor(r.remaining_horizons,dtype=torch.float32,device=self.device).reshape(-1)
  nh=torch.as_tensor(r.next_remaining_horizons,dtype=torch.float32,device=self.device).reshape(-1)
  q=self.iw_critic(flat(obs),flat(alive),source,h);qn=self.iw_critic(flat(nobs),flat(nalive),source,nh)
  terminal=dones.reshape(-1)>.5;qn=torch.where(terminal,torch.zeros_like(qn),qn)
  delta=(qn-q).reshape(T,E);metrics={}
  for wave in (1,2):
   mask=(waves==wave);vals=delta[mask]
   metrics[f"iw_delta_mean_wave{wave}"]=float(vals.mean()) if vals.numel() else 0.
   metrics[f"iw_delta_std_wave{wave}"]=float(vals.std(unbiased=False)) if vals.numel() else 0.
  delta,active=normalize_iw_deltas(delta,waves,{w:self.inter_wave_credit.ready(w) for w in (1,2)})
  valid=(waves<=2);metrics["iw_q_current_mean"]=float(q.reshape(T,E)[valid].mean()) if valid.any() else 0.
  metrics["iw_q_next_mean"]=float(qn.reshape(T,E)[valid].mean()) if valid.any() else 0.
  metrics["iw_active_wave1"]=float(active[waves==1].any()) if (waves==1).any() else 0.
  metrics["iw_active_wave2"]=float(active[waves==2].any()) if (waves==2).any() else 0.
  metrics["iw_actor_active"]=float(active.any())
  return delta.detach(),active,metrics

 def _update_flat_iwsc(self,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,iw_adv,iw_active):
  flat=lambda x:x.reshape(obs.shape[0]*obs.shape[1],*x.shape[2:])
  arrays=list(map(flat,(obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx)));fia=iw_adv.reshape(-1);fim=iw_active.reshape(-1);waves=[];rows=[];iw_rows=[]
  source_waves=getattr(self,"_iw_source_waves",None)
  if source_waves is None: source_waves=torch.zeros_like(fia,dtype=torch.long)
  params=self.actor.trainable_policy_parameters();N=arrays[0].shape[0]
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device);args=[x[ix] for x in arrays]
    loss=self._loss_step(*args);al,vl,en,anchor,*_=loss;tactical=al-self.entropy_coefficient*en+anchor
    dist,_=self.actor.distribution_step(args[0],self._ctx(args[9],True),None,None,args[4]);newlog=self.actor._squashed_log_prob(dist,args[2],args[1]);_,ratio=stable_ratio_terms(newlog,args[3])
    state_active=fim[ix];a=fia[ix][:,None]
    surrogate=torch.minimum(ratio*a,ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*a)
    iw_loss=combine_iw_actor_loss(surrogate,args[4],state_active,source_waves[ix],
                                  self.inter_wave_credit.wave_balanced_actor_loss)
    gt=torch.autograd.grad(tactical,params,retain_graph=True,allow_unused=True);gi=torch.autograd.grad(iw_loss,params,retain_graph=True,allow_unused=True)
    gt=[torch.zeros_like(p) if g is None else g for p,g in zip(params,gt)];gi=[torch.zeros_like(p) if g is None else g for p,g in zip(params,gi)]
    dot=sum((a*b).sum() for a,b in zip(gt,gi));nt=sum(a.square().sum() for a in gt);ni=sum(a.square().sum() for a in gi)
    conflict=bool(dot.detach()<0);projected,dot=asymmetric_tactical_projection(gt,gi)
    combined=[a+self.inter_wave_credit.actor_credit_coefficient*b for a,b in zip(gt,projected)]
    self.actor_optimizer.zero_grad();
    for p,g in zip(params,combined):p.grad=g
    combined_norm=float(torch.sqrt(sum(g.square().sum() for g in combined)).detach());ag=nn.utils.clip_grad_norm_(params,self.max_grad_norm);self.actor_optimizer.step();self.actor_update_count+=1
    self.critic_optimizer.zero_grad();(self.value_loss_coefficient*vl).backward();cg=nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm);self.critic_optimizer.step();self.critic_update_count+=1
    self.iw_gradient_step_count+=1;self.iw_conflict_count+=int(conflict)
    rows.append(self._row(loss,args[4],ag,cg));iw_rows.append({"iw_actor_loss":float(iw_loss.detach()),"iw_tactical_grad_norm":float(torch.sqrt(nt).detach()),"iw_aux_grad_norm":float(torch.sqrt(ni).detach()),"iw_combined_grad_norm":combined_norm,"iw_gradient_dot":float(dot.detach()),"iw_gradient_cosine_pre_projection":float((dot/(torch.sqrt(nt*ni)+1e-12)).detach()),"iw_gradient_conflict":float(conflict),"iw_projection_applied":float(conflict)})
  out=aggregate_update_rows(rows,self.clip_ratio)
  for key in iw_rows[0]:out[key]=float(np.mean([r[key] for r in iw_rows]))
  out["iw_cumulative_conflict_fraction"]=float(self.iw_conflict_count/max(1,self.iw_gradient_step_count));return out

 def _caiw_default_metrics(self):
  keys=("caiw_actor_active","caiw_active_wave1","caiw_active_wave2","caiw_critic_loss","caiw_adv_mean","caiw_adv_std","caiw_adv_abs_mean","caiw_adv_abs_p90","caiw_adv_abs_max","caiw_adv_mean_wave1","caiw_adv_std_wave1","caiw_adv_abs_mean_wave1","caiw_adv_abs_p90_wave1","caiw_adv_abs_max_wave1","caiw_adv_mean_wave2","caiw_adv_std_wave2","caiw_adv_abs_mean_wave2","caiw_adv_abs_p90_wave2","caiw_adv_abs_max_wave2","caiw_counterfactual_q_actual_mean","caiw_counterfactual_q_baseline_mean","caiw_action_sensitivity_abs_mean","caiw_tactical_grad_norm","caiw_aux_grad_norm_pre_projection","caiw_aux_grad_norm_post_projection","caiw_aux_grad_norm_post_trust_cap","caiw_aux_to_tactical_ratio_pre","caiw_aux_to_tactical_ratio_post","caiw_gradient_dot","caiw_gradient_cosine","caiw_conflict","caiw_projection_applied","caiw_trust_scale","caiw_trust_cap_applied","caiw_combined_grad_norm","caiw_combined_would_clip","caiw_aux_induced_clip","caiw_cumulative_conflict_fraction","caiw_cumulative_trust_cap_fraction","caiw_cumulative_aux_induced_clip_fraction")
  value={key:0. for key in keys}
  for agent in range(self.num_agents):value[f"caiw_agent{agent}_advantage_abs_mean"]=0.
  for task in CAIW_TASKS:
   for key in ("train_positive_segments","train_negative_segments","fresh_train_samples","freshness_candidate_fraction","freshness_accepted_fraction","validation_segments","validation_positive","validation_negative","validation_auroc","validation_brier","validation_climatology_brier","validation_brier_skill","prediction_mean","prediction_std","positive_prediction_mean","negative_prediction_mean","validation_pass","validation_pass_streak","ready","recent_prior","first_ready_step"):value[f"caiw_{task}_{key}"]=0.
  return value

 @torch.no_grad()
 def current_behavior_log_ratio(self,observations,alive_masks,actions,raw_actions,behavior_log_probs):
  conv=lambda x:torch.as_tensor(x,dtype=torch.float32,device=self.device)
  obs,alive,act,raw,old=map(conv,(observations,alive_masks,actions,raw_actions,behavior_log_probs))
  dist,_=self.actor.distribution_step(obs,None,None,None,alive);new=self.actor._squashed_log_prob(dist,raw,act)
  return (new-old).cpu().numpy()

 def _fresh_class_sample(self,segments,count):
  if not segments:return None,0,0
  attempts=max(count*20,count);picked=[]
  for _ in range(attempts):
   segment=segments[int(self.caiw_rng.integers(0,len(segments)))];index=int(self.caiw_rng.integers(0,len(segment["remaining_horizons"])))
   picked.append((segment,index))
  obs=np.asarray([s["observations"][i] for s,i in picked],np.float32);alive=np.asarray([s["alive_masks"][i] for s,i in picked],np.float32)
  act=np.asarray([s["actions"][i] for s,i in picked],np.float32);raw=np.asarray([s["raw_actions"][i] for s,i in picked],np.float32);old=np.asarray([s["behavior_log_probs"][i] for s,i in picked],np.float32)
  fresh=freshness_mask(self.current_behavior_log_ratio(obs,alive,act,raw,old),alive,self.counterfactual_inter_wave_credit.freshness_ratio_low,self.counterfactual_inter_wave_credit.freshness_ratio_high)
  keep=np.flatnonzero(fresh)[:count]
  if len(keep)<count:return None,attempts,int(fresh.sum())
  result={"observations":obs[keep],"alive_masks":alive[keep],"actions":act[keep],"horizons":np.asarray([picked[i][0]["remaining_horizons"][picked[i][1]] for i in keep],np.float32)}
  return result,attempts,int(fresh.sum())

 def _train_caiw_critic(self):
  module=self.counterfactual_inter_wave_credit;metrics={};loss_rows=[];fresh_total={task:0 for task in CAIW_TASKS};candidate_total={task:0 for task in CAIW_TASKS};accepted_total={task:0 for task in CAIW_TASKS}
  for task in CAIW_TASKS:
   positive,negative=module.task_classes(task);metrics[f"caiw_{task}_train_positive_segments"]=float(len(positive));metrics[f"caiw_{task}_train_negative_segments"]=float(len(negative))
  for _ in range(module.critic_updates_per_rollout):
   task_losses=[]
   for task in CAIW_TASKS:
    positive,negative=module.task_classes(task)
    if min(len(positive),len(negative))<module.min_train_segments_per_class:continue
    pos,pc,pa=self._fresh_class_sample(positive,module.train_states_per_class_per_task);neg,nc,na=self._fresh_class_sample(negative,module.train_states_per_class_per_task)
    candidate_total[task]+=pc+nc;accepted_total[task]+=pa+na
    if pos is None or neg is None:continue
    fresh_total[task]+=2*module.train_states_per_class_per_task
    join=lambda key:np.concatenate((pos[key],neg[key]),0);tt=lambda x,dtype=torch.float32:torch.as_tensor(x,dtype=dtype,device=self.device)
    source=torch.full((2*module.train_states_per_class_per_task,),TASK_SOURCE_WAVE[task],dtype=torch.long,device=self.device)
    logits=self.caiw_critic(tt(join("observations")),tt(join("actions")),tt(join("alive_masks")),source,tt(join("horizons")))[:,TASK_HEAD[task]]
    targets=torch.cat((torch.ones(module.train_states_per_class_per_task,device=self.device),torch.zeros(module.train_states_per_class_per_task,device=self.device)))
    task_losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits,targets))
   if task_losses:
    loss=torch.stack(task_losses).mean();self.caiw_critic_optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.caiw_critic.parameters(),self.max_grad_norm);self.caiw_critic_optimizer.step();loss_rows.append(float(loss.detach()))
  metrics["caiw_critic_loss"]=float(np.mean(loss_rows)) if loss_rows else 0.
  for task in CAIW_TASKS:
   metrics[f"caiw_{task}_fresh_train_samples"]=float(fresh_total[task]);metrics[f"caiw_{task}_freshness_candidate_fraction"]=float(candidate_total[task]>0);metrics[f"caiw_{task}_freshness_accepted_fraction"]=float(accepted_total[task]/candidate_total[task]) if candidate_total[task] else 0.
  return metrics

 @torch.no_grad()
 def _validate_caiw_tasks(self):
  module=self.counterfactual_inter_wave_credit;metrics={}
  if (self.ppo_update_count+1)%module.validation_interval_updates!=0:
   for task in CAIW_TASKS:
    metrics[f"caiw_{task}_ready"]=float(module.task_ready[task]);metrics[f"caiw_{task}_validation_pass_streak"]=float(module.validation_pass_streak[task]);metrics[f"caiw_{task}_recent_prior"]=module.recent_prior(task);metrics[f"caiw_{task}_first_ready_step"]=float(module.first_ready_sampled_steps[task] or 0)
   return metrics
  module.validation_check_count+=1
  for task in CAIW_TASKS:
   wave=TASK_SOURCE_WAVE[task];predictions=[];labels=[];prior=module.recent_prior(task)
   for segment in module.validation[wave]:
    ratios=self.current_behavior_log_ratio(segment["observations"],segment["alive_masks"],segment["actions"],segment["raw_actions"],segment["behavior_log_probs"])
    fresh=np.flatnonzero(freshness_mask(ratios,segment["alive_masks"],module.freshness_ratio_low,module.freshness_ratio_high))
    if len(fresh)<module.min_fresh_states_per_validation_segment:continue
    obs=torch.as_tensor(segment["observations"][fresh],dtype=torch.float32,device=self.device);act=torch.as_tensor(segment["actions"][fresh],dtype=torch.float32,device=self.device);alive=torch.as_tensor(segment["alive_masks"][fresh],dtype=torch.float32,device=self.device);h=torch.as_tensor(segment["remaining_horizons"][fresh],dtype=torch.float32,device=self.device);source=torch.full((len(fresh),),wave,dtype=torch.long,device=self.device)
    raw=self.caiw_critic(obs,act,alive,source,h)[:,TASK_HEAD[task]];predictions.append(float(prior_corrected_probability(raw,prior,module.prior_probability_epsilon).mean()));labels.append(module.task_label(task,segment))
   y=np.asarray(labels,dtype=np.float64);p=np.asarray(predictions,dtype=np.float64);n=len(y);pos=int(y.sum());neg=n-pos;auroc=binary_auroc(y,p) if n else None
   brier=float(np.mean((p-y)**2)) if n else None;clim=float(np.mean((y-y.mean())**2)) if n else None;bss=(1-brier/clim) if clim is not None and clim>0 else None
   vm={"validation_segments":n,"validation_positive":pos,"validation_negative":neg,"validation_auroc":auroc,"validation_brier":brier,"validation_climatology_brier":clim,"validation_brier_skill":bss,"prediction_mean":float(p.mean()) if n else None,"prediction_std":float(p.std()) if n else None,"positive_prediction_mean":float(p[y==1].mean()) if pos else None,"negative_prediction_mean":float(p[y==0].mean()) if neg else None}
   passed=module.apply_validation(task,vm,self.sampled_steps);vm.update({"validation_pass":float(passed),"validation_pass_streak":module.validation_pass_streak[task],"ready":float(module.task_ready[task]),"recent_prior":prior,"first_ready_step":module.first_ready_sampled_steps[task] or 0})
   for key,value in vm.items():metrics[f"caiw_{task}_{key}"]=float(value) if value is not None else 0.
  return metrics

 def _caiw_quality(self,logits,waves):
  module=self.counterfactual_inter_wave_credit;result=torch.zeros(len(waves),device=logits.device);active=torch.zeros(len(waves),dtype=torch.bool,device=logits.device)
  w1=waves==1;ready1=[]
  for task in (W1_TO_W2,W1_TO_W3):
   if module.task_ready[task]:ready1.append(prior_corrected_probability(logits[:,TASK_HEAD[task]],module.recent_prior(task),module.prior_probability_epsilon))
  if ready1 and w1.any():result[w1]=torch.stack(ready1).mean(0)[w1];active[w1]=True
  w2=waves==2
  if module.task_ready[W2_TO_W3] and w2.any():result[w2]=prior_corrected_probability(logits[:,1],module.recent_prior(W2_TO_W3),module.prior_probability_epsilon)[w2];active[w2]=True
  return result,active

 @torch.no_grad()
 def _caiw_rollout_credit(self,r,obs,actions,raw,oldlog,alive,waves):
  module=self.counterfactual_inter_wave_credit;T,E,A=alive.shape;flat=lambda x:x.reshape(T*E,*x.shape[2:]);o=flat(obs);a=flat(actions);m=flat(alive);w=waves.reshape(-1);h=torch.as_tensor(r.remaining_horizons,dtype=torch.float32,device=self.device).reshape(-1)
  advantages=torch.zeros(T*E,A,device=self.device);active_all=torch.zeros(T*E,dtype=torch.bool,device=self.device);actual_values=[];baseline_values=[]
  eligible=(w<=2)
  for start in range(0,T*E,module.counterfactual_batch_size):
   stop=min(T*E,start+module.counterfactual_batch_size);index=torch.arange(start,stop,device=self.device);index=index[eligible[index]]
   if not len(index):continue
   logits=self.caiw_critic(o[index],a[index],m[index],w[index],h[index]);q_actual,active=self._caiw_quality(logits,w[index]);active_index=index[active]
   if not len(active_index):continue
   local=torch.nonzero(active,as_tuple=False).reshape(-1);oa=o[active_index];aa=a[active_index];ma=m[active_index];wa=w[active_index];ha=h[active_index]
   dist,_=self.actor.distribution_step(oa,None,None,None,ma)
   for agent in range(A):
    alternatives=torch.tanh(antithetic_latent_actions(dist.mean[:,agent],dist.stddev[:,agent],self.caiw_rng,module.counterfactual_samples));n=len(active_index);joint=aa[:,None].expand(n,module.counterfactual_samples,A,aa.shape[-1]).clone();joint[:,:,agent]=alternatives
    rep=lambda x:x[:,None].expand(n,module.counterfactual_samples,*x.shape[1:]).reshape(n*module.counterfactual_samples,*x.shape[1:])
    cf_logits=self.caiw_critic(rep(oa),joint.reshape(n*module.counterfactual_samples,A,aa.shape[-1]),rep(ma),wa[:,None].expand(n,module.counterfactual_samples).reshape(-1),ha[:,None].expand(n,module.counterfactual_samples).reshape(-1));q_cf,_=self._caiw_quality(cf_logits,wa[:,None].expand(n,module.counterfactual_samples).reshape(-1));baseline=q_cf.reshape(n,module.counterfactual_samples).mean(1)
    advantages[active_index,agent]=(q_actual[local]-baseline)*ma[:,agent];baseline_values.append(baseline);actual_values.append(q_actual[local])
   active_all[active_index]=True
  values=advantages[active_all];metrics={"caiw_actor_active":float(active_all.any()),"caiw_active_wave1":float((active_all&(w==1)).any()),"caiw_active_wave2":float((active_all&(w==2)).any())}
  live=values[m[active_all]>.5] if active_all.any() else torch.empty(0,device=self.device)
  def stats(prefix,x):
   if not x.numel():return {f"{prefix}_mean":0.,f"{prefix}_std":0.,f"{prefix}_abs_mean":0.,f"{prefix}_abs_p90":0.,f"{prefix}_abs_max":0.}
   z=x.abs();return {f"{prefix}_mean":float(x.mean()),f"{prefix}_std":float(x.std(unbiased=False)),f"{prefix}_abs_mean":float(z.mean()),f"{prefix}_abs_p90":float(torch.quantile(z,.9)),f"{prefix}_abs_max":float(z.max())}
  metrics.update(stats("caiw_adv",live))
  for wave in (1,2):
   mask=active_all&(w==wave);x=advantages[mask];x=x[m[mask]>.5] if mask.any() else torch.empty(0,device=self.device);metrics.update(stats(f"caiw_adv_wave{wave}",x))
  metrics["caiw_counterfactual_q_actual_mean"]=float(torch.cat(actual_values).mean()) if actual_values else 0.;metrics["caiw_counterfactual_q_baseline_mean"]=float(torch.cat(baseline_values).mean()) if baseline_values else 0.;metrics["caiw_action_sensitivity_abs_mean"]=metrics["caiw_adv_abs_mean"]
  for agent in range(A):
   mask=active_all&(m[:,agent]>.5);metrics[f"caiw_agent{agent}_advantage_abs_mean"]=float(advantages[mask,agent].abs().mean()) if mask.any() else 0.
  return advantages.reshape(T,E,A),active_all.reshape(T,E),metrics

 def _update_flat_caiw(self,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,caiw_adv,caiw_active):
  flat=lambda x:x.reshape(obs.shape[0]*obs.shape[1],*x.shape[2:]);arrays=list(map(flat,(obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx)));ca=flat(caiw_adv);active=caiw_active.reshape(-1);waves=self._caiw_source_waves;params=self.actor.trainable_policy_parameters();N=arrays[0].shape[0];rows=[];crows=[];module=self.counterfactual_inter_wave_credit
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device);args=[x[ix] for x in arrays];loss=self._loss_step(*args);al,vl,en,anchor,*_=loss;tactical=al-self.entropy_coefficient*en+anchor
    dist,_=self.actor.distribution_step(args[0],None,None,None,args[4]);newlog=self.actor._squashed_log_prob(dist,args[2],args[1]);_,ratio=stable_ratio_terms(newlog,args[3]);state_active=active[ix]
    surrogate=torch.minimum(ratio*ca[ix],ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*ca[ix]);wave_losses=[]
    for wave in (1,2):
     mask=state_active&(waves[ix]==wave)
     if mask.any():wave_losses.append(-masked_mean(surrogate,args[4]*mask[:,None].to(args[4].dtype)))
    caiw_loss=torch.stack(wave_losses).mean() if wave_losses else surrogate.sum()*0
    gt=torch.autograd.grad(tactical,params,retain_graph=True,allow_unused=True);gc=torch.autograd.grad(caiw_loss,params,retain_graph=True,allow_unused=True);gt=[torch.zeros_like(p) if g is None else g for p,g in zip(params,gt)];gc=[torch.zeros_like(p) if g is None else g for p,g in zip(params,gc)]
    pre=torch.sqrt(sum(g.square().sum() for g in gc));projected,dot=asymmetric_tactical_projection(gt,gc);post_proj=torch.sqrt(sum(g.square().sum() for g in projected));trusted,scale,nt,_=caiw_trust_cap(gt,projected,module.auxiliary_gradient_ratio_cap);post=torch.sqrt(sum(g.square().sum() for g in trusted));combined=[a+b for a,b in zip(gt,trusted)];combined_norm=torch.sqrt(sum(g.square().sum() for g in combined));conflict=bool(dot.detach()<0);cap=bool(scale.detach()<1-1e-12);induced=bool(nt<=self.max_grad_norm and combined_norm>self.max_grad_norm)
    self.actor_optimizer.zero_grad();
    for p,g in zip(params,combined):p.grad=g
    ag=nn.utils.clip_grad_norm_(params,self.max_grad_norm);self.actor_optimizer.step();self.actor_update_count+=1
    self.critic_optimizer.zero_grad();(self.value_loss_coefficient*vl).backward();cg=nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm);self.critic_optimizer.step();self.critic_update_count+=1
    self.caiw_gradient_step_count+=1;self.caiw_conflict_count+=int(conflict);self.caiw_trust_cap_count+=int(cap);self.caiw_aux_induced_clip_count+=int(induced)
    rows.append(self._row(loss,args[4],ag,cg));crows.append({"caiw_tactical_grad_norm":float(nt),"caiw_aux_grad_norm_pre_projection":float(pre),"caiw_aux_grad_norm_post_projection":float(post_proj),"caiw_aux_grad_norm_post_trust_cap":float(post),"caiw_aux_to_tactical_ratio_pre":float(pre/(nt+1e-12)),"caiw_aux_to_tactical_ratio_post":float(post/(nt+1e-12)),"caiw_gradient_dot":float(dot),"caiw_gradient_cosine":float(dot/(nt*pre+1e-12)),"caiw_conflict":float(conflict),"caiw_projection_applied":float(conflict),"caiw_trust_scale":float(scale),"caiw_trust_cap_applied":float(cap),"caiw_combined_grad_norm":float(combined_norm),"caiw_combined_would_clip":float(combined_norm>self.max_grad_norm),"caiw_aux_induced_clip":float(induced)})
  out=aggregate_update_rows(rows,self.clip_ratio)
  for key in crows[0]:out[key]=float(np.mean([row[key] for row in crows]))
  count=max(1,self.caiw_gradient_step_count);out["caiw_cumulative_conflict_fraction"]=self.caiw_conflict_count/count;out["caiw_cumulative_trust_cap_fraction"]=self.caiw_trust_cap_count/count;out["caiw_cumulative_aux_induced_clip_fraction"]=self.caiw_aux_induced_clip_count/count;return out

 def _brsc_default_metrics(self):
  keys=("brsc_actor_active","brsc_active_wave1","brsc_active_wave2","brsc_boundary_events_wave1","brsc_boundary_events_wave2","brsc_credited_boundaries_wave1","brsc_credited_boundaries_wave2","brsc_segment_length_mean","brsc_segment_length_p50","brsc_segment_length_p90","brsc_credit_distance_mean","brsc_credit_distance_p90","brsc_credited_transition_fraction","brsc_credited_alive_agent_fraction","brsc_adv_mean","brsc_adv_std","brsc_adv_abs_mean","brsc_adv_abs_p90","brsc_adv_abs_max","brsc_critic_loss","brsc_tactical_grad_norm","brsc_aux_grad_norm_pre_projection","brsc_aux_grad_norm_post_projection","brsc_aux_grad_norm_post_trust_cap","brsc_aux_to_tactical_ratio_pre","brsc_aux_to_tactical_ratio_post","brsc_gradient_dot","brsc_gradient_cosine","brsc_conflict","brsc_projection_applied","brsc_trust_scale","brsc_trust_cap_applied","brsc_combined_grad_norm","brsc_combined_would_clip","brsc_aux_induced_clip","brsc_cumulative_conflict_fraction","brsc_cumulative_trust_cap_fraction","brsc_cumulative_aux_induced_clip_fraction")
  result={key:0. for key in keys}
  for wave in (1,2):
   for name in ("boundary_q_mean","boundary_baseline_mean","boundary_credit_mean","boundary_credit_abs_mean","adv_mean","adv_std","adv_abs_mean","adv_abs_p90","adv_abs_max"):result[f"brsc_{name}_wave{wave}"]=0.
  for task in BRSC_TASKS:
   for name in ("train_positive","train_negative","validation_segments","validation_positive","validation_negative","validation_auroc","validation_brier","validation_climatology_brier","validation_brier_skill","prediction_mean","prediction_std","positive_prediction_mean","negative_prediction_mean","validation_pass","validation_pass_streak","ready","recent_prior","first_ready_step"):result[f"brsc_{task}_{name}"]=0.
  return result

 def _brsc_boundary_tensors(self,rows):
  tt=lambda x,dtype=torch.float32:torch.as_tensor(x,dtype=dtype,device=self.device)
  return (tt(np.asarray([x["entry_observation"] for x in rows],np.float32)),
          tt(np.asarray([x["entry_alive_mask"] for x in rows],np.float32)),
          tt(np.asarray([x["source_wave"] for x in rows],np.int64),torch.long),
          tt(np.asarray([x["entry_remaining_horizon"] for x in rows],np.float32)))

 def _sample_brsc_class(self,rows,count):
  indices=self.brsc_rng.integers(0,len(rows),size=count);return [rows[int(i)] for i in indices]

 def _train_brsc_critic(self):
  module=self.boundary_redistributed_segment_credit;metrics={};loss_rows=[]
  for task in BRSC_TASKS:
   positive,negative=module.task_classes(task);metrics[f"brsc_{task}_train_positive"]=float(len(positive));metrics[f"brsc_{task}_train_negative"]=float(len(negative))
  for _ in range(module.critic_updates_per_rollout):
   task_losses=[]
   for task in BRSC_TASKS:
    positive,negative=module.task_classes(task)
    if min(len(positive),len(negative))<module.min_train_boundaries_per_class:continue
    n=module.critic_samples_per_class_per_task;rows=self._sample_brsc_class(positive,n)+self._sample_brsc_class(negative,n)
    obs,alive,source,horizon=self._brsc_boundary_tensors(rows);logits=self.brsc_critic(obs,alive,source,horizon)[:,BRSC_TASK_HEAD[task]]
    targets=torch.cat((torch.ones(n,device=self.device),torch.zeros(n,device=self.device)))
    task_losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits,targets))
   if task_losses:
    loss=torch.stack(task_losses).mean();self.brsc_critic_optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.brsc_critic.parameters(),self.max_grad_norm);self.brsc_critic_optimizer.step();loss_rows.append(float(loss.detach()))
  metrics["brsc_critic_loss"]=float(np.mean(loss_rows)) if loss_rows else 0.;return metrics

 @torch.no_grad()
 def _validate_brsc_tasks(self):
  module=self.boundary_redistributed_segment_credit;metrics={}
  if (self.ppo_update_count+1)%module.validation_interval_updates!=0:
   for task in BRSC_TASKS:
    metrics[f"brsc_{task}_ready"]=float(module.task_ready[task]);metrics[f"brsc_{task}_validation_pass_streak"]=float(module.validation_pass_streak[task]);metrics[f"brsc_{task}_recent_prior"]=module.recent_prior(task);metrics[f"brsc_{task}_first_ready_step"]=float(module.first_ready_sampled_steps[task] or 0)
   return metrics
  module.validation_check_count+=1
  for task in BRSC_TASKS:
   wave=BRSC_TASK_SOURCE_WAVE[task];rows=list(module.validation[wave]);prior=module.recent_prior(task)
   if rows:
    obs,alive,source,horizon=self._brsc_boundary_tensors(rows);raw=self.brsc_critic(obs,alive,source,horizon)[:,BRSC_TASK_HEAD[task]]
    predictions=prior_corrected_probability(raw,prior,module.prior_probability_epsilon).cpu().numpy();labels=np.asarray([module.task_label(task,row) for row in rows],np.float64)
   else:predictions=np.asarray([],np.float64);labels=np.asarray([],np.float64)
   n=len(labels);pos=int(labels.sum());neg=n-pos;auroc=binary_auroc(labels,predictions) if n else None
   brier=float(np.mean((predictions-labels)**2)) if n else None;clim=float(np.mean((labels-labels.mean())**2)) if n else None;bss=(1-brier/clim) if clim is not None and clim>0 else None
   row={"validation_segments":n,"validation_positive":pos,"validation_negative":neg,"validation_auroc":auroc,"validation_brier":brier,"validation_climatology_brier":clim,"validation_brier_skill":bss,"prediction_mean":float(predictions.mean()) if n else None,"prediction_std":float(predictions.std()) if n else None,"positive_prediction_mean":float(predictions[labels==1].mean()) if pos else None,"negative_prediction_mean":float(predictions[labels==0].mean()) if neg else None}
   passed=module.apply_validation(task,row,self.sampled_steps);row.update({"validation_pass":float(passed),"validation_pass_streak":module.validation_pass_streak[task],"ready":float(module.task_ready[task]),"recent_prior":prior,"first_ready_step":module.first_ready_sampled_steps[task] or 0})
   for key,value in row.items():metrics[f"brsc_{task}_{key}"]=float(value) if value is not None else 0.
  return metrics

 @torch.no_grad()
 def _brsc_quality(self,logits,waves):
  module=self.boundary_redistributed_segment_credit;q=torch.zeros(len(waves),device=logits.device);baseline=torch.zeros_like(q);active=torch.zeros(len(waves),dtype=torch.bool,device=logits.device)
  for index,wave in enumerate(waves.tolist()):
   tasks=(W1_BOUNDARY_TO_W2,W1_BOUNDARY_TO_W3) if wave==1 else ((W2_BOUNDARY_TO_W3,) if wave==2 else ())
   tasks=[task for task in tasks if module.task_ready[task]]
   if tasks:
    probs=[prior_corrected_probability(logits[index,BRSC_TASK_HEAD[task]],module.recent_prior(task),module.prior_probability_epsilon) for task in tasks]
    priors=[module.recent_prior(task) for task in tasks];q[index]=torch.stack(probs).mean();baseline[index]=float(np.mean(priors));active[index]=True
  return q,baseline,active

 @torch.no_grad()
 def _brsc_rollout_credit(self,r,next_obs,alive,next_alive,dones,waves):
  if r.wave_transition_flags is None or r.next_remaining_horizons is None:return None,None,{}
  module=self.boundary_redistributed_segment_credit;rollout_alive=alive;T,E=waves.shape;flags=torch.as_tensor(r.wave_transition_flags,dtype=torch.bool,device=self.device)&(waves<=2)
  boundary_credit=torch.zeros((T,E),device=self.device);credited=torch.zeros((T,E),dtype=torch.bool,device=self.device);q_values={1:[],2:[]};base_values={1:[],2:[]}
  indices=torch.nonzero(flags,as_tuple=False)
  if len(indices):
   boundary_obs=next_obs[indices[:,0],indices[:,1]];boundary_alive=next_alive[indices[:,0],indices[:,1]];boundary_source=waves[indices[:,0],indices[:,1]]
   boundary_horizon=torch.as_tensor(r.next_remaining_horizons,dtype=torch.float32,device=self.device)[indices[:,0],indices[:,1]]
   logits=self.brsc_critic(boundary_obs,boundary_alive,boundary_source,boundary_horizon);q,baseline,active=self._brsc_quality(logits,boundary_source);credit=q-baseline
   for j,(t,e) in enumerate(indices.tolist()):
    if active[j]:boundary_credit[t,e]=credit[j];credited[t,e]=True;q_values[int(boundary_source[j])].append(float(q[j]));base_values[int(boundary_source[j])].append(float(baseline[j]))
  adv_np,active_np,lengths=redistribute_boundary_credit(waves.cpu().numpy(),dones.cpu().numpy(),credited.cpu().numpy(),boundary_credit.cpu().numpy(),self.gamma*self.gae_lambda)
  advantages=torch.as_tensor(adv_np,dtype=torch.float32,device=self.device);active=torch.as_tensor(active_np,dtype=torch.bool,device=self.device)
  metrics={"brsc_actor_active":float(active.any()),"brsc_active_wave1":float((active&(waves==1)).any()),"brsc_active_wave2":float((active&(waves==2)).any())}
  for wave in (1,2):
   event=(flags&(waves==wave));used=(credited&(waves==wave));z=boundary_credit[used]
   metrics[f"brsc_boundary_events_wave{wave}"]=float(event.sum());metrics[f"brsc_credited_boundaries_wave{wave}"]=float(used.sum())
   metrics[f"brsc_boundary_q_mean_wave{wave}"]=float(np.mean(q_values[wave])) if q_values[wave] else 0.;metrics[f"brsc_boundary_baseline_mean_wave{wave}"]=float(np.mean(base_values[wave])) if base_values[wave] else 0.
   metrics[f"brsc_boundary_credit_mean_wave{wave}"]=float(z.mean()) if z.numel() else 0.;metrics[f"brsc_boundary_credit_abs_mean_wave{wave}"]=float(z.abs().mean()) if z.numel() else 0.
  def stats(prefix,x):
   if not x.numel():return {f"{prefix}_mean":0.,f"{prefix}_std":0.,f"{prefix}_abs_mean":0.,f"{prefix}_abs_p90":0.,f"{prefix}_abs_max":0.}
   absolute=x.abs();return {f"{prefix}_mean":float(x.mean()),f"{prefix}_std":float(x.std(unbiased=False)),f"{prefix}_abs_mean":float(absolute.mean()),f"{prefix}_abs_p90":float(torch.quantile(absolute,.9)),f"{prefix}_abs_max":float(absolute.max())}
  metrics.update(stats("brsc_adv",advantages[active]))
  for wave in (1,2):metrics.update(stats(f"brsc_adv_wave{wave}",advantages[active&(waves==wave)]))
  distances=[]
  for length in lengths:distances.extend(range(length))
  metrics["brsc_segment_length_mean"]=float(np.mean(lengths)) if lengths else 0.;metrics["brsc_segment_length_p50"]=float(np.quantile(lengths,.5)) if lengths else 0.;metrics["brsc_segment_length_p90"]=float(np.quantile(lengths,.9)) if lengths else 0.
  metrics["brsc_credit_distance_mean"]=float(np.mean(distances)) if distances else 0.;metrics["brsc_credit_distance_p90"]=float(np.quantile(distances,.9)) if distances else 0.
  metrics["brsc_credited_transition_fraction"]=float(active.float().mean());metrics["brsc_credited_alive_agent_fraction"]=float((active[...,None]&(rollout_alive>.5)).float().sum()/((rollout_alive>.5).float().sum().clamp_min(1)))
  return advantages.detach(),active,metrics

 def _update_flat_brsc(self,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,brsc_adv,brsc_active):
  flat=lambda x:x.reshape(obs.shape[0]*obs.shape[1],*x.shape[2:]);arrays=list(map(flat,(obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx)));ba=brsc_adv.reshape(-1);active=brsc_active.reshape(-1);waves=self._brsc_source_waves;params=self.actor.trainable_policy_parameters();N=arrays[0].shape[0];rows=[];brows=[];module=self.boundary_redistributed_segment_credit
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device);args=[x[ix] for x in arrays];loss=self._loss_step(*args);al,vl,en,anchor,*_=loss;tactical=al-self.entropy_coefficient*en+anchor
    dist,_=self.actor.distribution_step(args[0],None,None,None,args[4]);newlog=self.actor._squashed_log_prob(dist,args[2],args[1]);_,ratio=stable_ratio_terms(newlog,args[3]);state_active=active[ix];team=ba[ix][:,None]
    surrogate=torch.minimum(ratio*team,ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*team);wave_losses=[]
    for wave in (1,2):
     mask=state_active&(waves[ix]==wave)
     if mask.any():wave_losses.append(-masked_mean(surrogate,args[4]*mask[:,None].to(args[4].dtype)))
    brsc_loss=torch.stack(wave_losses).mean() if wave_losses else surrogate.sum()*0
    gt=torch.autograd.grad(tactical,params,retain_graph=True,allow_unused=True);gb=torch.autograd.grad(brsc_loss,params,retain_graph=True,allow_unused=True);gt=[torch.zeros_like(p) if g is None else g for p,g in zip(params,gt)];gb=[torch.zeros_like(p) if g is None else g for p,g in zip(params,gb)]
    pre=torch.sqrt(sum(g.square().sum() for g in gb));projected,dot=asymmetric_tactical_projection(gt,gb);post_proj=torch.sqrt(sum(g.square().sum() for g in projected));trusted,scale,nt,_=caiw_trust_cap(gt,projected,module.auxiliary_gradient_ratio_cap);post=torch.sqrt(sum(g.square().sum() for g in trusted));combined=[a+b for a,b in zip(gt,trusted)];combined_norm=torch.sqrt(sum(g.square().sum() for g in combined));conflict=bool(dot.detach()<0);cap=bool(scale.detach()<1-1e-12);induced=bool(nt<=self.max_grad_norm and combined_norm>self.max_grad_norm)
    self.actor_optimizer.zero_grad();
    for p,g in zip(params,combined):p.grad=g
    ag=nn.utils.clip_grad_norm_(params,self.max_grad_norm);self.actor_optimizer.step();self.actor_update_count+=1
    self.critic_optimizer.zero_grad();(self.value_loss_coefficient*vl).backward();cg=nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm);self.critic_optimizer.step();self.critic_update_count+=1
    self.brsc_gradient_step_count+=1;self.brsc_conflict_count+=int(conflict);self.brsc_trust_cap_count+=int(cap);self.brsc_aux_induced_clip_count+=int(induced)
    rows.append(self._row(loss,args[4],ag,cg));brows.append({"brsc_tactical_grad_norm":float(nt),"brsc_aux_grad_norm_pre_projection":float(pre),"brsc_aux_grad_norm_post_projection":float(post_proj),"brsc_aux_grad_norm_post_trust_cap":float(post),"brsc_aux_to_tactical_ratio_pre":float(pre/(nt+1e-12)),"brsc_aux_to_tactical_ratio_post":float(post/(nt+1e-12)),"brsc_gradient_dot":float(dot),"brsc_gradient_cosine":float(dot/(nt*pre+1e-12)),"brsc_conflict":float(conflict),"brsc_projection_applied":float(conflict),"brsc_trust_scale":float(scale),"brsc_trust_cap_applied":float(cap),"brsc_combined_grad_norm":float(combined_norm),"brsc_combined_would_clip":float(combined_norm>self.max_grad_norm),"brsc_aux_induced_clip":float(induced)})
  out=aggregate_update_rows(rows,self.clip_ratio)
  for key in brows[0]:out[key]=float(np.mean([row[key] for row in brows]))
  count=max(1,self.brsc_gradient_step_count);out["brsc_cumulative_conflict_fraction"]=self.brsc_conflict_count/count;out["brsc_cumulative_trust_cap_fraction"]=self.brsc_trust_cap_count/count;out["brsc_cumulative_aux_induced_clip_fraction"]=self.brsc_aux_induced_clip_count/count;return out

 def _pwtr_tensors(self,batch):
  tt=lambda key,dtype=torch.float32:torch.as_tensor(batch[key],dtype=dtype,device=self.device)
  return {"observations":tt("observations"),"next_observations":tt("next_observations"),
   "raw_actions":tt("raw_actions"),"behavior_log_probs":tt("behavior_log_probs"),
   "rewards":tt("rewards"),"dones":tt("dones"),"alive_masks":tt("alive_masks"),
   "next_alive_masks":tt("next_alive_masks"),"valid_time_mask":tt("valid_time_mask"),
   "wave_indices":tt("wave_indices",torch.long),
   "collection_ppo_update_ids":tt("collection_ppo_update_ids",torch.long),
   "segment_generations":tt("segment_generations",torch.long)}

 def _pwtr_policy_values(self,tensors):
  obs=tensors["observations"];nobs=tensors["next_observations"];alive=tensors["alive_masks"];nalive=tensors["next_alive_masks"]
  shape=obs.shape[:2];fo=obs.reshape(-1,*obs.shape[2:]);fn=nobs.reshape(-1,*nobs.shape[2:]);fa=alive.reshape(-1,alive.shape[-1]);fna=nalive.reshape(-1,nalive.shape[-1])
  dist,_=self.actor.distribution_step(fo,None,None,None,fa)
  raw=tensors["raw_actions"].reshape(-1,*tensors["raw_actions"].shape[2:]);actions=torch.tanh(raw)
  newlog=self.actor._squashed_log_prob(dist,raw,actions).reshape(*shape,self.num_agents)
  values,_=self.critic.forward_step(fo,fa,None,None,None);next_values,_=self.critic.forward_step(fn,fna,None,None,None)
  return newlog,values.reshape(*shape,self.num_agents),next_values.reshape(*shape,self.num_agents)

 @torch.no_grad()
 def _pwtr_segment_diagnostics(self,segment):
  stacked=self.persistent_wave_trajectory_replay.stack_segments([segment]);tensors=self._pwtr_tensors(stacked)
  newlog,values,next_values=self._pwtr_policy_values(tensors)
  _,_,joint_log,joint_rho=clipped_importance_weights(newlog,tensors["behavior_log_probs"],tensors["alive_masks"],tensors["valid_time_mask"])
  valid_alive=tensors["alive_masks"]*tensors["valid_time_mask"].unsqueeze(-1)
  continuation=(1-tensors["dones"]).unsqueeze(-1)*tensors["next_alive_masks"]
  td=(tensors["rewards"]+self.gamma*continuation*next_values-values).abs()
  learning=float((td*valid_alive).sum()/valid_alive.sum().clamp_min(1))
  valid_state=tensors["valid_time_mask"]*(tensors["alive_masks"].sum(-1)>0).to(torch.float32)
  ess,divergence,freshness=normalized_ess_and_freshness(joint_log,valid_state)
  priority=(learning+1e-6)*float(freshness)
  return {"learning_potential":learning,"normalized_ess":float(ess),"joint_abs_log_ratio":float(divergence),"freshness":float(freshness),"priority":priority,
   "rho_mean":float(joint_rho[valid_state>.5].mean()) if bool((valid_state>.5).any()) else 0.}

 def _pwtr_replay_phase(self):
  module=self.persistent_wave_trajectory_replay;generation=int(module.current_rollout_generation)
  result=module.default_metrics(None);budget=replay_batch_budget(module.current_later_states)
  for wave in (1,2,3):result.pop(f"pwtr_fresh_w{wave}_states",None)
  result["pwtr_replay_batches_budget"]=float(budget)
  result["pwtr_replay_budget_fill_fraction"]=replay_budget_fill_fraction(0,budget)
  if module.fresh_rollout_count<=1 or budget==0:return result
  rows=[];type_states={name:0 for name in (W2_INTERNAL,W3_INTERNAL,BRIDGE_12,BRIDGE_23)}
  actor_steps_before=module.replay_actor_optimizer_steps;critic_steps_before=module.replay_critic_optimizer_steps
  for _ in range(budget):
   segments=module.sample_batch(generation,self._pwtr_segment_diagnostics if module.priority_enabled else None)
   if not segments:break
   diagnostics=[]
   for segment in segments:
    diagnostic=self._pwtr_segment_diagnostics(segment);segment["priority_cache"]=diagnostic;diagnostics.append(diagnostic)
    type_states[segment["segment_type"]]+=int(segment["valid_time_mask"].sum())
   batch=module.stack_segments(segments);tensors=self._pwtr_tensors(batch)
   newlog,values,next_values=self._pwtr_policy_values(tensors)
   logratio,individual_rho,joint_log,joint_rho=clipped_importance_weights(
    newlog,tensors["behavior_log_probs"],tensors["alive_masks"],tensors["valid_time_mask"])
   with torch.no_grad():
    targets,advantages=vtrace_targets_and_advantages(tensors["rewards"],values.detach(),next_values.detach(),
     tensors["dones"],tensors["alive_masks"],tensors["next_alive_masks"],tensors["valid_time_mask"],joint_rho,self.gamma)
   valid_alive=tensors["alive_masks"]*tensors["valid_time_mask"].unsqueeze(-1)
   critic_loss=((values-targets).square()*valid_alive).sum()/valid_alive.sum().clamp_min(1)
   critic_grad=0.
   if module.critic_replay:
    self.critic_optimizer.zero_grad();(self.value_loss_coefficient*critic_loss).backward()
    critic_grad=float(nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm));self.critic_optimizer.step()
    self.critic_update_count+=1;module.replay_critic_optimizer_steps+=1
   transition_ages,actor_mask=transition_actor_age_mask(tensors["collection_ppo_update_ids"],generation,
    tensors["valid_time_mask"],tensors["alive_masks"])
   actor_loss=values.new_zeros(());actor_grad=0.
   if module.actor_replay and bool((actor_mask>.5).any()):
    actor_loss=-(individual_rho*advantages*newlog*actor_mask).sum()/actor_mask.sum().clamp_min(1)
    self.actor_optimizer.zero_grad();actor_loss.backward()
    if self.wave1_sensitivity_gating.enabled:self.wave1_sensitivity_gating.gate_actor_gradients(self.actor.named_parameters())
    actor_grad=float(nn.utils.clip_grad_norm_(self.actor.trainable_policy_parameters(),self.max_grad_norm));self.actor_optimizer.step()
    self.actor_update_count+=1;module.replay_actor_optimizer_steps+=1
   valid_state=tensors["valid_time_mask"]*(tensors["alive_masks"].sum(-1)>0).to(torch.float32)
   rho_values=joint_rho[valid_state>.5]
   rows.append({"valid_states":float(tensors["valid_time_mask"].sum()),"alive_samples":float(valid_alive.sum()),
    "mean_transition_age":float((transition_ages.float()*tensors["valid_time_mask"]).sum()/tensors["valid_time_mask"].sum().clamp_min(1)),
    "max_transition_age":float(transition_ages[tensors["valid_time_mask"]>.5].max()),
    "learning":float(np.mean([row["learning_potential"] for row in diagnostics])),
    "freshness":float(np.mean([row["freshness"] for row in diagnostics])),
    "priority":float(np.mean([row["priority"] for row in diagnostics])),
    "joint_abs_log_ratio":float(np.mean([row["joint_abs_log_ratio"] for row in diagnostics])),
    "normalized_ess":float(np.mean([row["normalized_ess"] for row in diagnostics])),
    "actor_eligible_fraction":float(actor_mask.sum()/valid_alive.sum().clamp_min(1)),
    "rho_mean":float(rho_values.mean()),"rho_min":float(rho_values.min()),"rho_max":float(rho_values.max()),
    "actor_loss":float(actor_loss.detach()),"critic_loss":float(critic_loss.detach()),
    "actor_grad":actor_grad,"critic_grad":critic_grad})
  if not rows:return result
  module.replay_phase_count+=1;total_states=sum(type_states.values())
  mean=lambda key:float(np.mean([row[key] for row in rows]))
  mean_transition_age=float(sum(row["mean_transition_age"]*row["valid_states"] for row in rows)/sum(row["valid_states"] for row in rows))
  max_transition_age=max(row["max_transition_age"] for row in rows)
  result.update(module.memory_counts());result.update({
   "pwtr_replay_batches":float(len(rows)),"pwtr_replay_valid_states":float(sum(row["valid_states"] for row in rows)),
   "pwtr_replay_alive_samples":float(sum(row["alive_samples"] for row in rows)),
   "pwtr_replay_w2_fraction":type_states[W2_INTERNAL]/max(total_states,1),
   "pwtr_replay_w3_fraction":type_states[W3_INTERNAL]/max(total_states,1),
   "pwtr_replay_bridge12_fraction":type_states[BRIDGE_12]/max(total_states,1),
   "pwtr_replay_bridge23_fraction":type_states[BRIDGE_23]/max(total_states,1),
   "pwtr_mean_segment_age":mean_transition_age,"pwtr_max_segment_age":max_transition_age,
   "pwtr_mean_transition_age":mean_transition_age,"pwtr_max_transition_age":max_transition_age,
   "pwtr_mean_learning_potential":mean("learning"),"pwtr_mean_freshness":mean("freshness"),
   "pwtr_mean_priority":mean("priority"),"pwtr_mean_joint_abs_log_ratio":mean("joint_abs_log_ratio"),
   "pwtr_mean_normalized_ess":mean("normalized_ess"),"pwtr_actor_eligible_fraction":mean("actor_eligible_fraction"),
   "pwtr_vtrace_rho_mean":mean("rho_mean"),"pwtr_vtrace_rho_min":min(row["rho_min"] for row in rows),
   "pwtr_vtrace_rho_max":max(row["rho_max"] for row in rows),"pwtr_actor_replay_loss":mean("actor_loss"),
   "pwtr_critic_replay_loss":mean("critic_loss"),"pwtr_actor_replay_grad_norm_preclip":mean("actor_grad"),
   "pwtr_critic_replay_grad_norm_preclip":mean("critic_grad"),
   "pwtr_replay_actor_optimizer_steps":float(module.replay_actor_optimizer_steps),
   "pwtr_replay_critic_optimizer_steps":float(module.replay_critic_optimizer_steps),
   "pwtr_replay_actor_optimizer_steps_this_phase":float(module.replay_actor_optimizer_steps-actor_steps_before),
   "pwtr_replay_critic_optimizer_steps_this_phase":float(module.replay_critic_optimizer_steps-critic_steps_before),
   "pwtr_replay_phase_count":float(module.replay_phase_count),
   "pwtr_replay_budget_fill_fraction":replay_budget_fill_fraction(len(rows),budget)})
  return result

 @torch.no_grad()
 def _dawe_diagnostics(self,r,obs,raw,alive,ctx,waves):
  if "deployment_aligned_wave_exploration" not in self.modules_config:return {}
  module=self.deployment_aligned_wave_exploration
  result={"dawe_enabled":float(module.enabled),
   "dawe_wave1_multiplier":module.multiplier(1),"dawe_wave2_multiplier":module.multiplier(2),
   "dawe_wave3_multiplier":module.multiplier(3)}
  base_scales=[];base_locs=[]
  for t in range(obs.shape[0]):
   hidden=None if r.actor_hidden_before_step is None else torch.as_tensor(r.actor_hidden_before_step[t],dtype=torch.float32,device=self.device)
   ep=None if r.episode_masks is None else torch.as_tensor(r.episode_masks[t],dtype=torch.float32,device=self.device)
   dist,_=self.actor.distribution_step(obs[t],self._ctx(ctx[t],True),hidden,ep,alive[t])
   base_scales.append(dist.scale);base_locs.append(dist.loc)
  scale=torch.stack(base_scales);loc=torch.stack(base_locs)
  for wave in (1,2,3):
   selected=(waves==wave).unsqueeze(-1)&(alive>.5)
   count=int(selected.sum())
   prefix=f"dawe_wave{wave}_"
   result[prefix+"alive_sample_count"]=float(count)
   if count==0:
    for name in ("base_std_mean","effective_std_mean","base_log_std_mean",
                 "effective_to_base_std_ratio","latent_deviation_abs_mean"):result[prefix+name]=None
    continue
   live_scale=scale[selected];multiplier=module.multiplier(wave) if module.enabled else 1.0
   result.update({prefix+"base_std_mean":float(live_scale.mean()),
    prefix+"effective_std_mean":float((live_scale*multiplier).mean()),
    prefix+"base_log_std_mean":float(live_scale.log().mean()),
    prefix+"effective_to_base_std_ratio":float(multiplier),
    prefix+"latent_deviation_abs_mean":float((raw-loc)[selected].abs().mean())})
  return result

 @torch.no_grad()
 def _reference_variance_diagnostics(self,obs,alive,waves):
  if not self.reference_variance.enabled:return {"rv_enabled":0.0}
  if self.reference_variance_actor is None:raise RuntimeError("RV reference actor missing")
  flat_obs=obs.reshape(-1,*obs.shape[2:]);flat_alive=alive.reshape(-1,*alive.shape[2:]);flat_waves=waves.reshape(-1)
  current,_=self.actor.distribution_step(flat_obs,None,None,None,flat_alive)
  reference,_=self.reference_variance_actor.distribution_step(flat_obs,None,None,None,flat_alive)
  optimizer_parameters={id(parameter) for group in self.actor_optimizer.param_groups for parameter in group["params"]}
  log_std_parameters=list(self.actor.log_std.parameters())
  result={"rv_enabled":1.0,"rv_reference_actor_sha256":self.reference_variance_actor_sha256(),
   "rv_log_std_head_requires_grad":float(any(parameter.requires_grad for parameter in log_std_parameters)),
   "rv_log_std_head_optimizer_membership":float(all(id(parameter) in optimizer_parameters for parameter in log_std_parameters)),
   "rv_log_std_head_grad_norm":self._gradient_norm(log_std_parameters),
   "rv_reference_any_grad_present":float(any(parameter.grad is not None for parameter in self.reference_variance_actor.parameters())),
   "rv_reference_mutation_detected":float(self.reference_variance_actor_sha256()!=self.reference_variance_initial_sha256)}
  for wave in (1,2,3):
   selected=(flat_waves==wave).unsqueeze(-1)&(flat_alive>.5);count=int(selected.sum());prefix=f"rv_wave{wave}_"
   result[prefix+"alive_sample_count"]=float(count)
   if not count:
    for key in ("reference_behavior_std_mean","current_unused_std_mean","current_unused_over_reference_std_ratio",
                "behavior_vs_reference_std_max_abs_error","reference_log_std_mean","current_unused_log_std_mean"):
     result[prefix+key]=None
    continue
   ref=reference.scale[selected];cur=current.scale[selected]
   behavior=self._behavior_actor_distribution(current,flat_obs,flat_alive,flat_waves).scale[selected]
   result.update({prefix+"reference_behavior_std_mean":float(ref.mean()),
    prefix+"current_unused_std_mean":float(cur.mean()),
    prefix+"current_unused_over_reference_std_ratio":float((cur/ref).mean()),
    prefix+"behavior_vs_reference_std_max_abs_error":float((behavior-ref).abs().max()),
    prefix+"reference_log_std_mean":float(ref.log().mean()),
    prefix+"current_unused_log_std_mean":float(cur.log().mean())})
  return result

 def _loss_step(self,obs,act,raw,oldlog,mask,adv,oldvalue,target,weights,ctx,ah=None,ch=None,ep=None,actor_weights=None,options=None,wave_indices=None):
  base_dist,newah=self.actor.distribution_step(obs,self._ctx(ctx,True),ah,ep,mask,option_ids=options)
  dist=self._behavior_actor_distribution(base_dist,obs,mask,wave_indices)
  newlog=self.actor._squashed_log_prob(dist,raw,act);sample_raw=dist.rsample();entropy=-self.actor._squashed_log_prob(dist,sample_raw,torch.tanh(sample_raw))
  logratio,ratio=stable_ratio_terms(newlog,oldlog);sur=torch.minimum(ratio*adv,ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*adv)
  weights=weights.unsqueeze(-1) if weights.ndim==mask.ndim-1 else weights
  aw=actor_weights if actor_weights is not None else (weights if self.wave_balance.actor_enabled else torch.ones_like(weights))
  aw=aw.unsqueeze(-1) if aw.ndim==mask.ndim-1 else aw
  actor_loss=-masked_mean(sur*aw,mask)
  critic_context=(torch.nn.functional.one_hot(options.long(),self.hierarchical_temporal_abstraction.num_options).to(obs.dtype) if self.hierarchical_temporal_abstraction.enabled else self._ctx(ctx,False))
  value,newch=self.critic.forward_step(obs,mask,critic_context,ch,ep)
  clipped=oldvalue+(value-oldvalue).clamp(-self.clip_ratio,self.clip_ratio);err=torch.maximum((value-target).square(),(clipped-target).square()) if self.clip_value_loss else (value-target).square()
  vw=weights if self.wave_balance.critic_enabled else torch.ones_like(weights);value_loss=.5*masked_mean(err*vw,mask);ent=masked_mean(entropy,mask)
  anchor_loss=torch.zeros((),device=self.device);akl=0.
  if self.anchor.enabled and self.anchor.reference_actor is not None:
   with torch.no_grad():ref=self.anchor.reference_actor.distribution(obs)
   anchor_loss,am=self.anchor.loss(dist,ref,self.sampled_steps,mask);akl=am["anchor_kl"]
  return actor_loss,value_loss,ent,anchor_loss,ratio,logratio,newlog,oldlog,newah,newch,akl
 @staticmethod
 def _gradient_norm(parameters):
  total=torch.zeros((),dtype=torch.float64)
  for parameter in parameters:
   if parameter.grad is not None:total+=parameter.grad.detach().double().square().sum().cpu()
  return float(total.sqrt())
 def _opt(self,losses):
  al,vl,en,anchor,*_=losses
  self.actor_optimizer.zero_grad();(al-self.entropy_coefficient*en+anchor).backward();arg=self._gradient_norm(self.actor.gru.parameters()) if self.recurrent.actor_enabled else 0.;actor_parameters=self.actor.trainable_policy_parameters();actor_limit=self._actor_grad_clip_limit();ag=nn.utils.clip_grad_norm_(actor_parameters,actor_limit)
  actor_post=self._gradient_norm(actor_parameters) if self.actor_gradient_clipping.enabled else 0.;self.actor_optimizer.step()
  self.critic_optimizer.zero_grad();(self.value_loss_coefficient*vl).backward();crg=self._gradient_norm(self.critic.gru.parameters()) if self.recurrent.critic_enabled else 0.;critic_parameters=list(self.critic.parameters());cg=nn.utils.clip_grad_norm_(critic_parameters,self.max_grad_norm)
  critic_post=self._gradient_norm(critic_parameters) if self.actor_gradient_clipping.enabled else 0.;self.critic_optimizer.step();self.actor_update_count+=1;self.critic_update_count+=1
  if self.actor_gradient_clipping.enabled:
   actor_pre=float(ag);critic_pre=float(cg);self.actor_gradient_clipping.record_step(actor_pre)
   self._last_actor_grad_clip_metrics={"actor_grad_clip_enabled":1.,"actor_grad_clip_limit":actor_limit,
    "actor_grad_clip_preclip_norm":actor_pre,"actor_grad_clip_postclip_norm":actor_post,
    "actor_grad_clip_exact_scale":actor_post/max(actor_pre,1e-30),"actor_grad_clip_pressure_fraction":float(actor_pre>actor_limit),
    "actor_grad_clip_critic_limit":self.max_grad_norm,"actor_grad_clip_critic_preclip_norm":critic_pre,
    "actor_grad_clip_critic_postclip_norm":critic_post,"actor_grad_clip_critic_exact_scale":critic_post/max(critic_pre,1e-30),
    "actor_grad_clip_critic_pressure_fraction":float(critic_pre>self.max_grad_norm)}
  return ag,cg,arg,crg
 def _row(self,losses,mask,ag,cg,arg=0.,crg=0.):
  al,vl,en,anchor,ratio,logratio,newlog,oldlog,_,_,akl=losses
  stable_ratio_terms(newlog,oldlog)
  live_mask=mask>.5;live=ratio[live_mask];live_logratio=logratio[live_mask]
  if live.numel()==0:raise FloatingPointError("no valid alive PPO samples")
  kl=(ratio-1)-logratio
  row={"actor_loss":float(al.detach()),"weighted_actor_loss":float(al.detach()),"value_loss":float(vl.detach()),"weighted_value_loss":float(vl.detach()),"entropy":float(en.detach()),"approx_kl":float(masked_mean(kl,mask).detach()),"clip_fraction":float(masked_mean((ratio.sub(1).abs()>self.clip_ratio).float(),mask).detach()),"actor_grad_norm":float(ag),"critic_grad_norm":float(cg),"actor_gru_grad_norm":float(arg),"critic_gru_grad_norm":float(crg),"gru_gradient_norm":float(np.hypot(arg,crg)),"anchor_kl":float(akl),"anchor_loss":float(anchor.detach()),"anchor_effective_coefficient":float(self.anchor.effective_coefficient(self.sampled_steps)),"_valid_count":int(live.numel()),"_ratio_values":live.detach().cpu().numpy(),"_log_ratio_values":live_logratio.detach().cpu().numpy()}
  if self.actor_gradient_clipping.enabled:row.update(self._last_actor_grad_clip_metrics)
  return row
 def _update_flat_marc(self,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,waves,marc_weights):
  """MARC actor objective; critic and entropy retain Plain MAPPO semantics."""
  if marc_weights is None:raise RuntimeError("MARC actor weights are missing")
  flat=lambda x:x.reshape(obs.shape[0]*obs.shape[1],*x.shape[2:])
  arrays=list(map(flat,(obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,waves,marc_weights)))
  N=arrays[0].shape[0];rows=[];actor_before=self.actor_update_count;critic_before=self.critic_update_count
  minibatches_per_epoch=(N+self.minibatch_size-1)//self.minibatch_size
  expected=self.ppo_epochs*minibatches_per_epoch
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device)
    o,a,r,ol,m,ad,ov,t,w,c,wave,mw=[x[ix] for x in arrays]
    losses=self._loss_step(o,a,r,ol,m,ad,ov,t,w,c,actor_weights=mw)
    actor_loss,value_loss,entropy=losses[:3]
    retention_loss,retention_metrics=self.milestone_aware_retention_credit.retention_loss(
     self.actor,self.device)
    self.actor_optimizer.zero_grad()
    (actor_loss+retention_loss-self.entropy_coefficient*entropy).backward()
    actor_parameters=self.actor.trainable_policy_parameters()
    ag=nn.utils.clip_grad_norm_(actor_parameters,self.max_grad_norm)
    self.actor_optimizer.step();self.actor_update_count+=1
    self.critic_optimizer.zero_grad()
    (self.value_loss_coefficient*value_loss).backward()
    cg=nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm)
    self.critic_optimizer.step();self.critic_update_count+=1
    row=self._row(losses,m,ag,cg);row.update(retention_metrics);rows.append(row)
  actor_steps=self.actor_update_count-actor_before;critic_steps=self.critic_update_count-critic_before
  if len(rows)!=expected or actor_steps!=expected or critic_steps!=expected:
   raise RuntimeError("MARC flat PPO epoch execution mismatch")
  result=aggregate_update_rows(rows,self.clip_ratio)
  result.update({"ppo_epochs_configured":float(self.ppo_epochs),
   "ppo_epochs_executed":float(len(rows)//minibatches_per_epoch),
   "ppo_minibatches_per_epoch":float(minibatches_per_epoch),
   "ppo_minibatches_executed":float(len(rows)),
   "actor_optimizer_steps_this_update":float(actor_steps),
   "critic_optimizer_steps_this_update":float(critic_steps)})
  return result

 def _update_flat(self,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,actor_weights=None,fresh_waves=None,wave_indices=None):
  flat=lambda x:x.reshape(obs.shape[0]*obs.shape[1],*x.shape[2:]); arrays=list(map(flat,(obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx)));flat_actor=None if actor_weights is None else flat(actor_weights);N=arrays[0].shape[0];rows=[]
  flat_waves=None if wave_indices is None else wave_indices.reshape(-1)
  actor_updates_before=self.actor_update_count;critic_updates_before=self.critic_update_count
  minibatches_per_epoch=(N+self.minibatch_size-1)//self.minibatch_size
  expected_total_minibatches=self.ppo_epochs*minibatches_per_epoch
  for _ in range(self.ppo_epochs):
   permutation=(self.persistent_wave_trajectory_replay.fresh_epoch_permutation(
    fresh_waves.detach().cpu().numpy().reshape(-1),self.minibatch_size,self.rng)
    if fresh_waves is not None else self.rng.permutation(N))
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device); args=[x[ix] for x in arrays];loss=self._loss_step(*args,actor_weights=None if flat_actor is None else flat_actor[ix],wave_indices=None if flat_waves is None else flat_waves[ix]);ag,cg,arg,crg=self._opt(loss);rows.append(self._row(loss,args[4],ag,cg,arg,crg))
  actor_steps=self.actor_update_count-actor_updates_before;critic_steps=self.critic_update_count-critic_updates_before
  if len(rows)!=expected_total_minibatches or actor_steps!=expected_total_minibatches or critic_steps!=expected_total_minibatches:
   raise RuntimeError("flat PPO epoch execution mismatch: "
    f"rows={len(rows)}, actor_steps={actor_steps}, critic_steps={critic_steps}, "
    f"expected={expected_total_minibatches}, epochs={self.ppo_epochs}, minibatches_per_epoch={minibatches_per_epoch}")
  result=aggregate_update_rows(rows,self.clip_ratio)
  executed_epochs=(len(rows)//minibatches_per_epoch if minibatches_per_epoch else 0)
  result.update({"ppo_epochs_configured":float(self.ppo_epochs),"ppo_epochs_executed":float(executed_epochs),
   "ppo_minibatches_per_epoch":float(minibatches_per_epoch),"ppo_minibatches_executed":float(len(rows)),
   "actor_optimizer_steps_this_update":float(actor_steps),"critic_optimizer_steps_this_update":float(critic_steps)})
  return result

 def _update_flat_wsai(self,obs,act,raw,oldlog,alive,adv,oldvalue,target,ctx,waves):
  """Plain PPO objective with environment-wave routing to isolated actors."""
  flat=lambda x:x.reshape(obs.shape[0]*obs.shape[1],*x.shape[2:])
  O,A,R,OL,M,AD,OV,TG,C,W=map(flat,(obs,act,raw,oldlog,alive,adv,oldvalue,target,ctx,waves))
  N=O.shape[0];rows=[];actors=self._wsai_actors();optimizers=self._wsai_actor_optimizers()
  optimizer_steps_before=list(self.wave_specific_actor_isolation.optimizer_steps)
  all_parameters=[parameter for actor in actors for parameter in actor.trainable_policy_parameters()]
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device)
    o,a,r,ol,m,ad,ov,t,c,w=[x[ix] for x in (O,A,R,OL,M,AD,OV,TG,C,W)]
    dist=self._routed_actor_distribution(o,m,w);newlog=self.actor._squashed_log_prob(dist,r,a)
    sample_raw=dist.rsample();entropy_values=-self.actor._squashed_log_prob(dist,sample_raw,torch.tanh(sample_raw))
    logratio,ratio=stable_ratio_terms(newlog,ol);surrogate=torch.minimum(ratio*ad,ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*ad)
    actor_loss=-masked_mean(surrogate,m);entropy=masked_mean(entropy_values,m)
    value,_=self.critic.forward_step(o,m,None,None,None);clipped=ov+(value-ov).clamp(-self.clip_ratio,self.clip_ratio)
    error=torch.maximum((value-t).square(),(clipped-t).square()) if self.clip_value_loss else (value-t).square()
    value_loss=.5*masked_mean(error,m);zero=torch.zeros((),device=self.device)
    losses=(actor_loss,value_loss,entropy,zero,ratio,logratio,newlog,ol,None,None,0.)
    for optimizer in optimizers:optimizer.zero_grad()
    (actor_loss-self.entropy_coefficient*entropy).backward()
    per_pre=[self._gradient_norm(actor.trainable_policy_parameters()) for actor in actors]
    global_pre=self._gradient_norm(all_parameters);ag=nn.utils.clip_grad_norm_(all_parameters,self.max_grad_norm)
    global_post=self._gradient_norm(all_parameters);per_post=[self._gradient_norm(actor.trainable_policy_parameters()) for actor in actors]
    counts=[int((m*(w==wave).to(m.dtype).unsqueeze(-1)).sum().item()) for wave in (1,2,3)]
    stepped=[]
    for count,optimizer in zip(counts,optimizers):
     stepped.append(count>0)
     if count>0:optimizer.step()
    self.wave_specific_actor_isolation.record_minibatch(counts,stepped);self.actor_update_count+=1
    self.critic_optimizer.zero_grad();(self.value_loss_coefficient*value_loss).backward()
    cg=nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm);self.critic_optimizer.step();self.critic_update_count+=1
    row=self._row(losses,m,ag,cg)
    total=max(sum(counts),1)
    row.update({"wsai_wave1_alive_fraction":counts[0]/total,"wsai_wave2_alive_fraction":counts[1]/total,
     "wsai_wave3_alive_fraction":counts[2]/total,"wsai_global_actor_grad_norm_preclip":global_pre,
     "wsai_global_actor_grad_norm_postclip":global_post})
    for index in range(3):
     wave=index+1;row[f"wsai_wave{wave}_actor_grad_norm_preclip"]=per_pre[index]
     row[f"wsai_wave{wave}_actor_grad_norm_postclip"]=per_post[index]
     row[f"wsai_wave{wave}_optimizer_stepped"]=float(stepped[index])
     row[f"wsai_wave{wave}_actor_lr"]=float(optimizers[index].param_groups[0]["lr"])
    rows.append(row)
  out=aggregate_update_rows(rows,self.clip_ratio);out.update(self.wave_specific_actor_isolation.diagnostics())
  total_alive=float(M.sum().item())
  for wave in (1,2,3):out[f"wsai_wave{wave}_alive_fraction"]=float((M*(W==wave).to(M.dtype).unsqueeze(-1)).sum().item()/max(total_alive,1.))
  for index in range(3):out[f"wsai_wave{index+1}_actor_optimizer_steps_this_update"]=float(self.wave_specific_actor_isolation.optimizer_steps[index]-optimizer_steps_before[index])
  out.update(pairwise_actor_l2_distances(actors));return out

 def _update_flat_wsmh(self,obs,act,raw,oldlog,alive,adv,oldvalue,target,ctx,waves):
  """Plain global PPO objective with only the Gaussian mean head routed by wave."""
  flat=lambda x:x.reshape(obs.shape[0]*obs.shape[1],*x.shape[2:])
  O,A,R,OL,M,AD,OV,TG,C,W=map(flat,(obs,act,raw,oldlog,alive,adv,oldvalue,target,ctx,waves))
  N=O.shape[0];rows=[];heads=self._wsmh_means()
  mean_steps_before=list(self.wave_specific_mean_heads.mean_optimizer_steps)
  shared_steps_before=self.wave_specific_mean_heads.shared_actor_optimizer_steps
  backbone_parameters=list(self.actor.backbone.parameters());logstd_parameters=list(self.actor.log_std.parameters())
  head_parameters=[list(head.parameters()) for head in heads]
  all_parameters=backbone_parameters+logstd_parameters+sum(head_parameters,[])
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device)
    o,a,r,ol,m,ad,ov,t,c,w=[x[ix] for x in (O,A,R,OL,M,AD,OV,TG,C,W)]
    dist=self._wsmh_routed_distribution(o,m,w);newlog=self.actor._squashed_log_prob(dist,r,a)
    sample_raw=dist.rsample();entropy_values=-self.actor._squashed_log_prob(dist,sample_raw,torch.tanh(sample_raw))
    logratio,ratio=stable_ratio_terms(newlog,ol);surrogate=torch.minimum(ratio*ad,ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*ad)
    actor_loss=-masked_mean(surrogate,m);entropy=masked_mean(entropy_values,m)
    value,_=self.critic.forward_step(o,m,None,None,None);clipped=ov+(value-ov).clamp(-self.clip_ratio,self.clip_ratio)
    error=torch.maximum((value-t).square(),(clipped-t).square()) if self.clip_value_loss else (value-t).square()
    value_loss=.5*masked_mean(error,m);zero=torch.zeros((),device=self.device)
    losses=(actor_loss,value_loss,entropy,zero,ratio,logratio,newlog,ol,None,None,0.)
    self.actor_optimizer.zero_grad();self.wave2_mean_optimizer.zero_grad();self.wave3_mean_optimizer.zero_grad()
    (actor_loss-self.entropy_coefficient*entropy).backward()
    counts=[int((m*(w==wave).to(m.dtype).unsqueeze(-1)).sum().item()) for wave in (1,2,3)]
    # None, not a zero tensor, is required to prevent Adam momentum drift.
    if counts[0]==0:
     for parameter in self.actor.mean.parameters():parameter.grad=None
    backbone_pre=self._gradient_norm(backbone_parameters);logstd_pre=self._gradient_norm(logstd_parameters)
    mean_pre=[self._gradient_norm(parameters) for parameters in head_parameters]
    global_pre=self._gradient_norm(all_parameters);clip_scale=min(1.,self.max_grad_norm/global_pre) if global_pre>0 else 1.
    ag=nn.utils.clip_grad_norm_(all_parameters,self.max_grad_norm)
    global_post=self._gradient_norm(all_parameters);backbone_post=self._gradient_norm(backbone_parameters);logstd_post=self._gradient_norm(logstd_parameters)
    mean_post=[self._gradient_norm(parameters) for parameters in head_parameters]
    self.actor_optimizer.step();stepped=[counts[0]>0,counts[1]>0,counts[2]>0]
    if counts[1]>0:self.wave2_mean_optimizer.step()
    if counts[2]>0:self.wave3_mean_optimizer.step()
    self.wave_specific_mean_heads.record_minibatch(counts,stepped);self.actor_update_count+=1
    self.critic_optimizer.zero_grad();(self.value_loss_coefficient*value_loss).backward()
    cg=nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm);self.critic_optimizer.step();self.critic_update_count+=1
    row=self._row(losses,m,ag,cg);total=max(sum(counts),1)
    row.update({"wsmh_wave1_alive_fraction":counts[0]/total,"wsmh_wave2_alive_fraction":counts[1]/total,
     "wsmh_wave3_alive_fraction":counts[2]/total,"wsmh_shared_backbone_grad_norm_preclip":backbone_pre,
     "wsmh_shared_logstd_grad_norm_preclip":logstd_pre,"wsmh_shared_backbone_grad_norm_postclip":backbone_post,
     "wsmh_shared_logstd_grad_norm_postclip":logstd_post,"wsmh_global_actor_grad_norm_preclip":global_pre,
     "wsmh_global_actor_grad_norm_postclip":global_post,"wsmh_global_clip_scale":clip_scale,
     "wsmh_shared_actor_lr":float(self.actor_optimizer.param_groups[0]["lr"]),
     "wsmh_wave1_mean_lr":float(self.actor_optimizer.param_groups[0]["lr"]),
     "wsmh_wave2_mean_lr":float(self.wave2_mean_optimizer.param_groups[0]["lr"]),
     "wsmh_wave3_mean_lr":float(self.wave3_mean_optimizer.param_groups[0]["lr"])})
    for index in range(3):
     row[f"wsmh_wave{index+1}_mean_grad_norm_preclip"]=mean_pre[index]
     row[f"wsmh_wave{index+1}_mean_grad_norm_postclip"]=mean_post[index]
    rows.append(row)
  out=aggregate_update_rows(rows,self.clip_ratio);out.update(self.wave_specific_mean_heads.diagnostics())
  total_alive=float(M.sum().item())
  for wave in (1,2,3):out[f"wsmh_wave{wave}_alive_fraction"]=float((M*(W==wave).to(M.dtype).unsqueeze(-1)).sum().item()/max(total_alive,1.))
  for index in range(3):out[f"wsmh_wave{index+1}_mean_optimizer_steps_this_update"]=float(self.wave_specific_mean_heads.mean_optimizer_steps[index]-mean_steps_before[index])
  out["wsmh_shared_actor_optimizer_steps_this_update"]=float(self.wave_specific_mean_heads.shared_actor_optimizer_steps-shared_steps_before)
  out.update(pairwise_mean_l2_distances(heads));return out

 def _update_flat_swgp(self,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,waves):
  """Flat Plain PPO with ordered projection applied only to wave surrogate gradients."""
  flat=lambda x:x.reshape(obs.shape[0]*obs.shape[1],*x.shape[2:])
  arrays=list(map(flat,(obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,waves)));N=arrays[0].shape[0];rows=[]
  params=list(self.actor.trainable_policy_parameters());module=self.sequential_wave_gradient_projection
  zeros=lambda:[torch.zeros_like(parameter) for parameter in params]
  grads=lambda loss:[torch.zeros_like(parameter) if value is None else value for parameter,value in zip(params,torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True))]
  as_float=lambda value:float(value.detach()) if torch.is_tensor(value) else float(value)
  safe_cos=lambda left,right:as_float(gradient_cosine(left,right,module.epsilon)) if as_float(gradient_norm(left))>module.epsilon and as_float(gradient_norm(right))>module.epsilon else 0.
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device);o,a,r,ol,m,ad,ov,t,w,c,wave=[x[ix] for x in arrays]
    dist,_=self.actor.distribution_step(o,None,None,None,m);newlog=self.actor._squashed_log_prob(dist,r,a);sample_raw=dist.rsample();entropy_values=-self.actor._squashed_log_prob(dist,sample_raw,torch.tanh(sample_raw))
    logratio,ratio=stable_ratio_terms(newlog,ol);surrogate=torch.minimum(ratio*ad,ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*ad)
    actor_loss=-masked_mean(surrogate,m);entropy=masked_mean(entropy_values,m)
    value,_=self.critic.forward_step(o,m,None,None,None);clipped=ov+(value-ov).clamp(-self.clip_ratio,self.clip_ratio)
    error=torch.maximum((value-t).square(),(clipped-t).square()) if self.clip_value_loss else (value-t).square();value_loss=.5*masked_mean(error,m)
    counts={k:int((m*(wave==k).to(m.dtype).unsqueeze(-1)).sum().item()) for k in (1,2,3)};alpha=natural_wave_fractions(counts)
    wave_gradients={};wave_losses={}
    for k in (1,2,3):
     if counts[k]:
      mask=m*(wave==k).to(m.dtype).unsqueeze(-1);wave_losses[k]=-masked_mean(surrogate,mask);wave_gradients[k]=grads(wave_losses[k])
    plain_surrogate=grads(actor_loss);projected,diagnostics=ordered_upstream_pairwise(wave_gradients,module.epsilon)
    projected_surrogate=weighted_gradient_sum(projected,alpha,params)
    entropy_gradient=grads(-self.entropy_coefficient*entropy)
    projection_applied=any(diagnostics[key] for key in ("w2_applied","w3_w1_applied","w3_w2_applied"))
    # Preserve the exact Plain autograd expression whenever projection is inactive.
    combined=([left+right for left,right in zip(projected_surrogate,entropy_gradient)] if projection_applied
              else grads(actor_loss-self.entropy_coefficient*entropy))
    for current,gradient in zip(params,combined):current.grad=gradient.detach().clone()
    pre_clip=as_float(gradient_norm(combined));ag=nn.utils.clip_grad_norm_(params,self.max_grad_norm);post_clip=self._gradient_norm(params)
    self.actor_optimizer.step();self.actor_update_count+=1
    self.critic_optimizer.zero_grad();(self.value_loss_coefficient*value_loss).backward();cg=nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm);self.critic_optimizer.step();self.critic_update_count+=1
    module.record(len(wave_gradients),diagnostics)
    def metric(name):
     value=diagnostics.get(name);return 0. if value is None else as_float(value)
    def pair_cos(first,second):return safe_cos(wave_gradients[first],wave_gradients[second]) if first in wave_gradients and second in wave_gradients else 0.
    w2_before=metric("w2_dot_before");w2_after=metric("w2_dot_after")
    if 1 in projected and 2 in projected:
     tolerance=-1e-7*as_float(gradient_norm(projected[1]))*as_float(gradient_norm(projected[2]))
     if w2_after<tolerance:raise FloatingPointError("SWGP W2 projection violated W1 protection")
    if 3 in projected:
     for reference in (1,2):
      if reference in projected:
       dot=as_float(gradient_dot(projected[3],projected[reference]));tolerance=-1e-7*as_float(gradient_norm(projected[3]))*as_float(gradient_norm(projected[reference]))
       if dot<tolerance:raise FloatingPointError(f"SWGP W3 projection violated W{reference} protection")
    losses=(actor_loss,value_loss,entropy,torch.zeros((),device=self.device),ratio,logratio,newlog,ol,None,None,0.)
    row=self._row(losses,m,ag,cg);row.update({
     "swgp_active":1.,"swgp_wave1_alive_fraction":alpha[1],"swgp_wave2_alive_fraction":alpha[2],"swgp_wave3_alive_fraction":alpha[3],
     "swgp_wave1_grad_norm":as_float(gradient_norm(wave_gradients.get(1,zeros()))),"swgp_wave2_grad_norm":as_float(gradient_norm(wave_gradients.get(2,zeros()))),"swgp_wave3_grad_norm":as_float(gradient_norm(wave_gradients.get(3,zeros()))),
     "swgp_raw_cos_w1_w2":pair_cos(1,2),"swgp_raw_cos_w1_w3":pair_cos(1,3),"swgp_raw_cos_w2_w3":pair_cos(2,3),
     "swgp_w2_projection_applied_fraction":float(diagnostics["w2_applied"]),"swgp_w2_dot_before":w2_before,"swgp_w2_dot_after":w2_after,
     "swgp_w2_cos_before":pair_cos(1,2),"swgp_w2_cos_after":safe_cos(projected[1],projected[2]) if 1 in projected and 2 in projected else 0.,
     "swgp_w3_w1_projection_applied_fraction":float(diagnostics["w3_w1_applied"]),"swgp_w3_w1_dot_before":metric("w3_w1_dot_before"),"swgp_w3_w1_dot_after":metric("w3_w1_dot_after"),
     "swgp_w3_w2_projection_applied_fraction":float(diagnostics["w3_w2_applied"]),"swgp_w3_w2_dot_before":metric("w3_w2_dot_before"),"swgp_w3_w2_dot_after":metric("w3_w2_dot_after"),
     "swgp_surrogate_grad_norm_plain":as_float(gradient_norm(plain_surrogate)),"swgp_surrogate_grad_norm_projected":as_float(gradient_norm(projected_surrogate)),
     "swgp_entropy_grad_norm":as_float(gradient_norm(entropy_gradient)),"swgp_actor_grad_norm_pre_clip":pre_clip,"swgp_actor_grad_norm_post_clip":post_clip,
     "swgp_no_projection_fraction":float(not projection_applied),"swgp_single_wave_minibatch_fraction":float(len(wave_gradients)==1)})
    rows.append(row)
  out=aggregate_update_rows(rows,self.clip_ratio);out.update({"swgp_total_minibatches":float(module.total_swgp_minibatches),"swgp_w2_projection_count":float(module.w2_projection_count),"swgp_w3_w1_projection_count":float(module.w3_w1_projection_count),"swgp_w3_w2_projection_count":float(module.w3_w2_projection_count),"swgp_no_projection_count":float(module.no_projection_count),"swgp_single_wave_minibatch_count":float(module.single_wave_minibatch_count)});return out

 def _update_flat_hta(self,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,options):
  flat=lambda x:x.reshape(obs.shape[0]*obs.shape[1],*x.shape[2:])
  arrays=list(map(flat,(obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,options)));N=arrays[0].shape[0];rows=[]
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device);args=[x[ix] for x in arrays]
    loss=self._loss_step(*args[:10],options=args[10]);ag,cg,arg,crg=self._opt(loss);rows.append(self._row(loss,args[4],ag,cg,arg,crg))
  return aggregate_update_rows(rows,self.clip_ratio)

 def _critic_loss_step(self,obs,mask,oldvalue,target,weights,ctx):
  value,_=self.critic.forward_step(obs,mask,self._ctx(ctx,False),None,None)
  clipped=oldvalue+(value-oldvalue).clamp(-self.clip_ratio,self.clip_ratio)
  err=torch.maximum((value-target).square(),(clipped-target).square()) if self.clip_value_loss else (value-target).square()
  weights=weights.unsqueeze(-1) if weights.ndim==mask.ndim-1 else weights
  vw=weights if self.wave_balance.critic_enabled else torch.ones_like(weights)
  return .5*masked_mean(err*vw,mask)

 def _update_flat_actor_kl_guard(self,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,actor_weights=None):
  """Preserve the flat PPO update order until the actor epoch guard fires."""
  flat=lambda x:x.reshape(obs.shape[0]*obs.shape[1],*x.shape[2:])
  arrays=list(map(flat,(obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx)))
  flat_actor=None if actor_weights is None else flat(actor_weights);N=arrays[0].shape[0]
  rows=[];critic_losses=[];critic_grad_norms=[];epoch_kls=[];actor_epochs=0;hard_stop=False;actor_active=True
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device);args=[x[ix] for x in arrays]
    if actor_active:
     loss=self._loss_step(*args,actor_weights=None if flat_actor is None else flat_actor[ix]);ag,cg,arg,crg=self._opt(loss)
     rows.append(self._row(loss,args[4],ag,cg,arg,crg));critic_losses.append(float(loss[1].detach()));critic_grad_norms.append(float(cg))
    else:
     value_loss=self._critic_loss_step(args[0],args[4],args[6],args[7],args[8],args[9])
     self.critic_optimizer.zero_grad();(self.value_loss_coefficient*value_loss).backward();cg=nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm);self.critic_optimizer.step();self.critic_update_count+=1
     critic_losses.append(float(value_loss.detach()));critic_grad_norms.append(float(cg))
   if actor_active:
    actor_epochs+=1;epoch_kl=self._full_rollout_kl(arrays[0],arrays[2],arrays[3],arrays[4],arrays[9]);epoch_kls.append(epoch_kl)
    if self.actor_kl_guard.should_stop_actor(epoch_kl):
     hard_stop=True;actor_active=False;self.kl_hard_stop_count+=1;self.actor_kl_guard_hard_stop_count+=1
  self.actor_kl_guard_actor_epochs_total+=actor_epochs
  self.actor_kl_guard_actor_epochs_min=(actor_epochs if self.actor_kl_guard_actor_epochs_min is None
                                       else min(self.actor_kl_guard_actor_epochs_min,actor_epochs))
  out=aggregate_update_rows(rows,self.clip_ratio)
  if hard_stop:
   out.update({"value_loss":float(np.mean(critic_losses)),"weighted_value_loss":float(np.mean(critic_losses)),"critic_grad_norm":float(np.mean(critic_grad_norms))})
  out.update({"actor_epochs_planned":float(self.ppo_epochs),"actor_epochs_used":float(actor_epochs),"critic_epochs_used":float(self.ppo_epochs),"epoch_kl_last":float(epoch_kls[-1]),"epoch_kl_max":float(max(epoch_kls)),"kl_hard_stop_triggered":float(hard_stop)})
  return out

 def _update_flat_stabilized(self,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,actor_weights):
  """Split actor/critic epochs so a hard actor KL stop never truncates critic work."""
  flat=lambda x:x.reshape(obs.shape[0]*obs.shape[1],*x.shape[2:])
  arrays=list(map(flat,(obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,actor_weights)));N=arrays[0].shape[0]
  actor_rows=[];critic_losses=[];critic_grad_norms=[];epoch_kls=[];actor_epochs=0;hard_stop=False
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device);args=[x[ix] for x in arrays[:10]]
    loss=self._loss_step(*args,actor_weights=arrays[10][ix]);al,_,en,anchor,*_=loss
    self.actor_optimizer.zero_grad();(al-self.entropy_coefficient*en+anchor).backward();ag=nn.utils.clip_grad_norm_(self.actor.trainable_policy_parameters(),self.max_grad_norm);self.actor_optimizer.step();self.actor_update_count+=1
    actor_rows.append(self._row(loss,args[4],ag,0.))
   actor_epochs+=1;epoch_kl=self._full_rollout_kl(arrays[0],arrays[2],arrays[3],arrays[4],arrays[9]);epoch_kls.append(epoch_kl)
   stop_actor=(self.ppo_stabilization.should_stop_actor(epoch_kl) if self.ppo_stabilization.enabled
               else self.actor_kl_guard.should_stop_actor(epoch_kl))
   if stop_actor:
    hard_stop=True;self.kl_hard_stop_count+=1
    if self.actor_kl_guard.enabled:self.actor_kl_guard_hard_stop_count+=1
    break
  if self.actor_kl_guard.enabled:
   self.actor_kl_guard_actor_epochs_total+=actor_epochs
   self.actor_kl_guard_actor_epochs_min=(actor_epochs if self.actor_kl_guard_actor_epochs_min is None
                                        else min(self.actor_kl_guard_actor_epochs_min,actor_epochs))
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device);args=[x[ix] for x in arrays[:10]]
    loss=self._loss_step(*args,actor_weights=arrays[10][ix]);vl=loss[1]
    self.critic_optimizer.zero_grad();(self.value_loss_coefficient*vl).backward();cg=nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm);self.critic_optimizer.step();self.critic_update_count+=1
    critic_losses.append(float(vl.detach()));critic_grad_norms.append(float(cg))
  out=aggregate_update_rows(actor_rows,self.clip_ratio)
  out.update({"value_loss":float(np.mean(critic_losses)),"weighted_value_loss":float(np.mean(critic_losses)),"critic_grad_norm":float(np.mean(critic_grad_norms)),"actor_epochs_planned":float(self.ppo_epochs),"actor_epochs_used":float(actor_epochs),"critic_epochs_used":float(self.ppo_epochs),"epoch_kl_last":float(epoch_kls[-1]),"epoch_kl_max":float(max(epoch_kls)),"kl_hard_stop_triggered":float(hard_stop)})
  return out

 @torch.no_grad()
 def _full_rollout_kl(self,obs,raw,oldlog,alive,ctx):
  values=[];batch=max(self.minibatch_size,1)
  for start in range(0,obs.shape[0],batch):
   dist,_=self.actor.distribution_step(obs[start:start+batch],self._ctx(ctx[start:start+batch],True),None,None,alive[start:start+batch]);newlog=self.actor._squashed_log_prob(dist,raw[start:start+batch],torch.tanh(raw[start:start+batch]));logratio,ratio=stable_ratio_terms(newlog,oldlog[start:start+batch]);mask=alive[start:start+batch]> .5;values.append(((ratio-1)-logratio)[mask])
  return float(torch.cat(values).double().mean())
 def _update_recurrent(self,r,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx):
  tt=lambda x:torch.as_tensor(x,dtype=torch.float32,device=self.device)
  if self.development_method=="jiao2025_core_3m":
   if r.episode_masks is None:raise RuntimeError("Jiao-Core requires true episode masks for recurrent chunks")
   if r.actor_hidden_before_step is None or r.critic_hidden_before_step is None:raise RuntimeError("Jiao-Core requires saved pre-action Actor and Critic hidden")
   chunks=episode_contiguous_chunks(r.episode_masks,self.recurrent.sequence_length)
  else:chunks=contiguous_chunks(obs.shape[0],obs.shape[1],self.recurrent.sequence_length)
  rows=[];actor_before=self.actor_update_count;critic_before=self.critic_update_count
  sequences_per_minibatch=max(1,self.minibatch_size//self.recurrent.sequence_length)
  for _ in range(self.ppo_epochs):
   order=self.rng.permutation(len(chunks))
   for start in range(0,len(chunks),sequences_per_minibatch):
    group=[chunks[int(i)] for i in order[start:start+sequences_per_minibatch]]
    rows.append(self._recurrent_minibatch(r,group,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,tt))
  minibatches=int(np.ceil(len(chunks)/sequences_per_minibatch));expected=self.ppo_epochs*minibatches
  if len(rows)!=expected or self.actor_update_count-actor_before!=expected or self.critic_update_count-critic_before!=expected:
   raise RuntimeError("joint recurrent PPO epoch/optimizer execution mismatch")
  out=aggregate_update_rows(rows,self.clip_ratio);out.update({"sequence_chunks":float(len(chunks)),"sequences_per_minibatch":float(sequences_per_minibatch),"recurrent_minibatches_per_epoch":float(minibatches),
   "ppo_epochs_configured":float(self.ppo_epochs),"ppo_epochs_executed":float(len(rows)//minibatches),
   "actor_optimizer_steps_this_update":float(self.actor_update_count-actor_before),"critic_optimizer_steps_this_update":float(self.critic_update_count-critic_before)});return out

 def _update_actor_recurrent_critic_flat(self,r,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,marc_weights=None):
  """Train an actor-only GRU with BPTT while retaining Plain flat critic updates."""
  tt=lambda x:torch.as_tensor(x,dtype=torch.float32,device=self.device)
  if self.recurrent.wave_boundary_reset:
   if r.actor_recurrent_phase_reset_flags is None:raise RuntimeError("missing actor_recurrent_phase_reset_flags")
   chunks=wave_segmented_chunks(r.wave_indices,r.actor_recurrent_phase_reset_flags,r.episode_masks,self.recurrent.sequence_length)
  else:chunks=contiguous_chunks(obs.shape[0],obs.shape[1],self.recurrent.sequence_length)
  actor_before,critic_before=self.actor_update_count,self.critic_update_count
  sequences_per_minibatch=max(1,self.minibatch_size//self.recurrent.sequence_length)
  actor_rows=[]
  # Actor shuffling is deliberately ephemeral: only the flat critic consumes
  # the trainer RNG, exactly as it does in the Plain MAPPO update.
  actor_rng=np.random.default_rng()
  actor_rng.bit_generator.state=deepcopy(self.rng.bit_generator.state)
  for _ in range(self.ppo_epochs):
   order=actor_rng.permutation(len(chunks))
   for start in range(0,len(chunks),sequences_per_minibatch):
    group=[chunks[int(i)] for i in order[start:start+sequences_per_minibatch]]
    actor_rows.append(self._actor_recurrent_minibatch(r,group,obs,act,raw,oldlog,alive,adv,marc_weights if marc_weights is not None else weights,ctx,tt))

  flat=lambda x:x.reshape(obs.shape[0]*obs.shape[1],*x.shape[2:])
  O,M,OV,TG,W,C=map(flat,(obs,alive,oldvalue,target,weights,ctx));N=O.shape[0]
  critic_losses=[];critic_grad_norms=[]
  for _ in range(self.ppo_epochs):
   permutation=self.rng.permutation(N)
   for start in range(0,N,self.minibatch_size):
    ix=torch.as_tensor(permutation[start:start+self.minibatch_size],device=self.device)
    value_loss=self._critic_loss_step(O[ix],M[ix],OV[ix],TG[ix],W[ix],C[ix])
    self.critic_optimizer.zero_grad();(self.value_loss_coefficient*value_loss).backward()
    cg=nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm)
    self.critic_optimizer.step();self.critic_update_count+=1
    critic_losses.append(float(value_loss.detach()));critic_grad_norms.append(float(cg))
  out=aggregate_update_rows(actor_rows,self.clip_ratio)
  out.update({"value_loss":float(np.mean(critic_losses)),"weighted_value_loss":float(np.mean(critic_losses)),
   "critic_grad_norm":float(np.mean(critic_grad_norms)),"critic_gru_grad_norm":0.0,
   "sequence_chunks":float(len(chunks)),"sequences_per_minibatch":float(sequences_per_minibatch),
   "recurrent_minibatches_per_epoch":float(np.ceil(len(chunks)/sequences_per_minibatch))})
  if self.recurrent.wave_boundary_reset:
   actor_expected=self.ppo_epochs*int(np.ceil(len(chunks)/sequences_per_minibatch))
   critic_expected=self.ppo_epochs*int(np.ceil(N/self.minibatch_size))
   if len(actor_rows)!=actor_expected or self.actor_update_count-actor_before!=actor_expected or self.critic_update_count-critic_before!=critic_expected:
    raise RuntimeError("wave-segmented MARC PPO epoch/optimizer execution mismatch")
   out.update({"ppo_epochs_configured":float(self.ppo_epochs),"ppo_epochs_executed":float(len(actor_rows)/(actor_expected/self.ppo_epochs)),
    "actor_optimizer_steps_expected":float(actor_expected),"critic_optimizer_steps_expected":float(critic_expected),
    "marc_v2_deployment_distill_loss":0.0,"actor_phase_resets":float(np.asarray(r.actor_recurrent_phase_reset_flags).sum())})
  return out

 def _actor_recurrent_minibatch(self,r,group,obs,act,raw,oldlog,alive,adv,weights,ctx,tt):
  """One actor-only contiguous-sequence PPO minibatch."""
  length=max(z-s for _,s,z in group);batch=len(group)
  def padded(source):
   out=torch.zeros((length,batch,*source.shape[2:]),dtype=source.dtype,device=self.device)
   for b,(e,s,z) in enumerate(group):out[:z-s,b]=source[s:z,e]
   return out
  O,A,R,L,M,ADV,W,C=map(padded,(obs,act,raw,oldlog,alive,adv,weights,ctx))
  valid=torch.zeros(length,batch,device=self.device)
  for b,(_,s,z) in enumerate(group):valid[:z-s,b]=1
  EP=torch.zeros(length,batch,device=self.device)
  if r.episode_masks is not None:
   episode=tt(r.episode_masks)
   for b,(e,s,z) in enumerate(group):EP[:z-s,b]=episode[s:z,e]
  ah=torch.stack([tt(r.actor_hidden_before_step[s,e]) for e,s,_ in group]).detach()
  if self.recurrent.wave_segmented:
   reset=torch.as_tensor([r.actor_recurrent_phase_reset_flags[s,e] for e,s,_ in group],dtype=torch.bool,device=self.device)
   if bool(reset.any()):
    phase=self.actor.phase_hidden([r.wave_indices[s,e] for e,s,_ in group],M[0])
    ah=torch.where(reset[:,None,None],phase,ah)
  elif self.recurrent.state_memory:
   reset=torch.as_tensor([r.actor_recurrent_phase_reset_flags[s,e] for e,s,_ in group],dtype=torch.bool,device=self.device)
   ah=torch.where(reset[:,None,None],torch.zeros_like(ah),ah)
  surrogates=[];entropies=[];ratios=[];logratios=[];newlogs=[];masks=[];waveweights=[];anchor_kls=[]
  for t in range(length):
   mask=M[t]*valid[t,:,None]
   dist,ah=self.actor.distribution_step(O[t],self._ctx(C[t],True),ah,EP[t],mask)
   newlog=self.actor._squashed_log_prob(dist,R[t],A[t]);logratio,ratio=stable_ratio_terms(newlog,L[t])
   surrogate=torch.minimum(ratio*ADV[t],ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*ADV[t])
   sampled=dist.rsample();entropy=-self.actor._squashed_log_prob(dist,sampled,torch.tanh(sampled))
   if self.anchor.enabled and self.anchor.reference_actor is not None:
    with torch.no_grad():reference=self.anchor.reference_actor.distribution(O[t])
    anchor_kls.append(kl_divergence(dist,reference).sum(-1))
   surrogates.append(surrogate);entropies.append(entropy);ratios.append(ratio);logratios.append(logratio);newlogs.append(newlog);masks.append(mask);waveweights.append(W[t,:,None].expand_as(mask))
  surrogate,entropy,ratio,logratio,newlog,mask,ww=map(lambda x:torch.stack(x),(surrogates,entropies,ratios,logratios,newlogs,masks,waveweights))
  aw=ww if self.wave_balance.actor_enabled or self.recurrent.wave_boundary_reset else torch.ones_like(ww)
  actor_loss=-recurrent_alive_mean(surrogate*aw,M,valid);entropy_mean=recurrent_alive_mean(entropy,M,valid)
  anchor_mean=recurrent_alive_mean(torch.stack(anchor_kls),M,valid) if anchor_kls else torch.zeros((),device=self.device)
  anchor_loss=anchor_mean*self.anchor.effective_coefficient(self.sampled_steps)
  zero=torch.zeros((),device=self.device)
  merged=(actor_loss,zero,entropy_mean,anchor_loss,ratio.reshape(-1,self.num_agents),logratio.reshape(-1,self.num_agents),newlog.reshape(-1,self.num_agents),L.reshape(-1,self.num_agents),ah,None,float(anchor_mean.detach()))
  self.actor_optimizer.zero_grad();(actor_loss-self.entropy_coefficient*entropy_mean+anchor_loss).backward()
  arg=self._gradient_norm(self.actor.gru.parameters());ag=nn.utils.clip_grad_norm_(self.actor.trainable_policy_parameters(),self.max_grad_norm)
  self.actor_optimizer.step();self.actor_update_count+=1
  row=self._row(merged,mask.reshape(-1,self.num_agents),ag,0.,arg,0.)
  if self.recurrent.wave_segmented:
   gradient=self.actor.phase_initial_hidden.grad
   for wave in (1,2,3):row[f"actor_phase_embedding_grad_norm_wave{wave}"]=0. if gradient is None else float(gradient[wave-1].norm())
  return row

 def _recurrent_minibatch(self,r,group,obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx,tt):
  length=max(z-s for _,s,z in group);batch=len(group)
  def padded(source):
   out=torch.zeros((length,batch,*source.shape[2:]),dtype=source.dtype,device=self.device)
   for b,(e,s,z) in enumerate(group):out[:z-s,b]=source[s:z,e]
   return out
  O,A,R,L,M,ADV,OV,TG,W,C=map(padded,(obs,act,raw,oldlog,alive,adv,oldvalue,target,weights,ctx))
  valid=torch.zeros(length,batch,device=self.device)
  for b,(_,s,z) in enumerate(group):valid[:z-s,b]=1
  EP=torch.zeros(length,batch,device=self.device)
  if r.episode_masks is not None:
   episode=tt(r.episode_masks)
   for b,(e,s,z) in enumerate(group):EP[:z-s,b]=episode[s:z,e]
  def initial(saved):
   if saved is None:return None
   return torch.stack([tt(saved[s,e]) for e,s,_ in group]).detach()
  ah,ch=initial(r.actor_hidden_before_step),initial(r.critic_hidden_before_step)
  surrogates=[];errors=[];entropies=[];ratios=[];logratios=[];newlogs=[];masks=[];waveweights=[];anchor_kls=[]
  for t in range(length):
   mask=M[t]*valid[t,:,None]
   dist,ah=self.actor.distribution_step(O[t],self._ctx(C[t],True),ah,EP[t],mask)
   newlog=self.actor._squashed_log_prob(dist,R[t],A[t]);logratio,ratio=stable_ratio_terms(newlog,L[t])
   surrogate=torch.minimum(ratio*ADV[t],ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*ADV[t])
   sampled=dist.rsample();entropy=-self.actor._squashed_log_prob(dist,sampled,torch.tanh(sampled))
   value,ch=self.critic.forward_step(O[t],mask,self._ctx(C[t],False),ch,EP[t])
   clipped=OV[t]+(value-OV[t]).clamp(-self.clip_ratio,self.clip_ratio)
   error=torch.maximum((value-TG[t]).square(),(clipped-TG[t]).square()) if self.clip_value_loss else (value-TG[t]).square()
   if self.anchor.enabled and self.anchor.reference_actor is not None:
    with torch.no_grad():reference=self.anchor.reference_actor.distribution(O[t])
    anchor_kls.append(kl_divergence(dist,reference).sum(-1))
   surrogates.append(surrogate);errors.append(error);entropies.append(entropy);ratios.append(ratio);logratios.append(logratio);newlogs.append(newlog);masks.append(mask);waveweights.append(W[t,:,None].expand_as(mask))
  surrogate,error,entropy,ratio,logratio,newlog,mask,ww=map(lambda x:torch.stack(x),(surrogates,errors,entropies,ratios,logratios,newlogs,masks,waveweights))
  aw=ww if self.wave_balance.actor_enabled else torch.ones_like(ww);vw=ww if self.wave_balance.critic_enabled else torch.ones_like(ww)
  actor_loss=-recurrent_alive_mean(surrogate*aw,M,valid);value_loss=.5*recurrent_alive_mean(error*vw,M,valid);entropy_mean=recurrent_alive_mean(entropy,M,valid)
  anchor_mean=recurrent_alive_mean(torch.stack(anchor_kls),M,valid) if anchor_kls else torch.zeros((),device=self.device)
  anchor_loss=anchor_mean*self.anchor.effective_coefficient(self.sampled_steps)
  merged=(actor_loss,value_loss,entropy_mean,anchor_loss,ratio.reshape(-1,self.num_agents),logratio.reshape(-1,self.num_agents),newlog.reshape(-1,self.num_agents),L.reshape(-1,self.num_agents),ah,ch,float(anchor_mean.detach()))
  ag,cg,arg,crg=self._opt(merged);return self._row(merged,mask.reshape(-1,self.num_agents),ag,cg,arg,crg)

 @torch.no_grad()
 def _policy_diagnostics(self,r,obs,actions,alive,ctx):
  logstd=[];attention=[]
  for t in range(obs.shape[0]):
   hidden=None if r.actor_hidden_before_step is None else torch.as_tensor(r.actor_hidden_before_step[t],dtype=torch.float32,device=self.device)
   ep=None if r.episode_masks is None else torch.as_tensor(r.episode_masks[t],dtype=torch.float32,device=self.device)
   saved_options=getattr(r,"hta_options",None)
   options=None if saved_options is None else torch.as_tensor(saved_options[t],dtype=torch.long,device=self.device)
   if self.entity_attention_enabled or self.mission_film.enabled:
    dist,_,diag=self.actor.distribution_step(obs[t],self._ctx(ctx[t],True),hidden,ep,alive[t],return_attention=True,option_ids=options);attention.append(diag)
   else:dist,_=self.actor.distribution_step(obs[t],self._ctx(ctx[t],True),hidden,ep,alive[t],option_ids=options)
   logstd.append(dist.scale.log())
  logs=torch.stack(logstd);live=alive>.5;live_actions=actions[live];live_logs=logs[live];result={}
  for index,name in enumerate(("psi","theta","v")):
   result[f"policy_log_std_mean_{name}"]=float(live_logs[:,index].mean())
   for threshold,label in ((.9,"0_9"),(.99,"0_99"),(.999,"0_999")):result[f"action_abs_gt_{label}_fraction_{name}"]=float((live_actions[:,index].abs()>threshold).float().mean())
  result["entity_attention_enabled"]=float(self.entity_attention_enabled)
  result["mission_film_enabled"]=float(self.mission_film.enabled)
  if self.mission_film.enabled:
   query_alive=alive>.5
   def film_live(key):
    values=torch.stack([item[key] for item in attention])[query_alive]
    if values.numel()==0 or not torch.all(torch.isfinite(values)):raise FloatingPointError(f"invalid {key} diagnostics")
    return values
   dg=film_live("film_delta_gamma").abs();beta=film_live("film_beta").abs();residual=film_live("film_residual").abs();gamma=film_live("film_gamma")
   result.update({"mission_feature_norm":float(film_live("mission_feature_norm").mean()),
    "film_delta_gamma_abs_mean":float(dg.mean()),"film_delta_gamma_abs_max":float(dg.max()),
    "film_beta_abs_mean":float(beta.mean()),"film_beta_abs_max":float(beta.max()),
    "film_residual_abs_mean":float(residual.mean()),"film_residual_abs_max":float(residual.max()),
    "film_gamma_mean":float(gamma.mean()),"film_gamma_min":float(gamma.min()),"film_gamma_max":float(gamma.max()),
    "film_hidden_base_norm":float(film_live("film_hidden_base_norm").mean()),
    "film_hidden_modulated_norm":float(film_live("film_hidden_modulated_norm").mean()),
    "film_saturation_fraction":float(film_live("film_saturation").mean())})
  if self.entity_attention_enabled and attention:
   query_alive=alive>.5
   def live_values(key):
    values=torch.stack([item[key] for item in attention])[query_alive]
    if values.numel()==0:raise FloatingPointError(f"no live samples for {key} diagnostics")
    if not torch.all(torch.isfinite(values)):raise FloatingPointError(f"non-finite {key} diagnostics")
    return values
   for group in ("ally","enemy"):
    weights=torch.stack([item[f"{group}_attention_weights"] for item in attention])
    entities=torch.stack([item[f"{group}_entity_alive"] for item in attention])
    entropy=-(weights*weights.clamp_min(1e-12).log()).sum(-1).mean(-1)
    top1=weights.max(-1).values.mean(-1)
    dead_mass=(weights*(entities<=.5).unsqueeze(-2)).sum(-1).mean(-1)
    result[f"{group}_attention_entropy"]=float(entropy[query_alive].mean())
    result[f"{group}_attention_top1_weight"]=float(top1[query_alive].mean())
    result[f"{group}_alive_entity_count"]=float(entities[query_alive].sum(-1).float().mean())
    result[f"{group}_attention_dead_mass"]=float(dead_mass[query_alive].mean())
   result["attention_dead_mass"]=max(result["ally_attention_dead_mass"],result["enemy_attention_dead_mass"])
   result["entity_feature_norm"]=float(live_values("entity_feature_norm").mean())
   if self.actor.entity_attention_mode in {"residual","gated_residual"}:
    for key in ("entity_base_feature_norm","entity_delta_norm","entity_delta_to_base_ratio"):
     result[key]=float(live_values(key).mean())
   if self.actor.entity_attention_mode=="gated_residual":
    gate=live_values("entity_gate")
    result.update({"entity_gate_mean":float(gate.mean()),"entity_gate_std":float(gate.std(unbiased=False)),
                   "entity_gate_min":float(gate.min()),"entity_gate_max":float(gate.max()),
                   "entity_gate_p10":float(torch.quantile(gate,.1)),"entity_gate_p50":float(torch.quantile(gate,.5)),
                   "entity_gate_p90":float(torch.quantile(gate,.9))})
   if self.actor.entity_attention_mode in {"frozen_base_mean_residual","frozen_base_dual_bounded_mean_residual"}:
    delta=live_values("entity_delta_mu");base_mean=live_values("base_mean");base_logstd=live_values("base_log_std")
    absolute=delta.abs();result.update({
     "entity_delta_mu_abs_mean":float(absolute.mean()),"entity_delta_mu_rms":float(delta.square().mean().sqrt()),
     "entity_delta_mu_abs_max":float(absolute.max()),
     "entity_delta_mu_heading_abs_mean":float(absolute[:,0].mean()),
     "entity_delta_mu_pitch_abs_mean":float(absolute[:,1].mean()),
     "entity_delta_mu_speed_abs_mean":float(absolute[:,2].mean()),
     "entity_delta_mu_bound_fraction":float((absolute>.9*self.actor.max_mean_correction).float().mean()),
     "base_mean_abs_mean":float(base_mean.abs().mean()),"final_mean_abs_mean":float((base_mean+delta).abs().mean()),
     "base_std_mean":float(base_logstd.exp().mean()),"final_std_mean":float(base_logstd.exp().mean()),
     "entity_delta_logstd_abs_max":float(live_values("entity_delta_logstd_abs_max").max()),
    })
    if self.actor.entity_attention_mode=="frozen_base_dual_bounded_mean_residual":
     sigma=base_logstd.exp();ratio=absolute/sigma;dual_scale=live_values("entity_dual_scale")
     relative_active=live_values("entity_relative_scale_active").float();effective=live_values("entity_dual_bound_effective").float()
     saturation=(absolute>.9*dual_scale).float();kl=.5*ratio.square().sum(-1)
     result.update({
      "entity_dual_scale_mean":float(dual_scale.mean()),"entity_dual_scale_min":float(dual_scale.min()),"entity_dual_scale_max":float(dual_scale.max()),
      "entity_dual_scale_heading_mean":float(dual_scale[:,0].mean()),"entity_dual_scale_pitch_mean":float(dual_scale[:,1].mean()),"entity_dual_scale_speed_mean":float(dual_scale[:,2].mean()),
      "entity_delta_mu_over_sigma_abs_mean":float(ratio.mean()),"entity_delta_mu_over_sigma_abs_max":float(ratio.max()),
      "entity_delta_mu_over_sigma_heading_abs_mean":float(ratio[:,0].mean()),"entity_delta_mu_over_sigma_heading_abs_max":float(ratio[:,0].max()),
      "entity_delta_mu_over_sigma_pitch_abs_mean":float(ratio[:,1].mean()),"entity_delta_mu_over_sigma_pitch_abs_max":float(ratio[:,1].max()),
      "entity_delta_mu_over_sigma_speed_abs_mean":float(ratio[:,2].mean()),"entity_delta_mu_over_sigma_speed_abs_max":float(ratio[:,2].max()),
      "entity_source_relative_kl_mean":float(kl.mean()),"entity_source_relative_kl_max":float(kl.max()),
      "relative_scale_active_fraction":float(relative_active.mean()),"relative_scale_active_heading_fraction":float(relative_active[:,0].mean()),
      "relative_scale_active_pitch_fraction":float(relative_active[:,1].mean()),"relative_scale_active_speed_fraction":float(relative_active[:,2].mean()),
      "dual_bound_effective_fraction":float(effective.mean()),"dual_bound_effective_heading_fraction":float(effective[:,0].mean()),
      "dual_bound_effective_pitch_fraction":float(effective[:,1].mean()),"dual_bound_effective_speed_fraction":float(effective[:,2].mean()),
      "relative_saturation_fraction":float(saturation.mean()),"relative_saturation_heading_fraction":float(saturation[:,0].mean()),
      "relative_saturation_pitch_fraction":float(saturation[:,1].mean()),"relative_saturation_speed_fraction":float(saturation[:,2].mean()),
     })
    result.update(self.frozen_actor_drift_metrics())
  return result

 def capture_frozen_actor_reference(self):
  if not self.fbmr_enabled:return
  self._frozen_actor_reference={name:parameter.detach().clone() for name,parameter in self.actor.frozen_baseline_named_parameters()}

 def frozen_actor_drift_metrics(self):
  if not self.fbmr_enabled:return {}
  if self._frozen_actor_reference is None:raise RuntimeError("FBMR frozen actor reference is unavailable")
  maxima={"backbone":0.,"mean":0.,"log_std":0.}
  for name,parameter in self.actor.frozen_baseline_named_parameters():
   family=name.split(".",1)[0];maximum=float((parameter.detach()-self._frozen_actor_reference[name]).abs().max())
   maxima[family]=max(maxima[family],maximum)
  return {"frozen_base_parameter_drift_max":max(maxima.values()),
          "frozen_backbone_parameter_drift_max":maxima["backbone"],
          "frozen_mean_head_parameter_drift_max":maxima["mean"],
          "frozen_logstd_head_parameter_drift_max":maxima["log_std"]}

 def frozen_actor_sha256(self):
  if not self.fbmr_enabled:return None
  digest=hashlib.sha256()
  for name,parameter in self.actor.frozen_baseline_named_parameters():
   digest.update(name.encode());digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
  return digest.hexdigest()
 @staticmethod
 def _module_state_sha256(module):
  digest=hashlib.sha256()
  for name,value in sorted(module.state_dict().items()):
   digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
  return digest.hexdigest()
 def reference_variance_actor_sha256(self):
  return None if self.reference_variance_actor is None else self._module_state_sha256(self.reference_variance_actor)
 def _attach_reference_variance_actor(self,source_checkpoint_sha256,source_sampled_steps,source_training_seed):
  cpu_rng=torch.get_rng_state().clone();cuda_rng=[value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
  self.reference_variance_actor=deepcopy(self.actor).to(self.device).eval()
  for parameter in self.reference_variance_actor.parameters():parameter.requires_grad_(False);parameter.grad=None
  for parameter in self.actor.log_std.parameters():parameter.requires_grad_(False);parameter.grad=None
  if not torch.equal(cpu_rng,torch.get_rng_state()):raise RuntimeError("creating RV reference actor consumed CPU RNG")
  if torch.cuda.is_available() and any(not torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())):raise RuntimeError("creating RV reference actor consumed CUDA RNG")
  self.reference_variance_initial_sha256=self.reference_variance_actor_sha256()
  if self.reference_variance_initial_sha256!=self._module_state_sha256(self.actor):raise RuntimeError("RV reference actor/source actor identity mismatch")
  self.reference_variance_source_checkpoint_sha256=source_checkpoint_sha256
  self.reference_variance_source_sampled_steps=int(source_sampled_steps)
  self.reference_variance_source_training_seed=int(source_training_seed)
 def finalize_reference_variance_branch(self):
  if self._pending_reference_variance_branch is None:return False
  values=self._pending_reference_variance_branch;self._pending_reference_variance_branch=None
  self._attach_reference_variance_actor(*values);return True
 def module_protocol(self):
  raw=json.dumps(self.modules_config,sort_keys=True,separators=(",",":"));return {"enabled_modules":enabled_module_names(self.modules_config),"module_config":deepcopy(self.modules_config),"module_config_sha256":hashlib.sha256(raw.encode()).hexdigest()}
 def capture_rng_state(self):
  cuda_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
  value={"python_random_state":random.getstate(),"numpy_random_state":np.random.get_state(),"torch_cpu_rng_state":torch.get_rng_state(),"torch_cuda_rng_state_all":cuda_states,"trainer_permutation_rng_state":deepcopy(self.rng.bit_generator.state)}
  if self.inter_wave_credit.enabled:value["iw_replay_rng_state"]=deepcopy(self.iw_rng.bit_generator.state)
  if self.counterfactual_inter_wave_credit.enabled:value["caiw_rng_state"]=deepcopy(self.caiw_rng.bit_generator.state)
  if self.boundary_redistributed_segment_credit.enabled:value["brsc_rng_state"]=deepcopy(self.brsc_rng.bit_generator.state)
  if self.hierarchical_temporal_abstraction.enabled:
   value["hta_rng_state"]=deepcopy(self.hta_rng.bit_generator.state)
   value["hta_manager_generator_state"]=self.hta_manager_generator.get_state().cpu()
  return value
 def restore_rng_state(self,state):
  saved=state.get("rng_state") if isinstance(state,dict) else None
  if not isinstance(saved,dict):
   self.rng_restore_metadata={"rng_state_available":False,"rng_state_restored":False,"cuda_rng_state_restored":False};return False
  required=("python_random_state","numpy_random_state","torch_cpu_rng_state","trainer_permutation_rng_state")
  if any(key not in saved for key in required):
   self.rng_restore_metadata={"rng_state_available":False,"rng_state_restored":False,"cuda_rng_state_restored":False};return False
  random.setstate(saved["python_random_state"]);np.random.set_state(saved["numpy_random_state"])
  torch.set_rng_state(saved["torch_cpu_rng_state"].cpu());self.rng.bit_generator.state=deepcopy(saved["trainer_permutation_rng_state"])
  if saved.get("iw_replay_rng_state") is not None:self.iw_rng.bit_generator.state=deepcopy(saved["iw_replay_rng_state"])
  if saved.get("caiw_rng_state") is not None:self.caiw_rng.bit_generator.state=deepcopy(saved["caiw_rng_state"])
  if saved.get("brsc_rng_state") is not None:self.brsc_rng.bit_generator.state=deepcopy(saved["brsc_rng_state"])
  if saved.get("hta_rng_state") is not None:self.hta_rng.bit_generator.state=deepcopy(saved["hta_rng_state"])
  if saved.get("hta_manager_generator_state") is not None:self.hta_manager_generator.set_state(saved["hta_manager_generator_state"].cpu())
  cuda_saved=saved.get("torch_cuda_rng_state_all",[]);cuda_restored=False
  if torch.cuda.is_available() and cuda_saved:
   torch.cuda.set_rng_state_all([value.cpu() for value in cuda_saved]);cuda_restored=True
  self.rng_restore_metadata={"rng_state_available":True,"rng_state_restored":True,"cuda_rng_state_restored":cuda_restored}
  return True
 def _base_checkpoint_state(self,extra=None):
  if self.fbmr_enabled and self.frozen_actor_drift_metrics()["frozen_base_parameter_drift_max"]!=0.:raise RuntimeError("FBMR frozen baseline actor changed before checkpoint save")
  return {"algorithm":"modular_mappo","modular_mappo_impl_version":MODULAR_MAPPO_IMPL_VERSION,"baseline_mappo_impl_version":MAPPO_IMPL_VERSION,"development_feature_versions":{"advantage_priority":ADVANTAGE_PRIORITY_VERSION,"ppo_stabilization":PPO_STABILIZATION_VERSION,"actor_lr_decay":ACTOR_LR_DECAY_VERSION,"entity_attention":1,"fbmr_ea":1,"fbmr_dual_bound":1,"mission_film":MISSION_FILM_VERSION,"actor_kl_guard":ACTOR_KL_GUARD_VERSION,"inter_wave_credit":IWSC_MAPPO_VERSION,"counterfactual_inter_wave_credit":CAIW_MAPPO_VERSION,"boundary_redistributed_segment_credit":BRSC_MAPPO_VERSION,"hierarchical_temporal_abstraction":HTA_MAPPO_VERSION,"hta_worker_consolidation":HTA_WORKER_CONSOLIDATION_VERSION,"sequential_wave_gradient_projection":SWGP_MAPPO_VERSION,"team_mean_credit":TEAM_MEAN_CREDIT_VERSION,"persistent_wave_trajectory_replay":PWTR_MAPPO_VERSION,"wave1_sensitivity_gating":W1SG_MAPPO_VERSION,"milestone_aware_retention_credit":MARC_MAPPO_VERSION},"actor":self.actor.state_dict(),"critic":self.critic.state_dict(),"actor_optimizer":self.actor_optimizer.state_dict(),"critic_optimizer":self.critic_optimizer.state_dict(),"manager_actor":None if self.manager_actor is None else self.manager_actor.state_dict(),"manager_critic":None if self.manager_critic is None else self.manager_critic.state_dict(),"manager_actor_optimizer":None if self.manager_actor_optimizer is None else self.manager_actor_optimizer.state_dict(),"manager_critic_optimizer":None if self.manager_critic_optimizer is None else self.manager_critic_optimizer.state_dict(),"manager_actor_updates":self.manager_actor_update_count,"manager_critic_updates":self.manager_critic_update_count,"manager_optimizer_steps":self.manager_optimizer_step_count,"hta_option_usage_counts":self.hta_option_usage_counts.tolist(),"hta_decision_reason_counts":deepcopy(self.hta_decision_reason_counts),"iw_critic":None if self.iw_critic is None else self.iw_critic.state_dict(),"iw_critic_optimizer":None if self.iw_critic_optimizer is None else self.iw_critic_optimizer.state_dict(),"inter_wave_credit_state":self.inter_wave_credit.state_dict(),"iw_conflict_count":self.iw_conflict_count,"iw_gradient_step_count":self.iw_gradient_step_count,"caiw_critic":None if self.caiw_critic is None else self.caiw_critic.state_dict(),"caiw_critic_optimizer":None if self.caiw_critic_optimizer is None else self.caiw_critic_optimizer.state_dict(),"counterfactual_inter_wave_credit_state":self.counterfactual_inter_wave_credit.state_dict(),"caiw_rng_state":deepcopy(self.caiw_rng.bit_generator.state),"caiw_conflict_count":self.caiw_conflict_count,"caiw_gradient_step_count":self.caiw_gradient_step_count,"caiw_trust_cap_count":self.caiw_trust_cap_count,"caiw_aux_induced_clip_count":self.caiw_aux_induced_clip_count,"brsc_critic":None if self.brsc_critic is None else self.brsc_critic.state_dict(),"brsc_critic_optimizer":None if self.brsc_critic_optimizer is None else self.brsc_critic_optimizer.state_dict(),"boundary_redistributed_segment_credit_state":self.boundary_redistributed_segment_credit.state_dict(),"brsc_rng_state":deepcopy(self.brsc_rng.bit_generator.state),"brsc_conflict_count":self.brsc_conflict_count,"brsc_gradient_step_count":self.brsc_gradient_step_count,"brsc_trust_cap_count":self.brsc_trust_cap_count,"brsc_aux_induced_clip_count":self.brsc_aux_induced_clip_count,"sequential_wave_gradient_projection_state":self.sequential_wave_gradient_projection.state_dict() if self.sequential_wave_gradient_projection.enabled else None,"persistent_wave_trajectory_replay_state":self.persistent_wave_trajectory_replay.state_dict() if self.persistent_wave_trajectory_replay.enabled else None,"wave1_sensitivity_gating_state":self.wave1_sensitivity_gating.state_dict() if self.wave1_sensitivity_gating.enabled else None,"milestone_aware_retention_credit_state":self.milestone_aware_retention_credit.state_dict() if self.milestone_aware_retention_credit.enabled else None,"popart":self.popart.state_dict(),"ppo_updates":self.ppo_update_count,"actor_updates":self.actor_update_count,"critic_updates":self.critic_update_count,"sampled_steps":self.sampled_steps,"vector_steps":self.vector_steps,"kl_hard_stop_count":self.kl_hard_stop_count,"actor_kl_guard_hard_stop_count":self.actor_kl_guard_hard_stop_count,"actor_kl_guard_actor_epochs_total":self.actor_kl_guard_actor_epochs_total,"actor_kl_guard_actor_epochs_min":self.actor_kl_guard_actor_epochs_min,"rng_state":self.capture_rng_state(),"rng_state_available":True,"rng_state_restored":self.rng_restore_metadata["rng_state_restored"],"cuda_rng_state_restored":self.rng_restore_metadata["cuda_rng_state_restored"],**self.module_protocol(),"warm_start_provenance":self.warm_start_provenance,"anchor_provenance":self.anchor_provenance,"anchor_reference_actor_state":None if self.anchor.reference_actor is None else self.anchor.reference_actor.state_dict(),"fbmr_branch_metadata":deepcopy(self.fbmr_branch_metadata),"frozen_base_actor_state":None if self._frozen_actor_reference is None else self._frozen_actor_reference,"frozen_base_actor_sha256":self.frozen_actor_sha256(),"extra":extra or {}}
 def checkpoint_state(self,extra=None):
  if self.reference_variance.enabled:
   if self.reference_variance_actor is None:raise RuntimeError("RV reference actor missing before checkpoint save")
   if self.reference_variance_actor_sha256()!=self.reference_variance_initial_sha256:raise RuntimeError("RV reference actor mutated")
  state=self._base_checkpoint_state(extra)
  if self.development_method in {"jiao2025_matched_plain_3m","jiao2025_core_3m","marc_credit_balance_ablation","marc_mappo_wsgru_v1","marc_mappo_state_memory_v1"}:
   state["extra"]["development_method"]=self.development_method
  state["development_feature_versions"]["milestone_aware_retention_credit"]=self.milestone_aware_retention_credit.version
  state["development_feature_versions"]["wave_specific_actor_isolation"]=WSAI_MAPPO_VERSION
  state["development_feature_versions"]["wave_specific_mean_heads"]=WSMH_MAPPO_VERSION
  state["development_feature_versions"]["actor_gradient_clipping"]=ACTOR_GRAD_CLIP_VERSION
  state["development_feature_versions"]["deployment_aligned_wave_exploration"]=DAWE_MAPPO_VERSION
  state["development_feature_versions"]["reference_variance"]=REFERENCE_VARIANCE_VERSION
  state.update({"wave2_actor_state":None if self.wave2_actor is None else self.wave2_actor.state_dict(),
   "wave3_actor_state":None if self.wave3_actor is None else self.wave3_actor.state_dict(),
   "wave2_actor_optimizer_state":None if self.wave2_actor_optimizer is None else self.wave2_actor_optimizer.state_dict(),
   "wave3_actor_optimizer_state":None if self.wave3_actor_optimizer is None else self.wave3_actor_optimizer.state_dict(),
   "wave_specific_actor_isolation_state":self.wave_specific_actor_isolation.state_dict() if self.wave_specific_actor_isolation.enabled else None})
  state.update({"wave2_mean_state":None if self.wave2_mean is None else self.wave2_mean.state_dict(),
   "wave3_mean_state":None if self.wave3_mean is None else self.wave3_mean.state_dict(),
   "wave2_mean_optimizer_state":None if self.wave2_mean_optimizer is None else self.wave2_mean_optimizer.state_dict(),
   "wave3_mean_optimizer_state":None if self.wave3_mean_optimizer is None else self.wave3_mean_optimizer.state_dict(),
   "wave_specific_mean_heads_state":self.wave_specific_mean_heads.state_dict() if self.wave_specific_mean_heads.enabled else None})
  state["actor_gradient_clipping_state"]=self.actor_gradient_clipping.state_dict() if self.actor_gradient_clipping.enabled else None
  state["deployment_aligned_wave_exploration_state"]=(self.deployment_aligned_wave_exploration.state_dict()
   if self.deployment_aligned_wave_exploration.enabled else None)
  state.update({"reference_variance_actor_state":None if self.reference_variance_actor is None else self.reference_variance_actor.state_dict(),
   "reference_variance_actor_sha256":self.reference_variance_actor_sha256(),
   "reference_variance_source_checkpoint_sha256":self.reference_variance_source_checkpoint_sha256,
   "reference_variance_source_sampled_steps":self.reference_variance_source_sampled_steps,
   "reference_variance_source_training_seed":self.reference_variance_source_training_seed,
   "reference_variance_state":self.reference_variance.state_dict() if self.reference_variance.enabled else None})
  return state
 def save(self,path,extra=None):Path(path).parent.mkdir(parents=True,exist_ok=True);torch.save(self.checkpoint_state(extra),path)
 def load(self,path,strict_protocol=True,restore_rng=True,defer_reference_attach=False):
  state=torch.load(path,map_location=self.device,weights_only=False)
  if strict_protocol and self.development_method in {"jiao2025_matched_plain_3m","jiao2025_core_3m","marc_credit_balance_ablation","marc_mappo_wsgru_v1","marc_mappo_state_memory_v1"} and state.get("extra",{}).get("development_method")!=self.development_method:
   raise RuntimeError("MARC GRU screen checkpoint development method mismatch")
  if state.get("algorithm")!="modular_mappo":raise RuntimeError("not a modular_mappo checkpoint")
  checkpoint_version=state.get("modular_mappo_impl_version")
  if checkpoint_version!=MODULAR_MAPPO_IMPL_VERSION:raise RuntimeError(f"modular implementation version mismatch: checkpoint={checkpoint_version}, current={MODULAR_MAPPO_IMPL_VERSION}")
  if state.get("baseline_mappo_impl_version")!=MAPPO_IMPL_VERSION:raise RuntimeError("baseline MAPPO implementation version mismatch")
  if strict_protocol and self.milestone_aware_retention_credit.enabled:
   checkpoint_marc_version=state.get("development_feature_versions",{}).get("milestone_aware_retention_credit")
   if checkpoint_marc_version!=self.milestone_aware_retention_credit.version:
    raise RuntimeError(f"checkpoint MARC version mismatch: checkpoint={checkpoint_marc_version}, current={self.milestone_aware_retention_credit.version}")
  if strict_protocol and state.get("module_config_sha256")!=self.module_protocol()["module_config_sha256"]:raise RuntimeError("checkpoint module protocol mismatch")
  if strict_protocol and (self.entity_attention_enabled or self.advantage_priority.enabled or self.ppo_stabilization.enabled or self.actor_lr_decay.enabled or self.mission_film.enabled or self.actor_kl_guard.enabled or self.inter_wave_credit.enabled or self.counterfactual_inter_wave_credit.enabled or self.boundary_redistributed_segment_credit.enabled or self.hierarchical_temporal_abstraction.enabled or self.sequential_wave_gradient_projection.enabled or self.persistent_wave_trajectory_replay.enabled or self.wave1_sensitivity_gating.enabled or self.wave_specific_actor_isolation.enabled or self.wave_specific_mean_heads.enabled or self.actor_gradient_clipping.enabled or self.deployment_aligned_wave_exploration.enabled or self.reference_variance.enabled or self.milestone_aware_retention_credit.enabled):
   versions=state.get("development_feature_versions",{});expected={"advantage_priority":ADVANTAGE_PRIORITY_VERSION,"ppo_stabilization":PPO_STABILIZATION_VERSION,"entity_attention":1}
   if any(versions.get(key)!=value for key,value in expected.items()):raise RuntimeError("checkpoint development feature version mismatch")
   if self.actor_lr_decay.enabled and versions.get("actor_lr_decay")!=ACTOR_LR_DECAY_VERSION:raise RuntimeError("checkpoint actor_lr_decay feature version mismatch")
   if self.mission_film.enabled and versions.get("mission_film")!=MISSION_FILM_VERSION:raise RuntimeError("checkpoint mission_film feature version mismatch")
   if self.actor_kl_guard.enabled and versions.get("actor_kl_guard")!=ACTOR_KL_GUARD_VERSION:raise RuntimeError("checkpoint actor_kl_guard feature version mismatch")
   if self.inter_wave_credit.enabled and versions.get("inter_wave_credit")!=IWSC_MAPPO_VERSION:raise RuntimeError("checkpoint IWSC feature version mismatch")
   if self.counterfactual_inter_wave_credit.enabled and versions.get("counterfactual_inter_wave_credit")!=CAIW_MAPPO_VERSION:raise RuntimeError("checkpoint CAIW feature version mismatch")
   if self.boundary_redistributed_segment_credit.enabled and versions.get("boundary_redistributed_segment_credit")!=BRSC_MAPPO_VERSION:raise RuntimeError("checkpoint BRSC feature version mismatch")
   if self.hierarchical_temporal_abstraction.enabled and versions.get("hierarchical_temporal_abstraction")!=HTA_MAPPO_VERSION:raise RuntimeError("checkpoint HTA feature version mismatch")
   if self.hta_worker_consolidation.enabled and versions.get("hta_worker_consolidation")!=HTA_WORKER_CONSOLIDATION_VERSION:raise RuntimeError("checkpoint HTA Worker consolidation feature version mismatch")
   if self.sequential_wave_gradient_projection.enabled and versions.get("sequential_wave_gradient_projection")!=SWGP_MAPPO_VERSION:raise RuntimeError("checkpoint SWGP feature version mismatch")
   if self.team_mean_credit.enabled and versions.get("team_mean_credit")!=TEAM_MEAN_CREDIT_VERSION:raise RuntimeError("checkpoint team_mean_credit feature version mismatch")
   if self.persistent_wave_trajectory_replay.enabled and versions.get("persistent_wave_trajectory_replay")!=PWTR_MAPPO_VERSION:raise RuntimeError("checkpoint PWTR feature version mismatch")
   if self.wave1_sensitivity_gating.enabled and versions.get("wave1_sensitivity_gating")!=W1SG_MAPPO_VERSION:raise RuntimeError("checkpoint W1SG feature version mismatch")
   if self.wave_specific_actor_isolation.enabled and versions.get("wave_specific_actor_isolation")!=WSAI_MAPPO_VERSION:raise RuntimeError("checkpoint WSAI feature version mismatch")
   if self.wave_specific_mean_heads.enabled and versions.get("wave_specific_mean_heads")!=WSMH_MAPPO_VERSION:raise RuntimeError("checkpoint WSMH feature version mismatch")
   if self.actor_gradient_clipping.enabled and versions.get("actor_gradient_clipping")!=ACTOR_GRAD_CLIP_VERSION:raise RuntimeError("checkpoint actor gradient clipping feature version mismatch")
   if self.deployment_aligned_wave_exploration.enabled and versions.get("deployment_aligned_wave_exploration")!=DAWE_MAPPO_VERSION:raise RuntimeError("checkpoint DAWE feature version mismatch")
   if self.reference_variance.enabled and versions.get("reference_variance")!=REFERENCE_VARIANCE_VERSION:raise RuntimeError("checkpoint RV feature version mismatch")
   if self.milestone_aware_retention_credit.enabled and versions.get("milestone_aware_retention_credit")!=self.milestone_aware_retention_credit.version:raise RuntimeError("checkpoint MARC feature version mismatch")
  self.actor.load_state_dict(state["actor"]);self.critic.load_state_dict(state["critic"]);self.popart.load_state_dict(state.get("popart",{}),strict=False)
  self.actor_optimizer.load_state_dict(state["actor_optimizer"]);self.critic_optimizer.load_state_dict(state["critic_optimizer"])
  if self.reference_variance.enabled:
   if strict_protocol:
    reference_state=state.get("reference_variance_actor_state")
    required=("reference_variance_actor_sha256","reference_variance_source_checkpoint_sha256",
              "reference_variance_source_sampled_steps","reference_variance_source_training_seed")
    if not isinstance(reference_state,dict) or any(state.get(key) is None for key in required):
     raise RuntimeError("strict RV resume checkpoint is not self-contained")
    self.reference_variance_actor=deepcopy(self.actor).to(self.device)
    self.reference_variance_actor.load_state_dict(reference_state);self.reference_variance_actor.eval()
    for parameter in self.reference_variance_actor.parameters():parameter.requires_grad_(False);parameter.grad=None
    for parameter in self.actor.log_std.parameters():parameter.requires_grad_(False);parameter.grad=None
    self.reference_variance_initial_sha256=state["reference_variance_actor_sha256"]
    self.reference_variance_source_checkpoint_sha256=state["reference_variance_source_checkpoint_sha256"]
    self.reference_variance_source_sampled_steps=int(state["reference_variance_source_sampled_steps"])
    self.reference_variance_source_training_seed=int(state["reference_variance_source_training_seed"])
    if self.reference_variance_actor_sha256()!=self.reference_variance_initial_sha256:raise RuntimeError("RV reference actor SHA mismatch")
   else:
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
     for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    extra=state.get("extra",{})
    values=(digest.hexdigest(),state.get("sampled_steps",-1),extra.get("training_seed",-1))
    if defer_reference_attach:self._pending_reference_variance_branch=values
    else:self._attach_reference_variance_actor(*values)
  if self.wave_specific_actor_isolation.enabled:
   if strict_protocol:
    required=("wave2_actor_state","wave3_actor_state","wave2_actor_optimizer_state","wave3_actor_optimizer_state")
    if any(state.get(key) is None for key in required):raise RuntimeError("WSAI checkpoint lacks isolated actor state")
    self.wave2_actor.load_state_dict(state["wave2_actor_state"]);self.wave3_actor.load_state_dict(state["wave3_actor_state"])
    self.wave2_actor_optimizer.load_state_dict(state["wave2_actor_optimizer_state"]);self.wave3_actor_optimizer.load_state_dict(state["wave3_actor_optimizer_state"])
   else:
    self.wave2_actor.load_state_dict(self.actor.state_dict());self.wave3_actor.load_state_dict(self.actor.state_dict())
    self.wave2_actor_optimizer.load_state_dict(deepcopy(self.actor_optimizer.state_dict()))
    self.wave3_actor_optimizer.load_state_dict(deepcopy(self.actor_optimizer.state_dict()))
  if self.wave_specific_mean_heads.enabled:
   if strict_protocol:
    required=("wave2_mean_state","wave3_mean_state","wave2_mean_optimizer_state","wave3_mean_optimizer_state")
    if any(state.get(key) is None for key in required):raise RuntimeError("WSMH checkpoint lacks wave mean state")
    self.wave2_mean.load_state_dict(state["wave2_mean_state"]);self.wave3_mean.load_state_dict(state["wave3_mean_state"])
    self.wave2_mean_optimizer.load_state_dict(state["wave2_mean_optimizer_state"]);self.wave3_mean_optimizer.load_state_dict(state["wave3_mean_optimizer_state"])
   else:
    self.wave2_mean.load_state_dict(self.actor.mean.state_dict());self.wave3_mean.load_state_dict(self.actor.mean.state_dict())
    self._clone_source_mean_optimizer_state(self.wave2_mean_optimizer);self._clone_source_mean_optimizer_state(self.wave3_mean_optimizer)
  if self.hierarchical_temporal_abstraction.enabled:
   required=("manager_actor","manager_critic","manager_actor_optimizer","manager_critic_optimizer")
   if any(state.get(key) is None for key in required):raise RuntimeError("HTA checkpoint lacks manager state")
   self.manager_actor.load_state_dict(state["manager_actor"]);self.manager_critic.load_state_dict(state["manager_critic"])
   self.manager_actor_optimizer.load_state_dict(state["manager_actor_optimizer"]);self.manager_critic_optimizer.load_state_dict(state["manager_critic_optimizer"])
   self.manager_actor_update_count=int(state.get("manager_actor_updates",0));self.manager_critic_update_count=int(state.get("manager_critic_updates",0));self.manager_optimizer_step_count=int(state.get("manager_optimizer_steps",0))
   self.hta_option_usage_counts=np.asarray(state.get("hta_option_usage_counts",[0,0,0,0]),dtype=np.int64)
   self.hta_decision_reason_counts={key:int(state.get("hta_decision_reason_counts",{}).get(key,0)) for key in self.hta_decision_reason_counts}
  if self.inter_wave_credit.enabled:
   if state.get("iw_critic") is None or state.get("iw_critic_optimizer") is None:raise RuntimeError("IWSC checkpoint lacks quality critic state")
   self.iw_critic.load_state_dict(state["iw_critic"]);self.iw_critic_optimizer.load_state_dict(state["iw_critic_optimizer"])
   self.inter_wave_credit.load_state_dict(state.get("inter_wave_credit_state"));self.iw_conflict_count=int(state.get("iw_conflict_count",0));self.iw_gradient_step_count=int(state.get("iw_gradient_step_count",0))
  if self.counterfactual_inter_wave_credit.enabled:
   if state.get("caiw_critic") is None or state.get("caiw_critic_optimizer") is None:raise RuntimeError("CAIW checkpoint lacks outcome critic state")
   self.caiw_critic.load_state_dict(state["caiw_critic"]);self.caiw_critic_optimizer.load_state_dict(state["caiw_critic_optimizer"]);self.counterfactual_inter_wave_credit.load_state_dict(state.get("counterfactual_inter_wave_credit_state"));self.caiw_rng.bit_generator.state=deepcopy(state["caiw_rng_state"]);self.caiw_conflict_count=int(state.get("caiw_conflict_count",0));self.caiw_gradient_step_count=int(state.get("caiw_gradient_step_count",0));self.caiw_trust_cap_count=int(state.get("caiw_trust_cap_count",0));self.caiw_aux_induced_clip_count=int(state.get("caiw_aux_induced_clip_count",0))
  if self.boundary_redistributed_segment_credit.enabled:
   if state.get("brsc_critic") is None or state.get("brsc_critic_optimizer") is None:raise RuntimeError("BRSC checkpoint lacks boundary critic state")
   self.brsc_critic.load_state_dict(state["brsc_critic"]);self.brsc_critic_optimizer.load_state_dict(state["brsc_critic_optimizer"]);self.boundary_redistributed_segment_credit.load_state_dict(state.get("boundary_redistributed_segment_credit_state"));self.brsc_rng.bit_generator.state=deepcopy(state["brsc_rng_state"]);self.brsc_conflict_count=int(state.get("brsc_conflict_count",0));self.brsc_gradient_step_count=int(state.get("brsc_gradient_step_count",0));self.brsc_trust_cap_count=int(state.get("brsc_trust_cap_count",0));self.brsc_aux_induced_clip_count=int(state.get("brsc_aux_induced_clip_count",0))
  self.warm_start_provenance=state.get("warm_start_provenance",{});self.anchor_provenance=state.get("anchor_provenance",{})
  self.fbmr_branch_metadata=deepcopy(state.get("fbmr_branch_metadata",{}))
  if self.fbmr_enabled:
   if versions.get("fbmr_ea")!=1:raise RuntimeError("checkpoint FBMR-EA feature version mismatch")
   if self.actor.entity_attention_mode=="frozen_base_dual_bounded_mean_residual" and versions.get("fbmr_dual_bound")!=1:raise RuntimeError("checkpoint FBMR dual-bound feature version mismatch")
   reference=state.get("frozen_base_actor_state")
   if not isinstance(reference,dict):raise RuntimeError("FBMR checkpoint lacks frozen baseline actor reference")
   self._frozen_actor_reference={name:value.to(self.device) for name,value in reference.items()}
   if self.frozen_actor_sha256()!=state.get("frozen_base_actor_sha256"):raise RuntimeError("FBMR frozen baseline actor hash mismatch")
  reference_state=state.get("anchor_reference_actor_state")
  if self.anchor.enabled:
   if reference_state is None:raise RuntimeError("policy-anchor checkpoint is not self-contained")
   reference=deepcopy(self.actor).to(self.device);reference.load_state_dict(reference_state);self.anchor.attach(reference,self.anchor_provenance.get("reference_checkpoint"))
  for key,attr in (("ppo_updates","ppo_update_count"),("actor_updates","actor_update_count"),("critic_updates","critic_update_count"),("sampled_steps","sampled_steps"),("vector_steps","vector_steps")):setattr(self,attr,int(state.get(key,0)))
  self.kl_hard_stop_count=int(state.get("kl_hard_stop_count",0))
  self.actor_kl_guard_hard_stop_count=int(state.get("actor_kl_guard_hard_stop_count",0))
  self.actor_kl_guard_actor_epochs_total=int(state.get("actor_kl_guard_actor_epochs_total",0))
  self.actor_kl_guard_actor_epochs_min=state.get("actor_kl_guard_actor_epochs_min")
  if self.sequential_wave_gradient_projection.enabled:
   self.sequential_wave_gradient_projection.load_state_dict(state.get("sequential_wave_gradient_projection_state"),branch_from_plain=not strict_protocol)
  if self.persistent_wave_trajectory_replay.enabled:
   self.persistent_wave_trajectory_replay.load_state_dict(state.get("persistent_wave_trajectory_replay_state"),branch_from_plain=not strict_protocol)
  if self.wave1_sensitivity_gating.enabled:
   self.wave1_sensitivity_gating.load_state_dict(state.get("wave1_sensitivity_gating_state"),branch_from_plain=not strict_protocol)
  if self.wave_specific_actor_isolation.enabled:
   self.wave_specific_actor_isolation.load_state_dict(state.get("wave_specific_actor_isolation_state"),branch_from_plain=not strict_protocol)
  if self.wave_specific_mean_heads.enabled:
   self.wave_specific_mean_heads.load_state_dict(state.get("wave_specific_mean_heads_state"),branch_from_plain=not strict_protocol)
  if self.actor_gradient_clipping.enabled:
   self.actor_gradient_clipping.load_state_dict(state.get("actor_gradient_clipping_state"),branch_from_plain=not strict_protocol)
  if self.deployment_aligned_wave_exploration.enabled:
   self.deployment_aligned_wave_exploration.load_state_dict(state.get("deployment_aligned_wave_exploration_state"),branch_from_plain=not strict_protocol)
  if self.reference_variance.enabled:
   self.reference_variance.load_state_dict(state.get("reference_variance_state"),branch_from_plain=not strict_protocol)
  if self.milestone_aware_retention_credit.enabled:
   self.milestone_aware_retention_credit.load_state_dict(
    state.get("milestone_aware_retention_credit_state"),strict=strict_protocol)
  if self.actor_lr_decay.enabled:
   baseline_lr=self.actor_lr_decay.apply(self.actor_optimizer,self.sampled_steps,self.base_actor_learning_rate)
   if self.wave_specific_actor_isolation.enabled:
    self.actor_lr_decay.apply(self.wave2_actor_optimizer,self.sampled_steps,self.base_actor_learning_rate)
    self.actor_lr_decay.apply(self.wave3_actor_optimizer,self.sampled_steps,self.base_actor_learning_rate)
   if self.wave_specific_mean_heads.enabled:
    self.actor_lr_decay.apply(self.wave2_mean_optimizer,self.sampled_steps,self.base_actor_learning_rate)
    self.actor_lr_decay.apply(self.wave3_mean_optimizer,self.sampled_steps,self.base_actor_learning_rate)
   if self.hierarchical_temporal_abstraction.enabled:self.actor_lr_decay.apply(self.manager_actor_optimizer,self.sampled_steps,self.base_actor_learning_rate)
   if self.hta_worker_consolidation.enabled:self.hta_worker_consolidation.apply(self.actor_optimizer,self.sampled_steps,baseline_lr)
  elif self.ppo_stabilization.enabled:
   lr=self.ppo_stabilization.actor_learning_rate(self.sampled_steps,self.total_sampled_steps)
   for group in self.actor_optimizer.param_groups:group["lr"]=lr
  if restore_rng:self.restore_rng_state(state)
  else:self.rng_restore_metadata={"rng_state_available":isinstance(state.get("rng_state"),dict),"rng_state_restored":False,"cuda_rng_state_restored":False}
  return state.get("extra",{})

 def load_fbmr_branch(self,path,source_checkpoint_sha256,restore_rng=True):
  """Create an FBMR Stage-2 continuation without loading the source actor optimizer."""
  if not self.fbmr_enabled:raise RuntimeError("load_fbmr_branch requires a frozen-base mean-residual mode")
  digest=hashlib.sha256()
  with Path(path).open("rb") as stream:
   for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
  if digest.hexdigest()!=source_checkpoint_sha256:raise RuntimeError("FBMR source checkpoint hash changed before load")
  state=torch.load(path,map_location=self.device,weights_only=False)
  source_actor=state.get("actor",{});expected={name for name,_ in self.actor.frozen_baseline_named_parameters()}
  if set(source_actor)!=expected:raise RuntimeError("FBMR source actor is not the exact baseline topology")
  current=self.actor.state_dict()
  for name in expected:current[name]=source_actor[name]
  self.actor.load_state_dict(current,strict=True);self.actor.freeze_baseline_policy();self.capture_frozen_actor_reference()
  self.critic.load_state_dict(state["critic"],strict=True);self.critic_optimizer.load_state_dict(state["critic_optimizer"])
  self.popart.load_state_dict(state.get("popart",{}),strict=False)
  self.warm_start_provenance=state.get("warm_start_provenance",{});self.anchor_provenance=state.get("anchor_provenance",{})
  for key,attr in (("ppo_updates","ppo_update_count"),("actor_updates","actor_update_count"),("critic_updates","critic_update_count"),("sampled_steps","sampled_steps"),("vector_steps","vector_steps")):setattr(self,attr,int(state.get(key,0)))
  self.kl_hard_stop_count=int(state.get("kl_hard_stop_count",0))
  extra=state.get("extra",{});intervention=self.actor.entity_attention_mode;self.fbmr_branch_metadata={
   "branch_intervention":intervention,"actor_optimizer_restore":False,
   "actor_optimizer_reason":"new_trainable_parameter_set","critic_optimizer_restore":True,
   "critic_optimizer_restored":True,"source_rng_restored":False,"RNG_restored_from_source":False,
   "base_actor_loaded_exact":True,
   "source_checkpoint_sha256":source_checkpoint_sha256,"source_sampled_steps":int(state["sampled_steps"]),
   "source_training_seed":int(extra["training_seed"]),"actor_optimizer_reset_for_new_params":True,
  }
  if intervention=="frozen_base_dual_bounded_mean_residual":
   self.fbmr_branch_metadata.update({"dual_bound_enabled":True,"alpha_abs":self.actor.alpha_abs,"alpha_rel":self.actor.alpha_rel})
  if restore_rng:
   restored=self.restore_rng_state(state);self.fbmr_branch_metadata["source_rng_restored"]=bool(restored);self.fbmr_branch_metadata["RNG_restored_from_source"]=bool(restored)
  else:self.rng_restore_metadata={"rng_state_available":isinstance(state.get("rng_state"),dict),"rng_state_restored":False,"cuda_rng_state_restored":False}
  return extra

__all__=["MODULAR_MAPPO_IMPL_VERSION","ModularMAPPOTrainer"]
