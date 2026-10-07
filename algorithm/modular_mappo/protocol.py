"""Canonical fingerprints and strict formal-checkpoint validation."""
from copy import deepcopy
import hashlib,json
from env.config import ENVIRONMENT_VERSION
from algorithm.common.protocol import config_sha256
from algorithm.mappo.trainer import MAPPO_IMPL_VERSION
from .trainer import MODULAR_MAPPO_IMPL_VERSION
def canonical_sha256(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()

MARC_FACTORIAL_CELLS={"none":(0.0,0.0),"credit_only":(1.0,0.0),"balance_only":(0.0,0.5),"full":(1.0,0.5)}

JIAO_3M_METHODS={"jiao2025_matched_plain_3m","jiao2025_core_3m"}
JIAO_3M_SOURCE_CLASSIFICATION={
 "paper_specified":["scalar round F","recurrent Actor","recurrent value network","trajectory/chunk training","PopArt","LR=5e-4","PPO epochs=10","clip=.1","entropy=.01","lambda=.95","gamma=.99"],
 "project_adaptations":["433 3-D environment","3-D tanh-Gaussian action","width256","GRU128","BPTT32","rollout256","24 environments","minibatch512","centralized attention Critic","PopArt beta=.999 epsilon=1e-5","3M budget","44M development bank","GAE+value lambda-return target","true-episode-only hidden reset"],
 "source_ambiguities":["centralized/global Critic notation","Algorithm1 reward-to-go vs project lambda-return"]}

def validate_jiao2025_3m_config(config,env=None):
 """Fail closed on non-package differences in the new Jiao transfer recipe."""
 if config.get("development_method") not in JIAO_3M_METHODS:return
 from pathlib import Path
 from algorithm.train_modular_mappo import load_config
 root=Path(__file__).resolve().parents[2]
 expected=load_config(root/"configs/diag_mappo_learnability_common_3m.yaml")
 expected["development_method"]=config["development_method"]
 expected["training"].update(actor_learning_rate=.0005,critic_learning_rate=.0005,gamma=.99,clip_ratio=.1)
 # The runtime smoke uses diagnostic seeds; formal preflight fixes seed5301.
 expected["training"]["seed"]=config["training"]["seed"]
 expected["modules"]["actor_lr_decay"]["enabled"]=False
 expected["modules"]["actor_kl_guard"]={"enabled":False}
 for name in ("actor_gradient_clipping","mission_film","inter_wave_credit","counterfactual_inter_wave_credit",
              "boundary_redistributed_segment_credit","hierarchical_temporal_abstraction","hta_worker_consolidation",
              "sequential_wave_gradient_projection","team_mean_credit","persistent_wave_trajectory_replay",
              "wave_entry_curriculum","wave1_sensitivity_gating","wave_specific_actor_isolation","wave_specific_mean_heads",
              "deployment_aligned_wave_exploration","reference_variance"):
  expected["modules"][name]={"enabled":False}
 expected["modules"]["milestone_aware_retention_credit"]={"enabled":False,"version":2,"deployment_distill_coefficient":0.0}
 expected["jiao_transfer"]={"source_doi":"10.1049/cth2.12781","role":"paper_aligned_transfer_not_exact_simulator_reproduction","primary_result":"exact_3000000_final_evaluation"}
 if config["development_method"]=="jiao2025_core_3m":
  expected["modules"]["wave_context"]={"enabled":True,"context_target":"actor_critic","encoding":"scalar_round","max_waves":3}
  expected["modules"]["recurrent_memory"]={"enabled":True,"mode":"actor_critic_gru","hidden_dim":128,"sequence_length":32}
  expected["modules"]["popart"]={"enabled":True,"beta":.999,"epsilon":.00001}
 if config!=expected:raise ValueError("Jiao 3M config differs outside the registered scalar-F/Actor+Critic-GRU/PopArt package")
 if env is not None and env!=load_config(root/"configs/persistent_wave_v2_blue433_environment.yaml"):
  raise ValueError("Jiao 3M requires the unchanged frozen 433 environment/reward/Blue/weapon")

def validate_marc_factorial_config(config,env=None):
 """Strict scalar-only ablation of the existing feed-forward MARC lineage."""
 if config.get("development_method")!="marc_core_factorial_v1":return
 from pathlib import Path
 from algorithm.train_modular_mappo import load_config
 root=Path(__file__).resolve().parents[2]
 cell=config.get("factorial_variant")
 if cell not in MARC_FACTORIAL_CELLS:raise ValueError("unknown MARC factorial cell")
 source=load_config(root/"configs/dev_marc_credit_balance_1m.yaml")
 expected=deepcopy(source);expected["development_method"]="marc_core_factorial_v1"
 expected["factorial_variant"]=cell;expected["training"]["total_sampled_steps"]=3000000
 # Runtime training seed is deliberately matched, not fixed to a single replicate.
 expected["training"]["seed"]=config["training"]["seed"]
 alpha,temp=MARC_FACTORIAL_CELLS[cell]
 expected["modules"]["milestone_aware_retention_credit"].update(continuation_alpha=alpha,wave_balance_temperature=temp)
 expected["development_protocol"]["primary_checkpoints"]=[900000,3000000]
 if config!=expected:raise ValueError("MARC factorial differs outside the four registered scalar cells/metadata/budget")
 if env is not None and env!=load_config(root/"configs/persistent_wave_v2_blue433_environment.yaml"):
  raise ValueError("MARC factorial requires unchanged frozen 433 environment including reward/Blue/weapon")

def validate_marc_gru_screen_config(config):
 method=config.get("development_method")
 if method not in {"marc_credit_balance_ablation","marc_mappo_wsgru_v1"}:return
 modules=config.get("modules",{});treatment=method=="marc_mappo_wsgru_v1"
 enabled={k for k,v in modules.items() if isinstance(v,dict) and v.get("enabled",False)}
 expected={"actor_lr_decay","milestone_aware_retention_credit"}|({"recurrent_memory"} if treatment else set())
 if enabled!=expected:raise ValueError(f"{method} requires exact enabled modules: {sorted(expected)}")
 marc=modules["milestone_aware_retention_credit"]
 constants={"version":2,"max_waves":3,"continuation_alpha":1.0,"wave_balance_temperature":.5,
  "wave_weight_min":.5,"wave_weight_max":2.0,"deployment_distill_coefficient":0.0}
 if any(marc.get(k)!=v for k,v in constants.items()):raise ValueError("MARC GRU screen credit/balance constants or zero retention mismatch")
 if treatment:
  recurrent=modules["recurrent_memory"]
  if any(recurrent.get(k)!=v for k,v in {"mode":"wave_segmented_actor_gru","hidden_dim":128,"sequence_length":32}.items()):
   raise ValueError("MARC GRU screen requires wave_segmented_actor_gru, hidden_dim=128, sequence_length=32")
 network=config["network"]
 if any(network.get(k)!=v for k,v in {"observation_dim":52,"action_dim":3,"num_agents":4,"actor_hidden_layers":[256,256],"critic_hidden_layers":[256,256],"attention_heads":2}.items()):
  raise ValueError("MARC GRU screen requires unchanged 52D/3D/4-agent 256-MLP attention-critic architecture")

def validate_marc_gru_screen_environment(config,env):
 if config.get("development_method") not in {"marc_credit_balance_ablation","marc_mappo_wsgru_v1","marc_mappo_state_memory_v1"}:return
 identity=(env.get("environment_variant"),env.get("persistent_waves",{}).get("blue_units_per_wave"),
  env.get("persistent_waves",{}).get("total_waves"),env.get("scenario",{}).get("team_size"),env.get("simulation",{}).get("max_steps"))
 if identity!=("persistent_wave_v2",[4,3,3],3,4,3000):raise ValueError("MARC GRU screen requires frozen 433/three-wave/3000-step environment")

def validate_marc_state_memory_config(config):
 if config.get("development_method")!="marc_mappo_state_memory_v1":return
 shadow=deepcopy(config);shadow["development_method"]="marc_credit_balance_ablation"
 shadow["modules"]["recurrent_memory"]["enabled"]=False
 validate_marc_gru_screen_config(shadow)
 expected={"enabled":True,"mode":"wave_state_memory_actor_gru","hidden_dim":128,
           "sequence_length":32,"wave_context_dim":3,"initial_hidden":"zeros"}
 if config["modules"].get("recurrent_memory")!=expected:
  raise ValueError("MARC-SM requires isolated 55D State Memory GRU128/BPTT32/3D context/literal zero initialization")
 if config["implementation"]["actor_activation"]!="relu":raise ValueError("MARC-SM requires existing ReLU policy")
 if int(config["training"]["ppo_epochs"])!=10:raise ValueError("MARC-SM requires all 10 PPO epochs")
def checkpoint_architecture(trainer):
 mode=trainer.actor.entity_attention_mode if trainer.actor.entity_attention_enabled else "disabled"
 fbmr=mode in {"frozen_base_mean_residual","frozen_base_dual_bounded_mean_residual"};dual=mode=="frozen_base_dual_bounded_mean_residual"
 critic_input_dim=trainer.critic.embedding[0].in_features
 result={"actor_class":type(trainer.actor).__name__,"critic_class":type(trainer.critic).__name__,"actor_input_dim":trainer.actor.backbone[0].in_features if hasattr(trainer.actor,"backbone") else trainer.actor.base_observation_dim+trainer.actor.context_dim,"critic_input_dim":critic_input_dim,"actor_context_dim":trainer.actor.context_dim,"critic_context_dim":trainer.critic.context_dim,"critic_context_injection":trainer.critic.context_injection,"wave_context_encoding":trainer.wave_context.encoding if trainer.wave_context.enabled else "disabled","wave_context_target":trainer.wave_context.target if trainer.wave_context.enabled else "disabled","actor_parameter_count":sum(parameter.numel() for parameter in trainer.actor.parameters()),"critic_parameter_count":sum(parameter.numel() for parameter in trainer.critic.parameters()),"hidden_dim":256 if trainer.actor.entity_attention_enabled else trainer.actor.backbone[0].out_features,"actor_gru_hidden_dim":trainer.actor.recurrent_hidden_dim,"critic_gru_hidden_dim":trainer.critic.recurrent_hidden_dim,"entity_attention_enabled":trainer.actor.entity_attention_enabled,"entity_attention_mode":mode,"entity_dim":trainer.actor.entity_dim if trainer.actor.entity_attention_enabled else 0,"entity_attention_heads":trainer.actor.entity_attention_heads if trainer.actor.entity_attention_enabled else 0,"base_actor_frozen":fbmr,"entity_mean_residual_enabled":fbmr,"max_mean_correction":trainer.actor.max_mean_correction if mode=="frozen_base_mean_residual" else 0.,"dual_bound_enabled":dual,"alpha_abs":trainer.actor.alpha_abs if dual else 0.,"alpha_rel":trainer.actor.alpha_rel if dual else 0.,"log_std_source":"frozen_baseline" if fbmr else "actor_head"}
 if trainer.mission_film.enabled:
  result.update({"mission_film_enabled":True,"mission_film_mode":trainer.mission_film.mode,
   "mission_encoder_hidden_dim":trainer.mission_film.encoder_hidden_dim,"mission_film_alpha":trainer.mission_film.alpha,
   "mission_film_augmented_residual":trainer.mission_film.augmented_residual,
   "mission_film_identity_init":trainer.mission_film.identity_init})
 if trainer.inter_wave_credit.enabled:
  result.update({"inter_wave_credit_enabled":True,"inter_wave_credit_version":trainer.inter_wave_credit.version,
   "iw_critic_class":type(trainer.iw_critic).__name__,"iw_critic_input_dim":56,
   "iw_critic_parameter_count":sum(parameter.numel() for parameter in trainer.iw_critic.parameters()),
   "iw_target_definition":{"wave1":"(clear_wave2 + clear_wave3) / 2","wave2":"clear_wave3","wave3":None},
   "iw_actor_credit":"q_next - q_current (no gamma, stop-gradient)",
   "iw_gradient_fusion":"asymmetric_tactical_preserving_projection"})
 if trainer.counterfactual_inter_wave_credit.enabled:
  module=trainer.counterfactual_inter_wave_credit
  result.update({"counterfactual_inter_wave_credit_enabled":True,"counterfactual_inter_wave_credit_version":module.version,
   "caiw_critic_class":type(trainer.caiw_critic).__name__,"caiw_state_observation_dim":52,"caiw_action_dim":3,
   "caiw_event_heads":["wave2_clear","wave3_clear"],"caiw_training_targets":{"w1_to_w2":"C2","w1_to_w3":"C3","w2_to_w3":"C3"},
   "caiw_critic_loss":"BCEWithLogits","caiw_class_sampling":"task_balanced","caiw_probability_correction":"recent_natural_prior_logit_correction",
   "caiw_actor_credit":"continuous_per_agent_counterfactual_probability_advantage","caiw_advantage_normalization":"none",
   "caiw_counterfactual_samples":module.counterfactual_samples,"caiw_gradient_projection":"asymmetric_tactical_preserving",
   "caiw_gradient_ratio_cap":module.auxiliary_gradient_ratio_cap,"caiw_readiness":"held_out_AUROC+BrierSkill+coverage+3_consecutive_passes"})
 if trainer.boundary_redistributed_segment_credit.enabled:
  module=trainer.boundary_redistributed_segment_credit
  result.update({"boundary_redistributed_segment_credit_enabled":True,"boundary_redistributed_segment_credit_version":module.version,
   "boundary_state_critic_class":type(trainer.brsc_critic).__name__,"boundary_tasks":["W1_boundary_to_W2","W1_boundary_to_W3","W2_boundary_to_W3"],
   "post_spawn_entry_state":True,"action_input":False,"critic_loss":"BCEWithLogits",
   "prior_correction":"train_split_natural_prior","readiness":"held_out_AUROC+BSS+coverage+3passes",
   "actor_credit":"boundary_quality_minus_prior","redistribution":"gamma_lambda_backward_within_current_rollout_segment",
   "single_step_counterfactual":False,"advantage_normalization":"none",
   "gradient_projection":"asymmetric_tactical_preserving","gradient_ratio_cap":module.auxiliary_gradient_ratio_cap,
   "deployment_actor_unchanged":True})
 if trainer.hierarchical_temporal_abstraction.enabled:
  module=trainer.hierarchical_temporal_abstraction
  result.update({"hierarchical_temporal_abstraction_enabled":True,"hta_version":module.version,
   "manager_actor_class":type(trainer.manager_actor).__name__,"manager_critic_class":type(trainer.manager_critic).__name__,
   "manager_observation_dim":52,"num_options":module.num_options,"decision_interval_steps":module.decision_interval_steps,
   "manager_recurrent":False,"worker_observation_dim":52,"worker_action_dim":3,
   "worker_option_conditioning":"zero_initialized_option_mean_residual","worker_logstd":"shared_plain_logstd",
   "tactical_critic_option_conditioning":"additive_zero_onehot",
   "manager_reward":"discounted_raw_environment_reward","manager_return":"semi_mdp_discounted_return",
   "manager_gae":"gamma_power_duration","wave_transition_terminal":False,
   "rollout_boundary_macro_truncation":True,"rollout_boundary_bootstrap":True,
   "rollout_boundary_trace_continuation":False,"manager_worker_parameter_sharing":False,
   "counterfactual":False,"auxiliary_credit_redistribution":False,"reward_shaping":False,
   "deployment_requires_manager":True,"manager_actor_parameter_count":sum(p.numel() for p in trainer.manager_actor.parameters()),
   "manager_critic_parameter_count":sum(p.numel() for p in trainer.manager_critic.parameters())})
  if trainer.hta_worker_consolidation.enabled:
   consolidation=trainer.hta_worker_consolidation
   result.update({"hta_worker_consolidation_enabled":True,
    "hta_worker_consolidation_version":consolidation.version,
    "worker_learning_timescale":"progressive_linear_lr_multiplier",
    "worker_consolidation_start_step":consolidation.start_step,
    "worker_consolidation_end_step":consolidation.end_step,
    "worker_final_lr_multiplier":consolidation.final_lr_multiplier,
    "manager_learning_schedule":"unchanged_actor_lr_decay",
    "tactical_critic_learning_schedule":"unchanged_constant"})
 if trainer.wave_specific_actor_isolation.enabled:
  result.update({"wave_specific_actor_isolation_enabled":True,"wsai_version":1,
   "wave_actor_count":3,"wave_actor_parameter_sharing":False,"wave_actor_routing":"environment_wave","shared_critic":True,
   "natural_wave_weighting":True,"global_actor_grad_clip":True,
   "actor1_trainable_parameter_count":sum(p.numel() for p in trainer.actor.trainable_policy_parameters()),
   "actor2_trainable_parameter_count":sum(p.numel() for p in trainer.wave2_actor.trainable_policy_parameters()),
   "actor3_trainable_parameter_count":sum(p.numel() for p in trainer.wave3_actor.trainable_policy_parameters()),
   "total_actor_parameter_count":sum(sum(p.numel() for p in actor.parameters()) for actor in trainer._wsai_actors())})
 if trainer.wave_specific_mean_heads.enabled:
  heads=trainer._wsmh_means();base=sum(p.numel() for p in trainer.actor.parameters());additional=sum(p.numel() for head in heads[1:] for p in head.parameters())
  result.update({"wave_specific_mean_heads_enabled":True,"wsmh_version":1,"wave_mean_head_count":3,
   "shared_backbone":True,"shared_log_std":True,"shared_critic":True,"wave_actor_routing":"environment_wave",
   "mean1_parameter_count":sum(p.numel() for p in heads[0].parameters()),"mean2_parameter_count":sum(p.numel() for p in heads[1].parameters()),
   "mean3_parameter_count":sum(p.numel() for p in heads[2].parameters()),"base_actor_parameter_count":base,
   "additional_mean_parameter_count":additional,"additional_wave_mean_parameters":additional,
   "total_policy_parameter_count":base+additional,"parameter_overhead_fraction":additional/base})
 if trainer.actor_gradient_clipping.enabled:
  module=trainer.actor_gradient_clipping
  result.update({"actor_gradient_clipping_enabled":True,"actor_grad_clip_version":module.version,
   "actor_grad_clip_mode":module.mode,"actor_max_grad_norm":module.actor_max_grad_norm,
   "critic_max_grad_norm":trainer.max_grad_norm,"actor_clip_only_intervention":True,"network_topology_unchanged":True})
 if trainer.recurrent.wave_segmented:
  result.update({"recurrent_mode":trainer.recurrent.mode,"actor_phase_embedding_shape":[3,128],
   "actor_hidden_lifecycle":"episode/wave pre-action live phase initialization; death zero; rollout carry",
   "actor_bptt_boundary":"explicit episode/wave phase reset; maximum 32 steps"})
 if trainer.milestone_aware_retention_credit.enabled:
  module=trainer.milestone_aware_retention_credit
  result.update({"milestone_aware_retention_credit_enabled":True,"marc_version":module.version,
   "actor_observation_dim":52,"actor_wave_input":bool(trainer.recurrent.state_memory),
   "credit_decomposition":"global_GAE=local_GAE+continuation_advantage",
   "local_trace_stops_at_wave_transition":True,"td_bootstrap_at_wave_transition":True,
   "continuation_alpha":module.continuation_alpha,
   "wave_balance_target":"actor_surrogate_only","wave_balance_temperature":module.wave_balance_temperature,
   "wave_weight_min":module.wave_weight_min,"wave_weight_max":module.wave_weight_max,
   "critic_target":"unchanged_full_horizon_return","entropy_weighting":"plain_alive_mean",
   "retention_target":("successful_milestone_policy_snapshot" if module.version==1 else
                       "elite_successful_executed_action_to_deterministic_tanh_mean"),
   "retention_samples_per_wave":module.retention_samples_per_wave,
   "retention_stride":module.retention_stride})
  if module.version==1:
   result.update({"retention_coefficient":module.retention_coefficient,
    "retention_bank_size_per_wave":module.retention_bank_size_per_wave,
    "retention_min_samples_per_wave":module.retention_min_samples_per_wave,
    "retention_distribution":"diagonal_gaussian_kl"})
  else:
   result.update({"deployment_distill_coefficient":module.deployment_distill_coefficient,
    "elite_segments_per_wave":module.elite_segments_per_wave,
    "elite_rows_per_segment":module.elite_rows_per_segment,
    "retention_min_rows_per_wave":module.retention_min_rows_per_wave,
    "elite_quality":"lexicographic(red_survivors_after_clear,-wave_duration_steps)",
    "elite_replacement":"strictly_better_replaces_current_worst_equal_rejected",
    "deployment_target":"post_tanh_action_actually_executed_in_environment",
    "log_std_retention":False})
 if trainer.recurrent.state_memory:
  result.update({"actor_input_dim":55,"actor_context_dim":3,"wave_context_encoding":"fixed_one_hot_3",
   "wave_context_target":"actor_only","recurrent_mode":trainer.recurrent.mode,
   "state_memory_input_dim":55,"fused_policy_input_dim":183,
   "actor_memory_initialization":"literal_zeros","actor_phase_embedding_shape":None,
   "actor_hidden_lifecycle":"zero_episode_start_zero_wave_start_dead_zero_rollout_carry",
   "bptt_boundaries":"episode_and_wave","retention_target":"disabled","deployment_retention_active":False})
 return result

def _validate_embedded_disabled_curriculum_runtime(extra,algorithm_config):
 source=extra.get("environment_config")
 curriculum=extra.get("curriculum_config",algorithm_config.get("modules",{}).get("curriculum",{}))
 if (not bool(curriculum.get("enabled",False)) and isinstance(source,dict)
     and extra.get("current_total_waves") is not None
     and int(extra["current_total_waves"])!=int(source.get("persistent_waves",{}).get("total_waves",1))):
  raise RuntimeError("checkpoint declared/runtime wave mismatch under disabled curriculum")

def validate_modular_checkpoint(state,env_config,algorithm_config,expected_runtime=None):
 if state.get("algorithm")!="modular_mappo":raise RuntimeError("checkpoint algorithm mismatch")
 checkpoint_version=state.get("modular_mappo_impl_version")
 if checkpoint_version!=MODULAR_MAPPO_IMPL_VERSION:raise RuntimeError(f"modular implementation version mismatch: checkpoint={checkpoint_version}, current={MODULAR_MAPPO_IMPL_VERSION}")
 if state.get("baseline_mappo_impl_version")!=MAPPO_IMPL_VERSION:raise RuntimeError("baseline MAPPO implementation version mismatch")
 extra=state.get("extra",{})
 source=extra.get("environment_config")
 curriculum_config=extra.get("curriculum_config",algorithm_config.get("modules",{}).get("curriculum",{}))
 curriculum_enabled=bool(curriculum_config.get("enabled",False))
 # Backward-compatible forensic guard: old checkpoints did not carry runtime
 # hashes, but did carry both the declared environment and the actual wave count.
 _validate_embedded_disabled_curriculum_runtime(extra,algorithm_config)
 new_keys=("declared_environment_config_sha256","declared_total_waves","declared_max_steps",
           "effective_training_environment_config_sha256","effective_training_total_waves","effective_training_max_steps",
           "runtime_environment_config_sha256","runtime_total_waves","runtime_max_steps",
           "evaluation_environment_config_sha256","evaluation_total_waves","evaluation_max_steps",
           "curriculum_enabled")
 if any(key in extra for key in new_keys):
  missing=[key for key in new_keys if key not in extra]
  if missing:raise RuntimeError(f"checkpoint runtime environment provenance incomplete: {missing}")
  if not isinstance(source,dict):raise RuntimeError("checkpoint runtime provenance lacks declared environment_config")
  runtime_config=extra.get("runtime_environment_config")
  if not isinstance(runtime_config,dict):raise RuntimeError("checkpoint runtime provenance lacks runtime_environment_config")
  source_waves=int(source.get("persistent_waves",{}).get("total_waves",1));source_steps=int(source["simulation"]["max_steps"])
  runtime_waves=int(runtime_config.get("persistent_waves",{}).get("total_waves",1));runtime_steps=int(runtime_config["simulation"]["max_steps"])
  strict={"declared_environment_config_sha256":config_sha256(source),"declared_total_waves":source_waves,
          "declared_max_steps":source_steps,"runtime_environment_config_sha256":config_sha256(runtime_config),
          "runtime_total_waves":runtime_waves,"runtime_max_steps":runtime_steps,
          "effective_training_environment_config_sha256":config_sha256(runtime_config),
          "effective_training_total_waves":runtime_waves,"effective_training_max_steps":runtime_steps,
          "evaluation_environment_config_sha256":config_sha256(source),"evaluation_total_waves":source_waves,
          "evaluation_max_steps":source_steps,"curriculum_enabled":curriculum_enabled}
  for key,expected in strict.items():
   if extra.get(key)!=expected:raise RuntimeError(f"checkpoint {key} mismatch: expected {expected!r}, got {extra.get(key)!r}")
  if not curriculum_enabled and config_sha256(runtime_config)!=config_sha256(source):
   raise RuntimeError("checkpoint declared/runtime environment mismatch under disabled curriculum")
 checks={
  "environment_version":str(env_config.get("environment_version",ENVIRONMENT_VERSION)),
  "environment_variant":str(env_config.get("environment_variant","direct_v2_3")),
  "environment_config_sha256":config_sha256(env_config),
  "algorithm_config_sha256":config_sha256(algorithm_config),
 }
 for key,expected in checks.items():
  if extra.get(key)!=expected:raise RuntimeError(f"checkpoint {key} mismatch: expected {expected!r}, got {extra.get(key)!r}")
 module_hash=canonical_sha256(algorithm_config.get("modules",{}))
 if state.get("module_config_sha256")!=module_hash:raise RuntimeError("checkpoint module config mismatch")
 network=algorithm_config["network"]
 for key,expected in (("observation_dim",int(network["observation_dim"])),("action_dim",int(network["action_dim"])),("num_agents",int(network["num_agents"]))):
  if int(extra.get(key,-1))!=expected:raise RuntimeError(f"checkpoint {key} mismatch")
 if expected_runtime:
  for key,expected in expected_runtime.items():
   if extra.get(key)!=expected:raise RuntimeError(f"checkpoint {key} mismatch")
 if extra.get("network_architecture") is None:raise RuntimeError("checkpoint lacks network architecture")
 pop_enabled=bool(algorithm_config.get("modules",{}).get("popart",{}).get("enabled",False))
 if pop_enabled != bool(state.get("module_config",{}).get("popart",{}).get("enabled",False)):raise RuntimeError("checkpoint PopArt protocol mismatch")
 return True

def _branch_comparable_config(config):
 value=deepcopy(config);value.get("training",{}).pop("total_sampled_steps",None)
 value.get("modules",{}).pop("actor_lr_decay",None)
 return value

def validate_modular_branch(state,env_config,algorithm_config,expected_runtime=None):
 """Allow only an explicit actor_lr_decay intervention at a branch boundary."""
 if state.get("algorithm")!="modular_mappo":raise RuntimeError("branch checkpoint algorithm mismatch")
 if state.get("modular_mappo_impl_version")!=MODULAR_MAPPO_IMPL_VERSION:raise RuntimeError("branch modular implementation version mismatch")
 if state.get("baseline_mappo_impl_version")!=MAPPO_IMPL_VERSION:raise RuntimeError("branch baseline MAPPO implementation version mismatch")
 extra=state.get("extra",{});_validate_embedded_disabled_curriculum_runtime(extra,algorithm_config);source_env=extra.get("environment_config");source_algorithm=extra.get("algorithm_config")
 if not isinstance(source_env,dict) or not isinstance(source_algorithm,dict):raise RuntimeError("branch checkpoint lacks self-describing source configs")
 if source_env!=env_config:raise RuntimeError("branch environment config differs from source checkpoint")
 if extra.get("environment_config_sha256")!=config_sha256(env_config):raise RuntimeError("branch environment hash mismatch")
 if _branch_comparable_config(source_algorithm)!=_branch_comparable_config(algorithm_config):raise RuntimeError("branch config differs outside the actor_lr_decay/total_sampled_steps whitelist")
 source_decay=source_algorithm.get("modules",{}).get("actor_lr_decay",{})
 destination_decay=algorithm_config.get("modules",{}).get("actor_lr_decay",{})
 if bool(source_decay.get("enabled",False)) and source_decay!=destination_decay:raise RuntimeError("branch cannot alter an already-enabled actor_lr_decay protocol")
 if expected_runtime:
  for key,expected in expected_runtime.items():
   if extra.get(key)!=expected:raise RuntimeError(f"branch checkpoint {key} mismatch: expected {expected!r}, got {extra.get(key)!r}")
 if extra.get("network_architecture") is None:raise RuntimeError("branch checkpoint lacks network architecture")
 required=("actor","critic","actor_optimizer","critic_optimizer","sampled_steps","vector_steps","module_config_sha256")
 missing=[key for key in required if key not in state]
 if missing:raise RuntimeError("branch checkpoint lacks required state: "+", ".join(missing))
 return {"intervention":"actor_lr_decay" if source_decay!=destination_decay else "fixed_lr_control","source_actor_lr_decay":deepcopy(source_decay),"destination_actor_lr_decay":deepcopy(destination_decay)}

def _swgp_branch_comparable_config(config):
 value=deepcopy(config);value.pop("development_method",None);value.pop("development_branch",None)
 value.get("training",{}).pop("total_sampled_steps",None)
 value.get("modules",{}).pop("sequential_wave_gradient_projection",None)
 return value

def validate_swgp_branch(state,env_config,algorithm_config,expected_runtime=None):
 """Validate a matched Plain checkpoint continuation into Control or SWGP V1."""
 if state.get("algorithm")!="modular_mappo":raise RuntimeError("SWGP branch source must be modular_mappo")
 if state.get("modular_mappo_impl_version")!=MODULAR_MAPPO_IMPL_VERSION or state.get("baseline_mappo_impl_version")!=MAPPO_IMPL_VERSION:raise RuntimeError("SWGP branch source implementation mismatch")
 extra=state.get("extra",{});source_env=extra.get("environment_config");source=extra.get("algorithm_config")
 if not isinstance(source_env,dict) or not isinstance(source,dict):raise RuntimeError("SWGP branch source lacks embedded configs")
 if source_env!=env_config or extra.get("environment_config_sha256")!=config_sha256(env_config):raise RuntimeError("SWGP branch environment mismatch")
 if _swgp_branch_comparable_config(source)!=_swgp_branch_comparable_config(algorithm_config):raise RuntimeError("SWGP branch differs outside the intervention whitelist")
 source_enabled=set(state.get("enabled_modules",[]));destination_enabled=set(name for name,value in algorithm_config.get("modules",{}).items() if isinstance(value,dict) and value.get("enabled",False))
 if source_enabled!={"actor_lr_decay"}:raise RuntimeError("SWGP branch source must be exact Plain actor_lr_decay")
 allowed=({"actor_lr_decay"},{"actor_lr_decay","sequential_wave_gradient_projection"})
 if destination_enabled not in allowed:raise RuntimeError("SWGP destination enabled modules mismatch")
 intervention=algorithm_config.get("development_branch",{}).get("intervention")
 branch=algorithm_config.get("development_branch",{})
 expected=("sequential_wave_gradient_projection" if "sequential_wave_gradient_projection" in destination_enabled else "sequential_wave_gradient_projection_control")
 if intervention!=expected:raise RuntimeError("SWGP branch intervention/config mismatch")
 if int(branch.get("source_training_seed",-1))!=int(extra.get("training_seed",-2)):raise RuntimeError("SWGP branch source training seed mismatch")
 if int(branch.get("source_sampled_steps",-1))!=int(state.get("sampled_steps",-2)):raise RuntimeError("SWGP branch source sampled_steps mismatch")
 if expected_runtime:
  for key,value in expected_runtime.items():
   if extra.get(key)!=value:raise RuntimeError(f"SWGP branch checkpoint {key} mismatch")
 required=("actor","critic","actor_optimizer","critic_optimizer","rng_state","sampled_steps","ppo_updates","actor_updates","critic_updates")
 missing=[key for key in required if key not in state]
 if missing:raise RuntimeError("SWGP branch source lacks required state: "+", ".join(missing))
 return {"intervention":intervention,"source_enabled_modules":sorted(source_enabled),"destination_enabled_modules":sorted(destination_enabled),"matched_causal_continuation":True,"historical_plain_bitwise_continuation":False,"reason":"environment episode state is not checkpointed"}

def _team_credit_comparable_config(config):
 value=deepcopy(config);value.pop("development_method",None);value.pop("development_branch",None)
 value.get("training",{}).pop("total_sampled_steps",None)
 value.get("modules",{}).pop("team_mean_credit",None)
 return value

def validate_team_credit_branch(state,env_config,algorithm_config,expected_runtime=None):
 """Validate the matched 1,505,280 -> 1,805,280 team-credit causal branch."""
 if state.get("algorithm")!="modular_mappo":raise RuntimeError("team-credit source algorithm must be modular_mappo")
 if state.get("modular_mappo_impl_version")!=MODULAR_MAPPO_IMPL_VERSION or state.get("baseline_mappo_impl_version")!=MAPPO_IMPL_VERSION:raise RuntimeError("team-credit source implementation version mismatch")
 extra=state.get("extra",{});source_env=extra.get("environment_config");source=extra.get("algorithm_config")
 if not isinstance(source_env,dict) or not isinstance(source,dict):raise RuntimeError("team-credit source lacks embedded configs")
 if source_env!=env_config or extra.get("environment_config_sha256")!=config_sha256(env_config):raise RuntimeError("team-credit branch environment mismatch")
 if _team_credit_comparable_config(source)!=_team_credit_comparable_config(algorithm_config):raise RuntimeError("team-credit branch differs outside the registered intervention whitelist")
 source_enabled=set(state.get("enabled_modules",[]));destination_enabled=set(name for name,value in algorithm_config.get("modules",{}).items() if isinstance(value,dict) and value.get("enabled",False))
 if source_enabled!={"actor_lr_decay"}:raise RuntimeError("team-credit source must enable actor_lr_decay only")
 allowed=({"actor_lr_decay"},{"actor_lr_decay","team_mean_credit"})
 if destination_enabled not in allowed:raise RuntimeError("team-credit destination enabled modules mismatch")
 branch=algorithm_config.get("development_branch",{});intervention=branch.get("intervention")
 expected_intervention="team_mean_credit" if "team_mean_credit" in destination_enabled else "team_mean_credit_control"
 if intervention!=expected_intervention:raise RuntimeError("team-credit intervention/config mismatch")
 if int(state.get("sampled_steps",-1))!=1_505_280 or int(branch.get("source_sampled_steps",-1))!=1_505_280:raise RuntimeError("team-credit source sampled_steps must be 1505280")
 if int(branch.get("additional_sampled_steps",-1))!=300_000 or int(branch.get("target_sampled_steps",-1))!=1_805_280:raise RuntimeError("team-credit branch budget mismatch")
 if int(algorithm_config.get("training",{}).get("total_sampled_steps",-1))!=1_805_280:raise RuntimeError("team-credit training target must be exact 1805280")
 if branch.get("actor_optimizer_restore") is not True or branch.get("critic_optimizer_restore") is not True or branch.get("rng_restore") is not True:raise RuntimeError("team-credit branch must restore both optimizers and RNG")
 source_decay=source.get("modules",{}).get("actor_lr_decay",{});destination_decay=algorithm_config.get("modules",{}).get("actor_lr_decay",{})
 if source_decay!=destination_decay:raise RuntimeError("team-credit branch changed actor_lr_decay")
 if float(state.get("actor_optimizer",{}).get("param_groups",[{}])[0].get("lr",-1))!=1e-4:raise RuntimeError("team-credit source actor LR must be 1e-4")
 required=("actor","critic","actor_optimizer","critic_optimizer","rng_state","sampled_steps","vector_steps","ppo_updates","actor_updates","critic_updates","module_config_sha256")
 missing=[key for key in required if key not in state]
 if missing:raise RuntimeError("team-credit source lacks required state: "+", ".join(missing))
 rng=state.get("rng_state",{});rng_required=("python_random_state","numpy_random_state","torch_cpu_rng_state","torch_cuda_rng_state_all","trainer_permutation_rng_state")
 if any(key not in rng for key in rng_required):raise RuntimeError("team-credit source RNG state incomplete")
 if not state["actor_optimizer"].get("state") or not state["critic_optimizer"].get("state"):raise RuntimeError("team-credit source optimizer state incomplete")
 if expected_runtime:
  for key,value in expected_runtime.items():
   if extra.get(key)!=value:raise RuntimeError(f"team-credit branch checkpoint {key} mismatch")
 return {"intervention":intervention,"source_enabled_modules":sorted(source_enabled),"destination_enabled_modules":sorted(destination_enabled),
  "source_sampled_steps":1_505_280,"additional_sampled_steps":300_000,"target_sampled_steps":1_805_280,
  "optimizer_restored":True,"actor_optimizer_restored":True,"critic_optimizer_restored":True,"RNG_restored":True,
  "matched_checkpoint_continuation":True,"identical_branch_reset_protocol":True,
  "historical_plain_bitwise_physical_continuation":False,"reason":"vector environment physical state is not checkpointed"}

def _pwtr_comparable_config(config):
 value=deepcopy(config);value.pop("development_method",None);value.pop("development_branch",None)
 value.get("training",{}).pop("total_sampled_steps",None)
 value.get("modules",{}).pop("persistent_wave_trajectory_replay",None)
 return value

def validate_pwtr_branch(state,env_config,algorithm_config,expected_runtime=None):
 """Validate one exact Plain 1,505,280 -> PWTR ablation 1,805,280 branch."""
 if state.get("algorithm")!="modular_mappo":raise RuntimeError("PWTR source algorithm must be modular_mappo")
 if state.get("modular_mappo_impl_version")!=MODULAR_MAPPO_IMPL_VERSION or state.get("baseline_mappo_impl_version")!=MAPPO_IMPL_VERSION:raise RuntimeError("PWTR source implementation version mismatch")
 extra=state.get("extra",{});source_env=extra.get("environment_config");source=extra.get("algorithm_config")
 if not isinstance(source_env,dict) or not isinstance(source,dict):raise RuntimeError("PWTR source lacks embedded configs")
 if source_env!=env_config or extra.get("environment_config_sha256")!=config_sha256(env_config):raise RuntimeError("PWTR branch environment mismatch")
 if _pwtr_comparable_config(source)!=_pwtr_comparable_config(algorithm_config):raise RuntimeError("PWTR branch differs outside its registered intervention whitelist")
 source_enabled=set(state.get("enabled_modules",[]));destination_enabled=set(name for name,value in algorithm_config.get("modules",{}).items() if isinstance(value,dict) and value.get("enabled",False))
 if source_enabled!={"actor_lr_decay"}:raise RuntimeError("PWTR source must enable actor_lr_decay only")
 if destination_enabled not in ({"actor_lr_decay"},{"actor_lr_decay","persistent_wave_trajectory_replay"}):raise RuntimeError("PWTR destination enabled modules mismatch")
 branch=algorithm_config.get("development_branch",{});intervention=branch.get("intervention")
 methods={
  "pwtr_plain_control":"pwtr_plain_matched_control","pwtr_stratified":"pwtr_stratified",
  "pwtr_current_extra":"pwtr_current_extra","pwtr_uniform_recent":"pwtr_uniform_recent",
  "pwtr_priority_recent":"pwtr_priority_recent","pwtr_full":"pwtr_full",
  "pwtr_current_actor_only":"pwtr_current_actor_only",
  "pwtr_current_critic_only":"pwtr_current_critic_only",
  "pwtr_recent_actor_only":"pwtr_recent_actor_only",
  "pwtr_recent_critic_only":"pwtr_recent_critic_only"}
 if intervention not in methods or algorithm_config.get("development_method")!=methods[intervention]:raise RuntimeError("PWTR intervention/development identity mismatch")
 module=algorithm_config.get("modules",{}).get("persistent_wave_trajectory_replay",{})
 expected_modes={
  "pwtr_plain_control":None,
  "pwtr_stratified":(True,False,"recent_uniform",False,False,False,False),
  "pwtr_current_extra":(True,True,"current",False,False,True,True),
  "pwtr_uniform_recent":(True,True,"recent_uniform",False,False,True,True),
  "pwtr_priority_recent":(True,True,"recent_priority",True,False,True,True),
  "pwtr_full":(True,True,"recent_priority",True,True,True,True),
  "pwtr_current_actor_only":(True,True,"current",False,False,True,False),
  "pwtr_current_critic_only":(True,True,"current",False,False,False,True),
  "pwtr_recent_actor_only":(True,True,"recent_uniform",False,False,True,False),
  "pwtr_recent_critic_only":(True,True,"recent_uniform",False,False,False,True)}
 expected=expected_modes[intervention]
 if expected is None:
  if "persistent_wave_trajectory_replay" in destination_enabled:raise RuntimeError("PWTR Plain control enabled replay")
 else:
  actual=(bool(module.get("fresh_wave_stratification")),bool(module.get("replay_enabled")),module.get("replay_source"),bool(module.get("priority_enabled")),bool(module.get("bridge_enabled")),bool(module.get("actor_replay")),bool(module.get("critic_replay")))
  if actual!=expected:raise RuntimeError(f"PWTR ablation mode mismatch: {actual}")
  fixed=(int(module.get("sequence_length",-1)),int(module.get("bridge_half_length",-1)),int(module.get("min_segment_length",-1)),int(module.get("partition_capacity",-1)),int(module.get("actor_max_age_updates",-1)))
  if fixed!=(128,64,32,32,2):raise RuntimeError("PWTR fixed constants mismatch")
 if int(state.get("sampled_steps",-1))!=1_505_280 or int(branch.get("source_sampled_steps",-1))!=1_505_280:raise RuntimeError("PWTR source sampled_steps must be 1505280")
 if int(branch.get("additional_sampled_steps",-1))!=300_000 or int(branch.get("target_sampled_steps",-1))!=1_805_280 or int(algorithm_config.get("training",{}).get("total_sampled_steps",-1))!=1_805_280:raise RuntimeError("PWTR branch budget mismatch")
 if branch.get("actor_optimizer_restore") is not True or branch.get("critic_optimizer_restore") is not True or branch.get("rng_restore") is not True:raise RuntimeError("PWTR branch must restore both optimizers and RNG")
 if source.get("modules",{}).get("actor_lr_decay")!=algorithm_config.get("modules",{}).get("actor_lr_decay"):raise RuntimeError("PWTR branch changed actor_lr_decay")
 if float(state.get("actor_optimizer",{}).get("param_groups",[{}])[0].get("lr",-1))!=1e-4:raise RuntimeError("PWTR source actor LR must be 1e-4")
 required=("actor","critic","actor_optimizer","critic_optimizer","rng_state","sampled_steps","vector_steps","ppo_updates","actor_updates","critic_updates","module_config_sha256")
 missing=[key for key in required if key not in state]
 if missing:raise RuntimeError("PWTR source lacks required state: "+", ".join(missing))
 rng=state.get("rng_state",{});rng_required=("python_random_state","numpy_random_state","torch_cpu_rng_state","torch_cuda_rng_state_all","trainer_permutation_rng_state")
 if any(key not in rng for key in rng_required):raise RuntimeError("PWTR source RNG state incomplete")
 if not state["actor_optimizer"].get("state") or not state["critic_optimizer"].get("state"):raise RuntimeError("PWTR source optimizer state incomplete")
 if expected_runtime:
  for key,value in expected_runtime.items():
   if extra.get(key)!=value:raise RuntimeError(f"PWTR branch checkpoint {key} mismatch")
 return {"intervention":intervention,"source_enabled_modules":sorted(source_enabled),"destination_enabled_modules":sorted(destination_enabled),
  "source_sampled_steps":1_505_280,"additional_sampled_steps":300_000,"target_sampled_steps":1_805_280,
  "optimizer_restored":True,"actor_optimizer_restored":True,"critic_optimizer_restored":True,"RNG_restored":True,
  "matched_checkpoint_continuation":True,"identical_branch_reset_protocol":True,
 "historical_plain_bitwise_physical_continuation":False,"reason":"vector environment physical state is not checkpointed"}

def validate_w1sg_branch(state,env_config,algorithm_config,expected_runtime=None):
 """Validate the sole W1SG V1 continuation: exact PWTR CurrentActor plus gating."""
 branch=algorithm_config.get("development_branch",{})
 if algorithm_config.get("development_method")!="w1sg_current_actor" or branch.get("intervention")!="w1sg_current_actor":raise RuntimeError("W1SG development identity mismatch")
 shadow=deepcopy(algorithm_config);shadow["development_method"]="pwtr_current_actor_only";shadow["development_branch"]["intervention"]="pwtr_current_actor_only";shadow.get("modules",{}).pop("wave1_sensitivity_gating",None)
 base=validate_pwtr_branch(state,env_config,shadow,expected_runtime)
 enabled=set(name for name,value in algorithm_config.get("modules",{}).items() if isinstance(value,dict) and value.get("enabled",False))
 required={"actor_lr_decay","persistent_wave_trajectory_replay","wave1_sensitivity_gating"}
 if enabled!=required:raise RuntimeError(f"W1SG destination enabled modules mismatch: {sorted(enabled)}")
 expected={"enabled":True,"source_wave":1,"importance_objective":"ppo_surrogate","importance_chunk_states":512,"gate_transform":"inverse_sqrt","preserve_global_gradient_norm":True,"epsilon":1e-12}
 if algorithm_config["modules"].get("wave1_sensitivity_gating")!=expected:raise RuntimeError("W1SG fixed config mismatch")
 base.update({"intervention":"w1sg_current_actor","destination_enabled_modules":sorted(enabled),"w1sg_version":1})
 return base

def validate_wsai_branch(state,env_config,algorithm_config,expected_runtime=None):
 """Validate the sole WSAI continuation from the matched Plain checkpoint."""
 branch=algorithm_config.get("development_branch",{})
 if algorithm_config.get("development_method")!="wsai_mappo" or branch.get("intervention")!="wave_specific_actor_isolation":raise RuntimeError("WSAI development identity mismatch")
 shadow=deepcopy(algorithm_config);shadow["development_method"]="pwtr_plain_matched_control";shadow["development_branch"]["intervention"]="pwtr_plain_control";shadow.get("modules",{}).pop("wave_specific_actor_isolation",None)
 base=validate_pwtr_branch(state,env_config,shadow,expected_runtime)
 enabled=set(name for name,value in algorithm_config.get("modules",{}).items() if isinstance(value,dict) and value.get("enabled",False))
 required={"actor_lr_decay","wave_specific_actor_isolation"}
 if enabled!=required:raise RuntimeError(f"WSAI destination enabled modules mismatch: {sorted(enabled)}")
 from algorithm.modules.wave_specific_actor_isolation import EXPECTED_WSAI_CONFIG
 if algorithm_config["modules"].get("wave_specific_actor_isolation")!=EXPECTED_WSAI_CONFIG:raise RuntimeError("WSAI fixed config mismatch")
 base.update({"intervention":"wave_specific_actor_isolation","destination_enabled_modules":sorted(enabled),
  "wsai_version":1,"isolated_actor_count":3,"shared_critic":True,"natural_wave_weighting":True})
 return base

def validate_wsmh_branch(state,env_config,algorithm_config,expected_runtime=None):
 """Validate WSMH as the sole structural change from matched Plain."""
 branch=algorithm_config.get("development_branch",{})
 if algorithm_config.get("development_method")!="wsmh_mappo" or branch.get("intervention")!="wave_specific_mean_heads":raise RuntimeError("WSMH development identity mismatch")
 shadow=deepcopy(algorithm_config);shadow["development_method"]="pwtr_plain_matched_control";shadow["development_branch"]["intervention"]="pwtr_plain_control";shadow.get("modules",{}).pop("wave_specific_mean_heads",None)
 base=validate_pwtr_branch(state,env_config,shadow,expected_runtime)
 enabled=set(name for name,value in algorithm_config.get("modules",{}).items() if isinstance(value,dict) and value.get("enabled",False))
 required={"actor_lr_decay","wave_specific_mean_heads"}
 if enabled!=required:raise RuntimeError(f"WSMH destination enabled modules mismatch: {sorted(enabled)}")
 from algorithm.modules.wave_specific_mean_heads import EXPECTED_WSMH_CONFIG
 if algorithm_config["modules"].get("wave_specific_mean_heads")!=EXPECTED_WSMH_CONFIG:raise RuntimeError("WSMH fixed config mismatch")
 base.update({"intervention":"wave_specific_mean_heads","destination_enabled_modules":sorted(enabled),"wsmh_version":1,
  "wave_mean_head_count":3,"shared_backbone":True,"shared_log_std":True,"shared_critic":True,"natural_wave_weighting":True})
 return base

def validate_actor_grad_clip_branch(state,env_config,algorithm_config,expected_runtime=None):
 """Validate either fixed Actor-only clipping branch against matched Plain."""
 branch=algorithm_config.get("development_branch",{});intervention=branch.get("intervention");method=algorithm_config.get("development_method")
 allowed={"actor_grad_clip_05_control":.5,"actor_grad_clip_10":1.0}
 if intervention not in allowed or method!=intervention:raise RuntimeError("Actor gradient clipping branch identity mismatch")
 shadow=deepcopy(algorithm_config);shadow["development_method"]="pwtr_plain_matched_control";shadow["development_branch"]["intervention"]="pwtr_plain_control";shadow.get("modules",{}).pop("actor_gradient_clipping",None)
 base=validate_pwtr_branch(state,env_config,shadow,expected_runtime)
 enabled=set(name for name,value in algorithm_config.get("modules",{}).items() if isinstance(value,dict) and value.get("enabled",False));required={"actor_lr_decay","actor_gradient_clipping"}
 if enabled!=required:raise RuntimeError(f"Actor gradient clipping destination modules mismatch: {sorted(enabled)}")
 expected={"enabled":True,"mode":"actor_only_fixed_norm","actor_max_grad_norm":allowed[intervention],"critic_max_grad_norm_unchanged":True,"critic_max_grad_norm":.5}
 if algorithm_config["modules"].get("actor_gradient_clipping")!=expected:raise RuntimeError("Actor gradient clipping fixed config mismatch")
 if float(algorithm_config["training"]["max_grad_norm"])!=.5:raise RuntimeError("training.max_grad_norm must remain the critic/default limit 0.5")
 base.update({"intervention":intervention,"destination_enabled_modules":sorted(enabled),"actor_grad_clip_version":1,
  "actor_max_grad_norm":allowed[intervention],"critic_max_grad_norm":.5,"actor_clip_only_intervention":True})
 return base

def validate_actor_grad_clip_config_pair(control,treatment):
 """Require exactly branch identity plus the Actor limit to differ."""
 left=deepcopy(control);right=deepcopy(treatment);left.pop("development_method",None);right.pop("development_method",None)
 left.get("development_branch",{}).pop("intervention",None);right.get("development_branch",{}).pop("intervention",None)
 left["modules"]["actor_gradient_clipping"].pop("actor_max_grad_norm",None);right["modules"]["actor_gradient_clipping"].pop("actor_max_grad_norm",None)
 if left!=right:raise RuntimeError("Actor gradient clipping configs differ beyond branch identity and Actor limit")
 return True

def validate_dawe_branch(state,env_config,algorithm_config,expected_runtime=None):
 """Validate the exact Fixed10 Control05 versus DAWE V1 continuation."""
 branch=algorithm_config.get("development_branch",{});intervention=branch.get("intervention")
 methods={"dawe_fixed10_control":"dawe_fixed10_control",
          "deployment_aligned_wave_exploration":"dawe_fixed10_v1"}
 if intervention not in methods or algorithm_config.get("development_method")!=methods[intervention]:
  raise RuntimeError("DAWE branch identity mismatch")
 shadow=deepcopy(algorithm_config);shadow["development_method"]="actor_grad_clip_05_control"
 shadow["development_branch"]["intervention"]="actor_grad_clip_05_control"
 shadow.get("modules",{}).pop("deployment_aligned_wave_exploration",None)
 base=validate_actor_grad_clip_branch(state,env_config,shadow,expected_runtime)
 enabled=set(name for name,value in algorithm_config.get("modules",{}).items()
             if isinstance(value,dict) and value.get("enabled",False))
 required={"actor_lr_decay","actor_gradient_clipping"}
 if intervention=="deployment_aligned_wave_exploration":required.add("deployment_aligned_wave_exploration")
 if enabled!=required:raise RuntimeError(f"DAWE destination enabled modules mismatch: {sorted(enabled)}")
 expected_module={"enabled":intervention=="deployment_aligned_wave_exploration",
  "mode":"fixed_wave_std_multiplier","wave1_multiplier":.25,"wave2_multiplier":.25,"wave3_multiplier":1.0}
 if algorithm_config.get("modules",{}).get("deployment_aligned_wave_exploration")!=expected_module:
  raise RuntimeError("DAWE fixed module config mismatch")
 clip=algorithm_config.get("modules",{}).get("actor_gradient_clipping",{})
 if float(clip.get("actor_max_grad_norm",-1))!=.5 or float(algorithm_config["training"].get("max_grad_norm",-1))!=.5:
  raise RuntimeError("DAWE requires Fixed10 Control05 clipping")
 if (int(branch.get("source_sampled_steps",-1)),int(branch.get("additional_sampled_steps",-1)),
     int(branch.get("target_sampled_steps",-1)))!=(1_505_280,300_000,1_805_280):
  raise RuntimeError("DAWE branch budget mismatch")
 if branch.get("actor_optimizer_restore") is not True or branch.get("critic_optimizer_restore") is not True or branch.get("rng_restore") is not True:
  raise RuntimeError("DAWE branch must restore both optimizers and RNG")
 base.update({"intervention":intervention,"destination_enabled_modules":sorted(enabled),
  "dawe_version":1,"dawe_fixed_multipliers":[.25,.25,1.0],"actor_mean_unchanged":True,
  "critic_unchanged":True,"reward_unchanged":True})
 return base

def validate_dawe_config_pair(control,treatment):
 """Require exactly the preregistered three DAWE pair differences."""
 left=deepcopy(control);right=deepcopy(treatment)
 left.pop("development_method",None);right.pop("development_method",None)
 left.get("development_branch",{}).pop("intervention",None)
 right.get("development_branch",{}).pop("intervention",None)
 left["modules"]["deployment_aligned_wave_exploration"].pop("enabled",None)
 right["modules"]["deployment_aligned_wave_exploration"].pop("enabled",None)
 if left!=right:raise RuntimeError("DAWE configs differ outside method, intervention, and enabled")
 return True

def validate_rv_branch(state,env_config,algorithm_config,expected_runtime=None):
 """Validate the exact Fixed10 Control versus RV-MAPPO V1 continuation."""
 branch=algorithm_config.get("development_branch",{});intervention=branch.get("intervention")
 methods={"rv_fixed10_control":"rv_fixed10_control","reference_variance":"rv_mappo_v1"}
 if intervention not in methods or algorithm_config.get("development_method")!=methods[intervention]:raise RuntimeError("RV branch identity mismatch")
 shadow=deepcopy(algorithm_config);shadow["development_method"]="actor_grad_clip_05_control";shadow["development_branch"]["intervention"]="actor_grad_clip_05_control";shadow.get("modules",{}).pop("reference_variance",None)
 base=validate_actor_grad_clip_branch(state,env_config,shadow,expected_runtime)
 enabled=set(name for name,value in algorithm_config.get("modules",{}).items() if isinstance(value,dict) and value.get("enabled",False))
 required={"actor_lr_decay","actor_gradient_clipping"}
 if intervention=="reference_variance":required.add("reference_variance")
 if enabled!=required:raise RuntimeError(f"RV destination enabled modules mismatch: {sorted(enabled)}")
 expected={"enabled":intervention=="reference_variance","mode":"frozen_source_state_dependent_variance","source_sampled_steps":1_505_280,"freeze_current_log_std_head":True,"behavior_uses_reference_mean":False}
 if algorithm_config.get("modules",{}).get("reference_variance")!=expected:raise RuntimeError("RV fixed module config mismatch")
 if algorithm_config.get("modules",{}).get("deployment_aligned_wave_exploration",{}).get("enabled",False):raise RuntimeError("RV protocol forbids DAWE")
 if (int(branch.get("source_sampled_steps",-1)),int(branch.get("additional_sampled_steps",-1)),int(branch.get("target_sampled_steps",-1)))!=(1_505_280,300_000,1_805_280):raise RuntimeError("RV branch budget mismatch")
 if branch.get("actor_optimizer_restore") is not True or branch.get("critic_optimizer_restore") is not True or branch.get("rng_restore") is not True:raise RuntimeError("RV branch must restore both optimizers and RNG")
 base.update({"intervention":intervention,"destination_enabled_modules":sorted(enabled),"reference_variance_version":1,"reference_actor_created_from_branch_source":intervention=="reference_variance","current_log_std_head_frozen":intervention=="reference_variance","actor_optimizer_param_membership_preserved":True,"actor_optimizer_restore":True,"critic_optimizer_restore":True,"rng_restore":True})
 return base

def validate_rv_config_pair(control,treatment):
 left=deepcopy(control);right=deepcopy(treatment);left.pop("development_method",None);right.pop("development_method",None)
 left.get("development_branch",{}).pop("intervention",None);right.get("development_branch",{}).pop("intervention",None)
 left["modules"]["reference_variance"].pop("enabled",None);right["modules"]["reference_variance"].pop("enabled",None)
 if left!=right:raise RuntimeError("RV configs differ outside method, intervention, and enabled")
 return True

def _fbmr_comparable_config(config):
 value=deepcopy(config)
 for key in ("formal_protocol","development_protocol","development_branch"):value.pop(key,None)
 value.get("training",{}).pop("total_sampled_steps",None);value.get("training",{}).pop("actor_learning_rate",None)
 value.get("implementation",{}).pop("evaluation_seed_base",None)
 value.get("modules",{}).pop("actor_lr_decay",None);value.get("modules",{}).pop("entity_attention",None)
 return value

def validate_fbmr_stage2_branch(state,env_config,algorithm_config,expected_runtime=None):
 """Strictly validate one 900k Formal MAPPO source and its Stage-2 intervention."""
 if state.get("algorithm")!="modular_mappo":raise RuntimeError("FBMR source algorithm must be modular_mappo")
 if state.get("modular_mappo_impl_version")!=MODULAR_MAPPO_IMPL_VERSION or state.get("baseline_mappo_impl_version")!=MAPPO_IMPL_VERSION:raise RuntimeError("FBMR source implementation version mismatch")
 extra=state.get("extra",{});_validate_embedded_disabled_curriculum_runtime(extra,algorithm_config);source_env=extra.get("environment_config");source=extra.get("algorithm_config")
 if not isinstance(source_env,dict) or not isinstance(source,dict):raise RuntimeError("FBMR source lacks embedded configs")
 if source_env!=env_config or extra.get("environment_config_sha256")!=config_sha256(env_config):raise RuntimeError("FBMR source environment mismatch")
 if extra.get("algorithm_config_sha256")!=config_sha256(source) or state.get("module_config_sha256")!=canonical_sha256(source.get("modules",{})):raise RuntimeError("FBMR source self-description hash mismatch")
 if extra.get("environment_variant")!="persistent_wave_v2":raise RuntimeError("FBMR source must be persistent_wave_v2")
 if int(state.get("sampled_steps",-1))!=900000:raise RuntimeError("FBMR source sampled_steps must be exactly 900000")
 if int(extra.get("training_num_envs",-1))!=24 or float(extra.get("training_gamma",0))!=.999:raise RuntimeError("FBMR source runtime mismatch")
 if bool(source.get("modules",{}).get("entity_attention",{}).get("enabled",False)):raise RuntimeError("FBMR source must be baseline MAPPO")
 decay=source.get("modules",{}).get("actor_lr_decay",{})
 expected_decay={"enabled":True,"schedule":"delayed_linear","start_step":600000,"end_step":900000,"start_lr":.0003,"end_lr":.0001}
 if decay!=expected_decay:raise RuntimeError("FBMR source actor LR schedule mismatch")
 if abs(float(state["actor_optimizer"]["param_groups"][0]["lr"])-1e-4)>1e-15:raise RuntimeError("FBMR source terminal actor LR is not 1e-4")
 if abs(float(source["training"]["critic_learning_rate"])-3e-4)>1e-15:raise RuntimeError("FBMR source critic LR mismatch")
 required=("actor","critic","actor_optimizer","critic_optimizer","rng_state","sampled_steps","vector_steps","ppo_updates","actor_updates","critic_updates")
 missing=[key for key in required if key not in state]
 if missing:raise RuntimeError("FBMR source lacks continuation state: "+", ".join(missing))
 rng=state.get("rng_state",{});rng_required=("python_random_state","numpy_random_state","torch_cpu_rng_state","torch_cuda_rng_state_all","trainer_permutation_rng_state")
 if state.get("rng_state_available") is not True or any(key not in rng for key in rng_required):raise RuntimeError("FBMR source RNG state is incomplete")
 if not state["actor_optimizer"].get("state") or not state["critic_optimizer"].get("state"):raise RuntimeError("FBMR source optimizer state is incomplete")
 architecture=extra.get("network_architecture",{})
 if architecture.get("entity_attention_enabled") is not False or architecture.get("entity_attention_mode","disabled")!="disabled":raise RuntimeError("FBMR source architecture metadata is not baseline MAPPO")
 if set(state["actor"])!={"backbone.0.weight","backbone.0.bias","backbone.2.weight","backbone.2.bias","mean.weight","mean.bias","log_std.weight","log_std.bias"}:raise RuntimeError("FBMR source actor topology is not baseline MAPPO")
 if _fbmr_comparable_config(source)!=_fbmr_comparable_config(algorithm_config):raise RuntimeError("FBMR Stage-2 config differs outside its strict whitelist")
 branch=algorithm_config.get("development_branch",{});intervention=branch.get("intervention")
 if intervention not in {"mappo_continuation","frozen_base_mean_residual"}:raise RuntimeError("unknown FBMR Stage-2 intervention")
 training=algorithm_config["training"]
 if int(training["total_sampled_steps"])!=1200000 or int(training["num_train_envs"])!=24 or float(training["critic_learning_rate"])!=3e-4:raise RuntimeError("FBMR Stage-2 budget/runtime mismatch")
 if int(algorithm_config["implementation"]["evaluation_seed_base"])!=34000000 or int(training["evaluation_episodes"])!=20 or int(training["evaluation_interval_sampled_steps"])!=100000:raise RuntimeError("FBMR Stage-2 validation protocol mismatch")
 entity=algorithm_config.get("modules",{}).get("entity_attention",{});destination_decay=algorithm_config.get("modules",{}).get("actor_lr_decay",{})
 if intervention=="mappo_continuation":
  if entity.get("enabled",False) or destination_decay!=expected_decay or float(training["actor_learning_rate"])!=3e-4:raise RuntimeError("MAPPO continuation intervention mismatch")
  actor_optimizer_restore=True
 else:
  expected_entity={"enabled":True,"mode":"frozen_base_mean_residual","entity_dim":32,"attention_heads":2,"max_mean_correction":.25}
  if entity!=expected_entity or destination_decay.get("enabled",False) or float(training["actor_learning_rate"])!=1e-4:raise RuntimeError("FBMR intervention mismatch")
  actor_optimizer_restore=False
 if expected_runtime:
  for key,expected in expected_runtime.items():
   if extra.get(key)!=expected:raise RuntimeError(f"FBMR source {key} mismatch")
 return {"intervention":intervention,"source_actor_lr_decay":deepcopy(decay),"destination_actor_lr_decay":deepcopy(destination_decay),"source_sampled_steps":900000,"target_sampled_steps":1200000,"actor_effective_lr":1e-4,"critic_lr":3e-4,"actor_optimizer_restore":actor_optimizer_restore,"critic_optimizer_restore":True,"rng_restore":True}

def _normalized_fbmr_bound_protocol(config):
 value=deepcopy(config)
 branch=value.get("development_branch",{});branch["intervention"]="FROZEN_BASE_BOUND_MODE"
 for key in ("dual_bound_enabled","alpha_abs","alpha_rel","max_mean_correction"):branch.pop(key,None)
 entity=value.get("modules",{}).get("entity_attention",{});entity["mode"]="FROZEN_BASE_BOUND_MODE"
 for key in ("alpha_abs","alpha_rel","max_mean_correction"):entity.pop(key,None)
 validation=value.get("development_protocol",{}).get("validation",{})
 if validation.get("role") in {"FBMR_STAGE2_DEVELOPMENT_VALIDATION","FBMR_V2_DEVELOPMENT_VALIDATION"}:validation["role"]="FBMR_BOUND_DEVELOPMENT_VALIDATION"
 return value

def validate_fbmr_v1_v2_only_bound_diff(v1_config,v2_config):
 """Require the resolved V1/V2 protocols to differ only in their fixed bound."""
 if _normalized_fbmr_bound_protocol(v1_config)!=_normalized_fbmr_bound_protocol(v2_config):
  raise RuntimeError("resolved FBMR V1/V2 configs differ outside the bound intervention")
 return True

def validate_fbmr_v2_stage2_branch(state,env_config,algorithm_config,expected_runtime=None):
 """Strict V2 validator reusing every V1 source invariant."""
 shadow=deepcopy(algorithm_config)
 shadow["modules"]["entity_attention"]={"enabled":True,"mode":"frozen_base_mean_residual","entity_dim":32,"attention_heads":2,"max_mean_correction":.25}
 shadow["development_branch"]["intervention"]="frozen_base_mean_residual"
 shadow.get("development_protocol",{}).get("validation",{})["role"]="FBMR_STAGE2_DEVELOPMENT_VALIDATION"
 result=validate_fbmr_stage2_branch(state,env_config,shadow,expected_runtime)
 entity=algorithm_config.get("modules",{}).get("entity_attention",{})
 expected={"enabled":True,"mode":"frozen_base_dual_bounded_mean_residual","entity_dim":32,"attention_heads":2,"alpha_abs":.25,"alpha_rel":.25}
 branch=algorithm_config.get("development_branch",{})
 if entity!=expected:raise RuntimeError("FBMR V2 entity config mismatch")
 if branch.get("intervention")!="frozen_base_dual_bounded_mean_residual":raise RuntimeError("FBMR V2 branch intervention mismatch")
 if branch.get("source_sampled_steps")!=900000 or branch.get("additional_sampled_steps")!=300000 or branch.get("target_sampled_steps")!=1200000:raise RuntimeError("FBMR V2 branch budget mismatch")
 if branch.get("actor_optimizer_restore") is not False or branch.get("critic_optimizer_restore") is not True or branch.get("rng_restore") is not True:raise RuntimeError("FBMR V2 restore protocol mismatch")
 if branch.get("base_actor_frozen") is not True or branch.get("log_std_source")!="frozen_baseline":raise RuntimeError("FBMR V2 frozen-policy protocol mismatch")
 result.update({"intervention":"frozen_base_dual_bounded_mean_residual","dual_bound_enabled":True,"alpha_abs":.25,"alpha_rel":.25})
 return result

def is_formal_v2_checkpoint(state):
 extra=state.get("extra",{}) if isinstance(state.get("extra",{}),dict) else {}
 required=("environment_version","environment_variant","environment_config_sha256",
           "algorithm_config_sha256","network_architecture","observation_dim",
           "action_dim","num_agents","training_seed","training_gamma",
           "training_num_envs","training_total_sampled_steps","training_smoke")
 return (state.get("algorithm")=="modular_mappo" and
         state.get("modular_mappo_impl_version")==MODULAR_MAPPO_IMPL_VERSION and
         state.get("baseline_mappo_impl_version")==MAPPO_IMPL_VERSION and
         all(key in extra for key in required))

__all__=["canonical_sha256","checkpoint_architecture","validate_modular_checkpoint","validate_modular_branch","validate_team_credit_branch","validate_pwtr_branch","validate_w1sg_branch","validate_wsai_branch","validate_wsmh_branch","validate_actor_grad_clip_branch","validate_actor_grad_clip_config_pair","validate_dawe_branch","validate_dawe_config_pair","validate_fbmr_stage2_branch","validate_fbmr_v2_stage2_branch","validate_fbmr_v1_v2_only_bound_diff","is_formal_v2_checkpoint"]
