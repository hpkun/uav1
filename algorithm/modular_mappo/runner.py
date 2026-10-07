"""Formal-grade modular MAPPO runner with bounded memory and strict lineage."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from env.config import ENVIRONMENT_VERSION
from algorithm.common.checkpoint import evaluation_selection_key
from algorithm.common.protocol import config_sha256, runtime_source_manifest
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.mappo.networks import SharedMAPPOActor
from algorithm.mappo.trainer import MAPPO_IMPL_VERSION
from .buffer import ModularRolloutBatch
from algorithm.modules import (ManagerTransitionBatch,smdp_boundary_masks,
                               successful_wave_from_transition)
from .evaluation import evaluate_modular
from .factory import build_modular_mappo_trainer
from .protocol import checkpoint_architecture, validate_modular_checkpoint
from .trainer import MODULAR_MAPPO_IMPL_VERSION
from algorithm.modules.wave_survival_pbrs import mission_context_numpy, mission_progress_from_wave_state


def environment_runtime_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Compact, auditable identity for every environment used by the runner."""
    return {
        "environment_variant": config.get("environment_variant", "direct_v2_3"),
        "environment_version": str(config.get("environment_version", ENVIRONMENT_VERSION)),
        "total_waves": int(config.get("persistent_waves", {}).get("total_waves", 1)),
        "max_steps": int(config["simulation"]["max_steps"]),
        "team_size": int(config["scenario"]["team_size"]),
        "action_config_sha256": config_sha256(config.get("action", {})),
        "weapon_config_sha256": config_sha256(config.get("weapon", {})),
        "reward_config_sha256": config_sha256(config.get("reward", {})),
        "blue_policy_config_sha256": config_sha256(config.get("blue_policy", {})),
        "config_sha256": config_sha256(config),
    }


def validate_runtime_environment_contract(
    declared: dict[str, Any], effective: dict[str, Any], evaluation: dict[str, Any],
    *, curriculum_enabled: bool,
) -> dict[str, Any]:
    """Fail before rollout when declared/training/evaluation semantics diverge."""
    declared_id = environment_runtime_identity(declared)
    effective_id = environment_runtime_identity(effective)
    evaluation_id = environment_runtime_identity(evaluation)
    if evaluation_id["config_sha256"] != declared_id["config_sha256"]:
        raise RuntimeError("evaluation environment must equal the declared environment contract")
    if not curriculum_enabled:
        if effective_id["config_sha256"] != declared_id["config_sha256"]:
            raise RuntimeError(
                "disabled curriculum changed runtime environment: "
                f"declared_waves={declared_id['total_waves']} effective_waves={effective_id['total_waves']} "
                f"declared_max_steps={declared_id['max_steps']} effective_max_steps={effective_id['max_steps']} "
                f"declared_hash={declared_id['config_sha256']} effective_hash={effective_id['config_sha256']}"
            )
    else:
        normalized = deepcopy(effective)
        normalized.setdefault("persistent_waves", {})["total_waves"] = declared_id["total_waves"]
        if config_sha256(normalized) != declared_id["config_sha256"]:
            raise RuntimeError("enabled curriculum changed environment fields other than persistent_waves.total_waves")
    return {"declared": declared_id, "effective_training": effective_id,
            "evaluation": evaluation_id, "curriculum_enabled": bool(curriculum_enabled)}


class ModularMAPPOTrainingRunner:
    def __init__(self, env_config: dict, algorithm_config: dict,
                 num_envs: int | None = None, total_sampled_steps: int | None = None,
                 device: str | None = None, seed: int | None = None,
                 output_dir: str | Path | None = None, smoke: bool = False,
                 warm_start_checkpoint: str | None = None,
                 reference_checkpoint: str | None = None,
                 resume_mode: bool = False,
                 branch_provenance: dict[str, Any] | None = None,
                 runtime_source_manifest_data: dict[str, Any] | None = None) -> None:
        self.declared_env_config = deepcopy(env_config)
        self.env_config = self.declared_env_config  # compatibility alias: always declared/source
        self.evaluation_env_config = deepcopy(env_config)
        self.algorithm_config = deepcopy(algorithm_config)
        from .protocol import validate_marc_gru_screen_environment,validate_marc_factorial_config,validate_jiao2025_3m_config,JIAO_3M_METHODS
        validate_jiao2025_3m_config(algorithm_config,env_config)
        validate_marc_factorial_config(algorithm_config,env_config)
        validate_marc_gru_screen_environment(algorithm_config,env_config)
        self.output_dir = Path(output_dir)
        self.branch_provenance = deepcopy(branch_provenance or {})
        self.runtime_source_manifest = deepcopy(
            runtime_source_manifest_data
            if runtime_source_manifest_data is not None
            else runtime_source_manifest(Path(__file__).resolve().parents[2])
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("training_metrics.jsonl", "optimization_metrics.jsonl"):
            (self.output_dir / name).touch(exist_ok=True)
        self.smoke = bool(smoke)
        training = algorithm_config["training"]
        implementation = algorithm_config["implementation"]
        logging = algorithm_config["runtime_logging"]
        self.num_envs = int(num_envs or training["num_train_envs"])
        self.total_sampled_steps = int(total_sampled_steps or training["total_sampled_steps"])
        self.seed = int(training["seed"] if seed is None else seed)
        self.device = str(device or training["device"])
        configured = deepcopy(algorithm_config)
        configured["training"]["seed"] = self.seed
        warm_enabled = bool(configured.get("modules", {}).get("warm_start", {}).get("enabled", False))
        anchor_enabled = bool(configured.get("modules", {}).get("policy_anchor", {}).get("enabled", False))
        entity_enabled = bool(configured.get("modules", {}).get("entity_attention", {}).get("enabled", False))
        self.effective_hidden_dim = int(configured["network"]["actor_hidden_layers"][0])
        if self.smoke and not (warm_enabled or anchor_enabled or entity_enabled) and configured.get("development_method") not in JIAO_3M_METHODS|{"marc_core_factorial_v1","marc_credit_balance_ablation","marc_mappo_wsgru_v1","marc_mappo_state_memory_v1"}:
            self.effective_hidden_dim = 64
        self.trainer = build_modular_mappo_trainer(configured, self.device, self.effective_hidden_dim,self.total_sampled_steps)
        if self.trainer.wave_entry_curriculum.enabled:
            if (self.declared_env_config.get("environment_variant") != "persistent_wave_v2"
                    or int(self.declared_env_config.get("persistent_waves", {}).get("total_waves", 0)) != 3):
                raise ValueError("wave_entry_curriculum requires persistent_wave_v2 with exactly three waves")
        self.rollout_steps = 4 if self.smoke else int(training["rollout_steps"])
        self.eval_episodes = min(2, int(training["evaluation_episodes"])) if self.smoke else int(training["evaluation_episodes"])
        self.eval_base = int(implementation["evaluation_seed_base"])
        self.evaluation_interval = int(training["evaluation_interval_sampled_steps"])
        self.checkpoint_interval = int(implementation["checkpoint_interval_sampled_steps"])
        self.console_interval = int(logging["console_interval_sampled_steps"])
        self.recent_window = int(logging["recent_episode_window"])
        self.next_evaluation = self.evaluation_interval
        self.next_checkpoint = self.checkpoint_interval
        self.next_console = self.console_interval
        self.curriculum_enabled = bool(self.trainer.curriculum.enabled)
        self.current_stage, _ = self.trainer.curriculum.stage(0)
        self.runtime_env_config = self.trainer.curriculum.runtime_config(self.env_config, 0)
        self.current_waves = int(self.runtime_env_config.get("persistent_waves", {}).get("total_waves", 1))
        self.environment_contract = validate_runtime_environment_contract(
            self.declared_env_config, self.runtime_env_config, self.evaluation_env_config,
            curriculum_enabled=self.curriculum_enabled,
        )
        self.vector: ParallelVectorEnv | None = None
        self._make_vector()
        self.recent_episodes: deque[dict[str, Any]] = deque(maxlen=self.recent_window)
        self.completed_episode_count = 0
        self.raw_episode_returns = np.zeros((self.num_envs, 4), dtype=np.float64)
        self.training_episode_returns = np.zeros((self.num_envs, 4), dtype=np.float64)
        self.paper_episode_blue = np.zeros(self.num_envs, dtype=np.float64)
        self.paper_episode_red = np.zeros(self.num_envs, dtype=np.float64)
        self.paper_episode_by_wave = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.pbrs_episode_sum = np.zeros(self.num_envs, dtype=np.float64)
        self.pbrs_episode_abs_sum = np.zeros(self.num_envs, dtype=np.float64)
        self.pbrs_episode_by_wave = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.pbrs_episode_phi_pre = np.zeros(self.num_envs, dtype=np.float64)
        self.pbrs_episode_phi_next = np.zeros(self.num_envs, dtype=np.float64)
        self.pbrs_episode_samples = np.zeros(self.num_envs, dtype=np.int64)
        self.pbrs_totals = np.zeros(5, dtype=np.float64)
        self.transition_counts = np.zeros(3, dtype=np.int64)
        self.alive_agent_counts = np.zeros(3, dtype=np.int64)
        self.wave_clear_transition_counts = np.zeros(3, dtype=np.int64)
        self.reward_bonus_totals = np.zeros(4, dtype=np.float64)
        self.paper_reward_totals = np.zeros(5, dtype=np.float64)
        self.death_index_totals = np.zeros((2, self.alive.shape[1]), dtype=np.int64)
        self.death_cause_names = (
            "red_weapon_deaths", "red_boundary_deaths", "red_ground_deaths",
            "blue_weapon_deaths", "blue_boundary_deaths", "blue_ground_deaths",
        )
        self.death_cause_info_keys = (
            "blue_attack_kills", "red_boundary_exits", "red_ground_losses",
            "red_attack_kills", "blue_boundary_exits", "blue_ground_losses",
        )
        self.previous_cause_counts = np.zeros((self.num_envs, len(self.death_cause_names)), dtype=np.int64)
        self.death_cause_totals = np.zeros(len(self.death_cause_names), dtype=np.int64)
        self.hidden_reset_count = 0
        self.curriculum_transitions = [{"sampled_steps": 0, "stage": self.current_stage, "total_waves": self.current_waves}]
        self.evaluation_history: list[dict[str, Any]] = []
        self.best_evaluation: dict[str, Any] | None = None
        self.best_sampled_steps: int | None = None
        self.latest_evaluation: dict[str, Any] | None = None
        self.last_metrics: dict[str, float] = {}
        self.last_rollout_metrics: dict[str, float] = {}
        self.iw_pending_episode = [{1: [], 2: []} for _ in range(self.num_envs)]
        self.iw_pending_segments_dropped_on_resume = 0
        self.iw_completed_episode_counter = 0
        self.caiw_pending_episode = [{1: [], 2: []} for _ in range(self.num_envs)]
        self.caiw_pending_segments_dropped_on_resume = 0
        self.caiw_completed_episode_counter = 0
        self.brsc_pending_episode = [{1: None, 2: None} for _ in range(self.num_envs)]
        self.brsc_pending_boundaries_dropped_on_resume = 0
        self.brsc_completed_episode_counter = 0
        self.marc_pending_episode = [{1: [], 2: [], 3: []} for _ in range(self.num_envs)]
        self.marc_wave_start_steps = np.zeros(self.num_envs, dtype=np.int64)
        self.marc_pending_dropped_on_resume = 0
        self.resume_count = 0
        if not resume_mode:
            if warm_start_checkpoint:
                self.trainer.warm_start_provenance = self.trainer.warm_start.initialize(self.trainer, warm_start_checkpoint)
            if self.trainer.anchor.enabled:
                if not reference_checkpoint:
                    raise ValueError("fresh policy-anchor training requires --reference-checkpoint")
                self._attach_reference(reference_checkpoint)

    def _attach_reference(self, checkpoint: str) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        network = self.algorithm_config["network"]
        reference = SharedMAPPOActor(
            int(network["observation_dim"]), int(network["action_dim"]),
            int(network["actor_hidden_layers"][0]),
            float(self.algorithm_config["implementation"]["log_std_min"]),
            float(self.algorithm_config["implementation"]["log_std_max"]),
            str(self.algorithm_config["implementation"]["actor_activation"]),
        ).to(self.trainer.device)
        reference.load_state_dict(state["actor"])
        self.trainer.anchor.attach(reference, str(checkpoint))
        extra = state.get("extra", {})
        with open(checkpoint, "rb") as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        self.trainer.anchor_provenance = {
            "reference_checkpoint": str(checkpoint),
            "source_algorithm": state.get("algorithm"),
            "source_sampled_steps": int(state.get("sampled_steps", 0)),
            "source_environment_variant": extra.get("environment_variant"),
            "source_training_seed": extra.get("training_seed"),
            "source_checkpoint_sha256": digest,
        }

    def _make_vector(self, episode_indices: np.ndarray | None = None) -> None:
        self.environment_contract = validate_runtime_environment_contract(
            self.declared_env_config, self.runtime_env_config, self.evaluation_env_config,
            curriculum_enabled=self.curriculum_enabled,
        )
        if self.vector is not None:
            self.vector.close()
        self.vector = ParallelVectorEnv(
            self.num_envs, self.runtime_env_config, self.seed,
            range(self.eval_base, self.eval_base + self.eval_episodes),
        )
        configured_dimensions = (
            int(self.algorithm_config["network"]["observation_dim"]),
            int(self.algorithm_config["network"]["action_dim"]),
            int(self.algorithm_config["network"]["num_agents"]),
        )
        runtime_dimensions = (
            self.vector.observation_dim, self.vector.action_dim, self.vector.team_size,
        )
        if configured_dimensions != runtime_dimensions:
            self.vector.close()
            raise RuntimeError(
                "algorithm/runtime environment dimensions mismatch: "
                f"configured={configured_dimensions}, runtime={runtime_dimensions}"
            )
        if episode_indices is not None:
            self.vector.episode_indices = np.asarray(episode_indices, dtype=np.int64)
        self.observations = self.vector.reset()
        self.alive = self.vector.current_alive_masks.copy()
        self.blue_alive = np.ones_like(self.alive, dtype=np.float32)
        self.wave = np.ones(self.num_envs, dtype=np.int64)
        self.total = np.full(self.num_envs, self.current_waves, dtype=np.int64)
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_mask = np.zeros(self.num_envs, dtype=np.float32)
        self.actor_phase_reset_flags = np.ones(self.num_envs, dtype=bool)
        self.episode_start_wave = np.ones(self.num_envs, dtype=np.int64)
        self.curriculum_source_wave = np.zeros(self.num_envs, dtype=np.int64)
        self.curriculum_snapshot_id = np.full(self.num_envs, None, dtype=object)
        self.episode_reset_seed = self.vector.last_reset_seeds.copy()
        self.actor_hidden, self.critic_hidden = self.trainer.initial_hidden(self.num_envs)
        (self.output_dir / "runtime_env_config.yaml").write_text(
            yaml.safe_dump(self.runtime_env_config, sort_keys=False), encoding="utf-8"
        )

    def _maybe_curriculum(self) -> None:
        if not self.curriculum_enabled:
            return
        stage, waves = self.trainer.curriculum.stage(self.trainer.sampled_steps)
        if (stage, waves) == (self.current_stage, self.current_waves):
            return
        previous = self.vector.episode_indices.copy() + 1
        self.current_stage, self.current_waves = stage, waves
        self.curriculum_transitions.append({"sampled_steps": self.trainer.sampled_steps, "stage": stage, "total_waves": waves})
        self.runtime_env_config = self.trainer.curriculum.runtime_config(self.env_config, self.trainer.sampled_steps)
        self.environment_contract = validate_runtime_environment_contract(
            self.declared_env_config, self.runtime_env_config, self.evaluation_env_config,
            curriculum_enabled=True,
        )
        self._make_vector(previous)

    @staticmethod
    def _fractions(counts: np.ndarray) -> np.ndarray:
        return counts.astype(np.float64) / max(float(counts.sum()), 1.0)

    def _write_episode(self, info: dict[str, Any], raw_return: np.ndarray,
                       training_return: np.ndarray, sampled_steps: int,
                       episode_start_wave: int = 1,
                       curriculum_source_wave: int = 0,
                       curriculum_snapshot_id: str | None = None,
                       paper_blue: float = 0.0, paper_red: float = 0.0,
                       paper_by_wave: np.ndarray | None = None,
                       pbrs_sum: float = 0.0, pbrs_abs_sum: float = 0.0,
                       pbrs_by_wave: np.ndarray | None = None,
                       pbrs_phi_pre: float = 0.0, pbrs_phi_next: float = 0.0,
                       pbrs_samples: int = 0) -> None:
        waves = int(info.get("waves_cleared", 0))
        paper_wave = np.zeros(3, dtype=np.float64) if paper_by_wave is None else np.asarray(paper_by_wave)
        shaped_wave = np.zeros(3, dtype=np.float64) if pbrs_by_wave is None else np.asarray(pbrs_by_wave)
        wave_records = {int(row["wave_index"]): row for row in info.get("per_wave_metrics", [])}
        record = {
            "sampled_steps": int(sampled_steps),
            "episode_length": int(info["episode_length"]),
            "team_raw_environment_return": float(raw_return.sum()),
            "team_training_return": float(training_return.sum()),
            "training_reward": float(training_return.sum()),
            "raw_environment_reward": float(raw_return.sum()),
            "jiao_training_reward": float(training_return.sum()),
            "paper_R2_blue_kill_component": float(paper_blue),
            "paper_R2_red_loss_component": float(paper_red),
            **{f"paper_R2_wave{k}": float(paper_wave[k - 1]) for k in (1, 2, 3)},
            "red_success": float(info["red_success"]),
            "blue_win": float(info["blue_win"]),
            "red_losses": int(info["red_losses"]),
            "blue_losses": int(info["blue_losses"]),
            "red_attack_kills": int(info["red_attack_kills"]),
            "blue_attack_kills": int(info["blue_attack_kills"]),
            "red_boundary_exits": int(info["red_boundary_exits"]),
            "blue_boundary_exits": int(info["blue_boundary_exits"]),
            "red_ground_losses": int(info["red_ground_losses"]),
            "blue_ground_losses": int(info["blue_ground_losses"]),
            "red_weapon_deaths": int(info["blue_attack_kills"]),
            "red_boundary_deaths": int(info["red_boundary_exits"]),
            "red_ground_deaths": int(info["red_ground_losses"]),
            "blue_weapon_deaths": int(info["red_attack_kills"]),
            "blue_boundary_deaths": int(info["blue_boundary_exits"]),
            "blue_ground_deaths": int(info["blue_ground_losses"]),
            "waves_cleared": waves,
            "episode_start_wave": int(episode_start_wave),
            "curriculum_episode": bool(int(episode_start_wave) > 1),
            "curriculum_source_wave": (
                int(curriculum_source_wave) if int(curriculum_source_wave) > 0 else None
            ),
            "curriculum_snapshot_id": curriculum_snapshot_id,
            "additional_waves_cleared": max(
                0, waves - (int(episode_start_wave) - 1)
            ),
            "total_waves": int(info.get("total_waves", 1)),
            **{f"wave_{k}_cleared": float(waves >= k) for k in (1, 2, 3)},
            "red_survivors_enter_wave2": float(wave_records[2]["red_survivors_start"]) if 2 in wave_records else None,
            "red_survivors_enter_wave3": float(wave_records[3]["red_survivors_start"]) if 3 in wave_records else None,
            **{f"wave{k}_clear_step": (int(wave_records[k]["end_step"]) if k in wave_records and wave_records[k]["wave_cleared"] else None) for k in (1,2,3)},
            **{f"time_to_clear_wave{k}": (int(wave_records[k]["duration_steps"]) if k in wave_records and wave_records[k]["wave_cleared"] else None) for k in (1,2,3)},
            "pbrs_shaping_sum": float(pbrs_sum), "pbrs_shaping_abs_sum": float(pbrs_abs_sum),
            "pbrs_phi_pre_mean": float(pbrs_phi_pre / max(1, pbrs_samples)),
            "pbrs_phi_next_mean": float(pbrs_phi_next / max(1, pbrs_samples)),
            **{f"pbrs_shaping_wave{k}": float(shaped_wave[k-1]) for k in (1,2,3)},
            "mission_progress": float(info.get("blue_losses", 0) / max(1, int(info.get("total_waves", 1)) * self.alive.shape[1])),
            "red_survival_ratio": float(info.get("red_survivors", 0) / self.alive.shape[1]),
        }
        with (self.output_dir / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        self.recent_episodes.append(record)
        self.completed_episode_count += 1

    def _iw_record_transition(self, env_id, source_wave, observation, alive_mask, horizon,
                              next_observation, next_alive_mask, next_horizon, spawned_next_wave):
        if not self.trainer.inter_wave_credit.enabled or int(source_wave) not in (1,2):return
        segment=self.iw_pending_episode[int(env_id)][int(source_wave)]
        segment.append({"observation":np.asarray(observation).copy(),"alive_mask":np.asarray(alive_mask).copy(),
                        "horizon":float(horizon),"boundary":False})
        if spawned_next_wave:
            segment.append({"observation":np.asarray(next_observation).copy(),"alive_mask":np.asarray(next_alive_mask).copy(),
                            "horizon":float(next_horizon),"boundary":True})

    def _iw_finalize_episode(self, env_id, waves_cleared):
        completed=[]
        if not self.trainer.inter_wave_credit.enabled:return completed
        group_id=self.iw_completed_episode_counter;self.iw_completed_episode_counter+=1
        for source_wave in (1,2):
            states=self.iw_pending_episode[int(env_id)][source_wave]
            if states:
                segment=self.trainer.inter_wave_credit.cap_segment(
                    states,self.trainer.inter_wave_credit.target(source_wave,int(waves_cleared)),source_wave)
                segment.update({"episode_waves_cleared":int(waves_cleared),"episode_group_id":group_id,
                                "source_env_id":int(env_id)});completed.append(segment)
        self.iw_pending_episode[int(env_id)]={1:[],2:[]}
        return completed

    def _caiw_record_transition(self,env_id,source_wave,observation,alive_mask,action,raw_action,
                                behavior_log_prob,remaining_horizon,wave_cleared_this_step,
                                spawned_next_wave,collection_sampled_steps):
        if not self.trainer.counterfactual_inter_wave_credit.enabled or int(source_wave) not in (1,2):return
        self.caiw_pending_episode[int(env_id)][int(source_wave)].append({
            "observation":np.asarray(observation).copy(),"alive_mask":np.asarray(alive_mask).copy(),
            "action":np.asarray(action).copy(),"raw_action":np.asarray(raw_action).copy(),
            "behavior_log_prob":np.asarray(behavior_log_prob).copy(),"remaining_horizon":float(remaining_horizon),
            "source_wave":int(source_wave),"collection_ppo_update_id":int(self.trainer.ppo_update_count),
            "collection_sampled_steps":int(collection_sampled_steps),
            "wave_cleared_this_step":bool(wave_cleared_this_step),"spawned_next_wave":bool(spawned_next_wave),
        })

    def _caiw_finalize_episode(self,env_id,waves_cleared):
        completed=[]
        if not self.trainer.counterfactual_inter_wave_credit.enabled:return completed
        group_id=self.caiw_completed_episode_counter;self.caiw_completed_episode_counter+=1
        for wave in (1,2):
            rows=self.caiw_pending_episode[int(env_id)][wave]
            if rows:completed.append(self.trainer.counterfactual_inter_wave_credit.cap_segment(rows,wave,int(waves_cleared),group_id,int(env_id)))
        self.caiw_pending_episode[int(env_id)]={1:[],2:[]};return completed

    def _brsc_record_boundary(self,env_id,source_wave,entry_observation,entry_alive_mask,
                              entry_remaining_horizon,spawned_next_wave,collection_sampled_steps):
        if not self.trainer.boundary_redistributed_segment_credit.enabled or not spawned_next_wave:return
        wave=int(source_wave)
        if wave not in (1,2):return
        pending=self.brsc_pending_episode[int(env_id)]
        if pending[wave] is not None:raise RuntimeError(f"duplicate BRSC wave-{wave} boundary in one episode")
        pending[wave]={"source_wave":wave,"entry_observation":np.asarray(entry_observation,dtype=np.float32).copy(),
                       "entry_alive_mask":np.asarray(entry_alive_mask,dtype=np.float32).copy(),
                       "entry_remaining_horizon":float(entry_remaining_horizon),
                       "collection_sampled_steps":int(collection_sampled_steps),
                       "collection_ppo_update_id":int(self.trainer.ppo_update_count),
                       "post_spawn_entry_state":True}

    def _brsc_finalize_episode(self,env_id,waves_cleared):
        completed=[]
        if not self.trainer.boundary_redistributed_segment_credit.enabled:return completed
        group_id=self.brsc_completed_episode_counter;self.brsc_completed_episode_counter+=1
        for wave in (1,2):
            pending=self.brsc_pending_episode[int(env_id)][wave]
            if pending is not None:
                completed.append(self.trainer.boundary_redistributed_segment_credit.complete_boundary(
                    pending,waves_cleared,group_id,int(env_id)))
        self.brsc_pending_episode[int(env_id)]={1:None,2:None};return completed

    def _marc_collect_candidates(self, observations, alive_masks, waves, actions=None) -> None:
        """Keep aligned, strided candidates until their milestone is known.

        V1 intentionally preserves its observation-only pre-action snapshots.
        V2 is called after ``trainer.act`` and stores the exact post-tanh action
        that is subsequently passed to ``env.step``; it never resamples targets.
        """
        module = self.trainer.milestone_aware_retention_credit
        if not module.retention_active:
            return
        if (module.version == 1) != (actions is None):
            return
        for env_id in range(self.num_envs):
            wave = int(waves[env_id])
            if wave not in (1, 2, 3) or int(self.episode_steps[env_id]) % module.retention_stride:
                continue
            rows = self.marc_pending_episode[env_id][wave]
            live = np.asarray(alive_masks[env_id]) > .5
            live_observations = np.asarray(observations[env_id], dtype=np.float32)[live]
            if module.version == 1:
                rows.extend(live_observations)
                # V1 pending storage retains its historical FIFO-compatible bound.
                overflow = len(rows) - module.retention_bank_size_per_wave
                if overflow > 0:
                    del rows[:overflow]
            else:
                live_actions = np.asarray(actions[env_id], dtype=np.float32)[live]
                rows.extend({"observation": observation.copy(), "target_action": action.copy()}
                            for observation, action in zip(live_observations, live_actions))

    def _marc_finalize_step(self, pre_waves, done, infos) -> list[dict[str, Any]]:
        """Commit only successful source-wave candidates; failures never enter banks."""
        module = self.trainer.milestone_aware_retention_credit
        if not module.retention_active:
            return []
        completed = []
        for env_id, info in enumerate(infos):
            wave = int(pre_waves[env_id])
            successful_wave = successful_wave_from_transition(
                wave,
                spawned_next_wave=bool(info.get("spawned_next_wave", False)),
                episode_done=bool(done[env_id]),
                red_success=bool(info.get("red_success", False)),
            )
            if successful_wave is not None:
                rows = self.marc_pending_episode[env_id][successful_wave]
                if rows:
                    if module.version == 1:
                        completed.append({"wave": successful_wave,
                                          "observations": np.asarray(rows, dtype=np.float32)})
                    else:
                        episode_step = int(info.get("episode_length", self.episode_steps[env_id] + 1))
                        duration = episode_step - int(self.marc_wave_start_steps[env_id])
                        if duration <= 0:
                            raise RuntimeError("MARC V2 wave duration is not positive")
                        completed.append({
                            "wave": successful_wave,
                            "observations": np.stack([row["observation"] for row in rows]),
                            "target_actions": np.stack([row["target_action"] for row in rows]),
                            "wave_duration_steps": duration,
                            "red_survivors_after_clear": int(np.asarray(
                                info.get("red_alive_mask", []), dtype=np.float32).sum()),
                        })
                self.marc_pending_episode[env_id][successful_wave] = []
                if module.version == 2 and successful_wave in (1, 2):
                    self.marc_wave_start_steps[env_id] = int(
                        info.get("episode_length", self.episode_steps[env_id] + 1)
                    )
            if bool(done[env_id]):
                self.marc_pending_episode[env_id] = {1: [], 2: [], 3: []}
                if hasattr(self, "marc_wave_start_steps"):
                    self.marc_wave_start_steps[env_id] = 0
        return completed

    def _marc_discard_pending_after_resume(self, extra: dict[str, Any]) -> None:
        """Drop non-checkpointed live-episode candidates after environment restart.

        Completed V1/V2 banks live in trainer checkpoint state and are untouched.
        Pending observation/action pairs cannot be aligned to freshly reset live
        environments, so the fail-closed resume behavior is to count and drop them.
        """
        dropped = sum(int(count) for rows in extra.get("marc_pending_candidate_counts", [])
                      for count in rows.values())
        self.marc_pending_dropped_on_resume = int(
            extra.get("marc_pending_dropped_on_resume", 0)
        ) + dropped
        self.marc_pending_episode = [{1: [], 2: [], 3: []} for _ in range(self.num_envs)]
        self.marc_wave_start_steps = np.zeros(self.num_envs, dtype=np.int64)

    def _collect_wave_entry_snapshots(
        self, infos: list[dict[str, Any]], sampled_steps: int
    ) -> None:
        module = self.trainer.wave_entry_curriculum
        if not module.enabled:
            return
        env_ids = [
            env_id for env_id, info in enumerate(infos)
            if bool(info.get("spawned_next_wave", False))
            and int(self.episode_start_wave[env_id]) == 1
        ]
        if not env_ids:
            return
        snapshots = self.vector.export_curriculum_states(env_ids)
        for env_id in env_ids:
            info = infos[env_id]
            module.add_natural_entry(
                int(info["wave_index"]), snapshots[env_id],
                source_sampled_steps=int(sampled_steps),
                source_training_seed=self.seed,
                source_env_id=env_id,
                source_episode_reset_seed=int(self.episode_reset_seed[env_id]),
                red_survivors=int(info["red_survivors"]),
            )

    def _cause_baseline_from_restore(self, result: dict[str, Any]) -> np.ndarray:
        counts = result["combat_counts"]
        return np.asarray([
            counts["blue"]["attack_kills"], counts["red"]["boundary_exits"],
            counts["red"]["ground_losses"], counts["red"]["attack_kills"],
            counts["blue"]["boundary_exits"], counts["blue"]["ground_losses"],
        ], dtype=np.int64)

    def _apply_wave_entry_curriculum_resets(
        self, done: np.ndarray, infos: list[dict[str, Any]]
    ) -> dict[int, dict[str, Any]]:
        """Replace selected auto-resets and synchronize every runner baseline."""
        module = self.trainer.wave_entry_curriculum
        done_ids = np.flatnonzero(done).tolist()
        if not module.enabled or not done_ids:
            return {}
        for env_id in done_ids:
            if int(self.episode_start_wave[env_id]) == 1:
                module.record_natural_episode(int(infos[env_id].get("waves_cleared", 0)))
        selected: dict[int, dict[str, Any]] = {}
        entries: dict[int, dict[str, Any]] = {}
        for env_id in done_ids:
            wave, entry = module.sample_reset()
            self.episode_start_wave[env_id] = wave
            self.curriculum_source_wave[env_id] = 0 if entry is None else int(entry["entry_wave"])
            self.curriculum_snapshot_id[env_id] = None if entry is None else str(entry["snapshot_id"])
            self.episode_reset_seed[env_id] = int(self.vector.last_reset_seeds[env_id])
            if entry is not None:
                selected[env_id] = entry["snapshot"]
                entries[env_id] = entry
        restored = self.vector.restore_curriculum_states(selected) if selected else {}
        for env_id, result in restored.items():
            self.previous_cause_counts[env_id] = self._cause_baseline_from_restore(result)
        return restored

    def collect_rollout(self, steps: int | None = None) -> ModularRolloutBatch:
        keys = ("observations", "actions", "raw_actions", "old_log_probs", "rewards",
                "raw_environment_rewards", "dones", "alive_masks", "next_observations",
                "next_alive_masks", "wave_indices", "total_waves", "contexts",
                "next_contexts", "actor_hidden_before_step", "critic_hidden_before_step",
                "episode_masks", "remaining_horizons", "next_remaining_horizons",
                "wave_transition_flags", "hta_options", "hta_next_options", "actor_recurrent_phase_reset_flags")
        storage = {key: [] for key in keys}
        completed_iw_segments = []
        completed_caiw_segments = []
        completed_brsc_boundaries = []
        completed_hta_macros = []
        completed_marc_segments = []
        hta_enabled = self.trainer.hierarchical_temporal_abstraction.enabled
        rollout_length = int(steps or self.rollout_steps)
        current_options = current_manager_log_probs = None
        macros = None
        if hta_enabled:
            current_options,current_manager_log_probs,_,_=self.trainer.manager_act(
                self.observations,self.alive,False,True)
            macros=[]
            for env_id in range(self.num_envs):
                macros.append({"observation":self.observations[env_id].copy(),"alive":self.alive[env_id].copy(),
                    "options":current_options[env_id].copy(),"old_log_probs":current_manager_log_probs[env_id].copy(),
                    "reward":np.zeros(self.alive.shape[1],dtype=np.float32),"duration":0,"decision_reason":"rollout_start"})
        rollout_transition = np.zeros(3, dtype=np.int64)
        rollout_alive = np.zeros(3, dtype=np.int64)
        natural_entry_count = np.zeros(2, dtype=np.int64)
        natural_entry_survivor_sum = np.zeros(2, dtype=np.int64)
        reward_rows: list[dict[str, float]] = []
        actor_hidden_norms: list[np.ndarray] = []
        critic_hidden_norms: list[np.ndarray] = []
        context_progress_rows: list[np.ndarray] = []
        context_horizon_rows: list[np.ndarray] = []
        rollout_hidden_resets = 0
        rollout_causes = np.zeros(len(self.death_cause_names), dtype=np.int64)
        rollout_death_indices = np.zeros_like(self.death_index_totals)
        for rollout_index in range(rollout_length):
            obs, alive, pre_wave = self.observations.copy(), self.alive.copy(), self.wave.copy()
            self._marc_collect_candidates(obs, alive, pre_wave)
            blue_alive = self.blue_alive.copy()
            remaining_horizon = np.clip(
                (float(self.runtime_env_config["simulation"]["max_steps"]) - self.episode_steps)
                / float(self.runtime_env_config["simulation"]["max_steps"]), 0.0, 1.0,
            ).astype(np.float32)
            context = mission_context_numpy(
                self.trainer, pre_wave, self.total, blue_alive, self.episode_steps,
                self.runtime_env_config["simulation"]["max_steps"],
            )
            context = self.trainer.actor_context_numpy(pre_wave, alive, context)
            # The mission-progress component is penultimate in mission_markov.
            progress = mission_progress_from_wave_state(pre_wave, blue_alive, self.total)
            context_progress_rows.append(progress.copy())
            context_horizon_rows.append(remaining_horizon.copy())
            phase_flags = self.actor_phase_reset_flags.copy() if self.trainer.recurrent.wave_boundary_reset else None
            if self.trainer.recurrent.wave_boundary_reset:
                self.actor_hidden = self.trainer.prepare_actor_hidden(self.actor_hidden, alive, pre_wave, phase_flags)
            actor_before = None if self.actor_hidden is None else self.actor_hidden.copy()
            critic_before = None if self.critic_hidden is None else self.critic_hidden.copy()
            actions, raw, log_prob, new_actor = self.trainer.act(
                obs, alive, False, True, context, self.actor_hidden, self.episode_mask,
                option_ids=current_options, wave_indices=pre_wave, actor_phase_reset_flags=phase_flags
            )
            self._marc_collect_candidates(obs, alive, pre_wave, actions)
            _, new_critic = self.trainer.values_step(
                obs, alive, context, self.critic_hidden, self.episode_mask,
                option_ids=current_options
            )
            result = self.vector.step_batch(actions)
            done = result.terminated | result.truncated
            training_reward, reward_metrics = self.trainer.reward_adapter.adapt(
                result.rewards, result.infos, pre_wave, alive, blue_alive
            )
            training_reward, pbrs_metrics = self.trainer.wave_survival_pbrs.adapt(
                training_reward, result.infos, pre_wave, alive, blue_alive,
                result.next_alive_masks, done
            )
            reward_metrics.update(pbrs_metrics)
            if hta_enabled and not np.array_equal(training_reward,result.rewards):
                raise RuntimeError("HTA manager/worker training reward must equal raw environment reward")
            pbrs = self.trainer.wave_survival_pbrs.last_transition
            shaping = np.asarray(pbrs["shaping"], dtype=np.float64)
            phi_pre = np.asarray(pbrs["phi_pre"], dtype=np.float64)
            phi_next = np.asarray(pbrs["phi_next"], dtype=np.float64)
            self.pbrs_episode_sum += shaping.sum(axis=1)
            self.pbrs_episode_abs_sum += np.abs(shaping).sum(axis=1)
            self.pbrs_episode_phi_pre += phi_pre.sum(axis=1)
            self.pbrs_episode_phi_next += phi_next.sum(axis=1)
            self.pbrs_episode_samples += shaping.shape[1]
            for k in (1, 2, 3):
                self.pbrs_episode_by_wave[:, k-1] += np.where(pre_wave == k, shaping.sum(axis=1), 0.0)
            self.pbrs_totals += np.asarray([shaping.sum(), np.abs(shaping).sum(),
                *[shaping[pre_wave == k].sum() for k in (1,2,3)]])
            paper = self.trainer.reward_adapter.last_transition
            blue_deaths = np.asarray(paper["blue_death_mask"], dtype=bool)
            red_deaths = np.asarray(paper["red_death_mask"], dtype=bool)
            step_indices = np.stack((blue_deaths.sum(0), red_deaths.sum(0)))
            rollout_death_indices += step_indices
            self.death_index_totals += step_indices
            current_causes = np.asarray([
                [int(info.get(key, 0)) for key in self.death_cause_info_keys]
                for info in result.infos
            ], dtype=np.int64)
            cause_delta = current_causes - self.previous_cause_counts
            # Counters are cumulative within an episode; auto-reset happens
            # after terminal info is returned, so the next baseline is zero.
            cause_delta = np.maximum(cause_delta, 0)
            step_causes = cause_delta.sum(0)
            rollout_causes += step_causes
            self.death_cause_totals += step_causes
            self.previous_cause_counts = np.where(done[:, None], 0, current_causes)
            self.paper_episode_blue += paper["blue_component"]
            self.paper_episode_red += paper["red_component"]
            self.paper_episode_by_wave += paper["per_wave"]
            self.paper_reward_totals += np.asarray([
                paper["blue_component"].sum(), paper["red_component"].sum(),
                paper["per_wave"][:, 0].sum(), paper["per_wave"][:, 1].sum(),
                paper["per_wave"][:, 2].sum(),
            ])
            reward_rows.append(reward_metrics)
            self.reward_bonus_totals += np.asarray([reward_metrics.get("reward_bonus_total",0.0),reward_metrics.get("reward_bonus_wave1",0.0),reward_metrics.get("reward_bonus_wave2",0.0),reward_metrics.get("reward_bonus_wave3",0.0)])
            next_wave = np.asarray([int(row.get("wave_index", 1)) for row in result.infos])
            next_total = np.asarray([int(row.get("total_waves", self.current_waves)) for row in result.infos])
            next_steps = np.asarray([int(row.get("episode_length", 0)) for row in result.infos], dtype=np.int64)
            next_remaining_horizon = np.clip(
                (float(self.runtime_env_config["simulation"]["max_steps"]) - next_steps)
                / float(self.runtime_env_config["simulation"]["max_steps"]), 0.0, 1.0,
            ).astype(np.float32)
            post_blue = np.stack([np.asarray(row["blue_alive_mask"], dtype=np.float32) for row in result.infos])
            next_context = mission_context_numpy(
                self.trainer, next_wave, next_total, post_blue, next_steps,
                self.runtime_env_config["simulation"]["max_steps"],
            )
            next_context = self.trainer.actor_context_numpy(next_wave, result.next_alive_masks, next_context)
            completed_marc_segments.extend(
                self._marc_finalize_step(pre_wave, done, result.infos)
            )
            hta_next_options=None
            if hta_enabled:
                for env_id in range(self.num_envs):
                    macro=macros[env_id]
                    macro["reward"] += (self.trainer.gamma ** macro["duration"]) * result.rewards[env_id]
                    macro["duration"] += 1
                hta_next_options=current_options.copy()
                next_current_options=current_options.copy()
                last_step=rollout_index==rollout_length-1
                spawned=np.asarray([bool(row.get("spawned_next_wave",False)) for row in result.infos])
                periodic=np.asarray([macro["duration"]>=self.trainer.hierarchical_temporal_abstraction.decision_interval_steps for macro in macros])
                boundary=done|spawned|periodic|last_step
                for env_id in np.flatnonzero(boundary):
                    terminal=bool(done[env_id]);wave_end=bool(spawned[env_id]) and not terminal
                    rollout_end=bool(last_step) and not terminal
                    reason=("episode_terminal" if terminal else "wave_transition" if wave_end else
                            "rollout_truncation" if rollout_end else "periodic")
                    macro=macros[env_id];next_alive=np.asarray(result.next_alive_masks[env_id],dtype=np.float32)
                    bootstrap,trace=smdp_boundary_masks(next_alive,episode_terminal=terminal,rollout_truncation=last_step and not terminal)
                    completed_hta_macros.append({"env_id":int(env_id),"observation":macro["observation"],
                        "alive_mask":macro["alive"],"options":macro["options"],"old_log_probs":macro["old_log_probs"],
                        "reward":macro["reward"],"next_observation":np.asarray(result.transition_next_observations[env_id]).copy(),
                        "next_alive_mask":next_alive,"duration":int(macro["duration"]),"bootstrap_mask":bootstrap,
                        "trace_mask":trace,"end_reason":reason,"decision_reason":macro["decision_reason"]})
                if last_step:
                    live=np.flatnonzero(~done)
                    if live.size:
                        sampled=self.trainer.manager_act(result.transition_next_observations[live],result.next_alive_masks[live],False,False)
                        hta_next_options[live]=sampled
                        next_current_options[live]=sampled
                    hta_next_options[done]=0
                else:
                    for env_id in np.flatnonzero(boundary):
                        decision_obs=(result.observations[env_id] if done[env_id] else result.transition_next_observations[env_id])
                        decision_alive=(self.vector.current_alive_masks[env_id] if done[env_id] else result.next_alive_masks[env_id])
                        option,new_log,_,_=self.trainer.manager_act(decision_obs[None],decision_alive[None],False,True)
                        next_current_options[env_id]=option[0]
                        hta_next_options[env_id]=0 if done[env_id] else option[0]
                        macros[env_id]={"observation":np.asarray(decision_obs).copy(),"alive":np.asarray(decision_alive).copy(),
                            "options":option[0].copy(),"old_log_probs":new_log[0].copy(),
                            "reward":np.zeros(self.alive.shape[1],dtype=np.float32),"duration":0,
                            "decision_reason":"episode_reset" if done[env_id] else ("wave_transition" if spawned[env_id] else "periodic")}
            for k in (1, 2, 3):
                transition = pre_wave == k
                rollout_transition[k - 1] += int(transition.sum())
                rollout_alive[k - 1] += int(alive[transition].sum())
                self.transition_counts[k - 1] += int(transition.sum())
                self.alive_agent_counts[k - 1] += int(alive[transition].sum())
            for env_id, info in enumerate(result.infos):
                source_wave = int(pre_wave[env_id])
                if bool(info.get("spawned_next_wave",False)) and source_wave in (1,2):
                    index=source_wave-1
                    natural_entry_count[index]+=1
                    natural_entry_survivor_sum[index]+=int(info["red_survivors"])
                self._iw_record_transition(env_id,source_wave,obs[env_id],alive[env_id],remaining_horizon[env_id],
                    result.transition_next_observations[env_id],result.next_alive_masks[env_id],next_remaining_horizon[env_id],
                    info.get("spawned_next_wave",False))
                self._caiw_record_transition(env_id,source_wave,obs[env_id],alive[env_id],actions[env_id],raw[env_id],log_prob[env_id],remaining_horizon[env_id],info.get("wave_cleared_this_step",False),info.get("spawned_next_wave",False),self.trainer.sampled_steps+self.num_envs)
                self._brsc_record_boundary(env_id,source_wave,result.transition_next_observations[env_id],
                    result.next_alive_masks[env_id],next_remaining_horizon[env_id],
                    info.get("spawned_next_wave",False),self.trainer.sampled_steps+self.num_envs)
                if info.get("wave_cleared_this_step", False):
                    self.wave_clear_transition_counts[max(1, min(3, int(pre_wave[env_id]))) - 1] += 1
            self._collect_wave_entry_snapshots(result.infos, self.trainer.sampled_steps + self.num_envs)
            values = (obs, actions, raw, log_prob, training_reward, result.rewards.copy(),
                      done.astype(np.float32), alive, result.transition_next_observations,
                      result.next_alive_masks, pre_wave, self.total.copy(), context,
                      next_context, actor_before, critic_before, self.episode_mask.copy(),
                      remaining_horizon, next_remaining_horizon,
                      np.asarray([bool(row.get("spawned_next_wave", False)) for row in result.infos], dtype=np.float32),
                      None if not hta_enabled else current_options.copy(),
                      None if not hta_enabled else hta_next_options.copy(), phase_flags)
            for key, value in zip(keys, values):
                storage[key].append(value)
            self.raw_episode_returns += result.rewards
            self.training_episode_returns += training_reward
            step_after = self.trainer.sampled_steps + self.num_envs
            for env_id, is_done in enumerate(done):
                if is_done:
                    completed_iw_segments.extend(self._iw_finalize_episode(env_id,result.infos[env_id].get("waves_cleared",0)))
                    completed_caiw_segments.extend(self._caiw_finalize_episode(env_id,result.infos[env_id].get("waves_cleared",0)))
                    completed_brsc_boundaries.extend(self._brsc_finalize_episode(env_id,result.infos[env_id].get("waves_cleared",0)))
                    self._write_episode(result.infos[env_id], self.raw_episode_returns[env_id],
                                        self.training_episode_returns[env_id], step_after,
                                        episode_start_wave=int(self.episode_start_wave[env_id]),
                                        curriculum_source_wave=int(self.curriculum_source_wave[env_id]),
                                        curriculum_snapshot_id=self.curriculum_snapshot_id[env_id],
                                        paper_blue=self.paper_episode_blue[env_id],
                                        paper_red=self.paper_episode_red[env_id],
                                        paper_by_wave=self.paper_episode_by_wave[env_id],
                                        pbrs_sum=self.pbrs_episode_sum[env_id],
                                        pbrs_abs_sum=self.pbrs_episode_abs_sum[env_id],
                                        pbrs_by_wave=self.pbrs_episode_by_wave[env_id],
                                        pbrs_phi_pre=self.pbrs_episode_phi_pre[env_id],
                                        pbrs_phi_next=self.pbrs_episode_phi_next[env_id],
                                        pbrs_samples=int(self.pbrs_episode_samples[env_id]))
                    self.raw_episode_returns[env_id].fill(0)
                    self.training_episode_returns[env_id].fill(0)
                    self.paper_episode_blue[env_id] = 0.0
                    self.paper_episode_red[env_id] = 0.0
                    self.paper_episode_by_wave[env_id].fill(0.0)
                    self.pbrs_episode_sum[env_id] = 0.0
                    self.pbrs_episode_abs_sum[env_id] = 0.0
                    self.pbrs_episode_by_wave[env_id].fill(0.0)
                    self.pbrs_episode_phi_pre[env_id] = 0.0
                    self.pbrs_episode_phi_next[env_id] = 0.0
                    self.pbrs_episode_samples[env_id] = 0
            restored = self._apply_wave_entry_curriculum_resets(done, result.infos)
            self.observations = result.observations
            if restored:
                self.observations = self.vector.current_observations.copy()
            self.alive = self.vector.current_alive_masks.copy()
            self.blue_alive = np.where(done[:, None], np.ones_like(post_blue), post_blue)
            self.wave = np.where(done, 1, next_wave)
            self.total = np.where(done, self.current_waves, next_total)
            self.episode_steps = np.where(done, 0, next_steps)
            for env_id, metadata in restored.items():
                self.blue_alive[env_id] = np.asarray(metadata["blue_alive_mask"], dtype=np.float32)
                self.wave[env_id] = int(metadata["wave_index"])
                self.total[env_id] = int(metadata["total_waves"])
                self.episode_steps[env_id] = int(metadata["steps"])
            self.episode_mask = (~done).astype(np.float32)
            self.actor_phase_reset_flags = done | np.asarray([bool(row.get("spawned_next_wave", False)) for row in result.infos])
            self.actor_hidden = self.trainer.recurrent.apply_alive(new_actor, self.alive)
            self.critic_hidden = self.trainer.recurrent.apply_alive(new_critic, self.alive)
            self.trainer.recurrent.reset_for_episode(self.actor_hidden, done)
            self.trainer.recurrent.reset_for_episode(self.critic_hidden, done)
            resets = int(done.sum())
            rollout_hidden_resets += resets
            self.hidden_reset_count += resets
            if self.actor_hidden is not None:
                actor_hidden_norms.append(np.linalg.norm(self.actor_hidden, axis=-1))
            if self.critic_hidden is not None:
                critic_hidden_norms.append(np.linalg.norm(self.critic_hidden, axis=-1))
            self.trainer.sampled_steps += self.num_envs
            self.trainer.vector_steps += 1
            if hta_enabled:current_options=next_current_options
        transition_fraction = self._fractions(rollout_transition)
        alive_fraction = self._fractions(rollout_alive)
        self.last_rollout_metrics = {
            **{f"transition_samples_wave_{k}": float(rollout_transition[k-1]) for k in (1,2,3)},
            **{f"transition_fraction_wave_{k}": float(transition_fraction[k-1]) for k in (1,2,3)},
            **{f"alive_agent_samples_wave_{k}": float(rollout_alive[k-1]) for k in (1,2,3)},
            **{f"alive_agent_fraction_wave_{k}": float(alive_fraction[k-1]) for k in (1,2,3)},
            "actor_hidden_norm": float(np.concatenate([x.ravel() for x in actor_hidden_norms]).mean()) if actor_hidden_norms else 0.0,
            "critic_hidden_norm": float(np.concatenate([x.ravel() for x in critic_hidden_norms]).mean()) if critic_hidden_norms else 0.0,
            "actor_hidden_norm_max": float(np.concatenate([x.ravel() for x in actor_hidden_norms]).max()) if actor_hidden_norms else 0.0,
            "critic_hidden_norm_max": float(np.concatenate([x.ravel() for x in critic_hidden_norms]).max()) if critic_hidden_norms else 0.0,
            "hidden_norm_mean": float(np.mean([
                value for rows in (actor_hidden_norms, critic_hidden_norms)
                for array in rows for value in array.ravel()
            ])) if (actor_hidden_norms or critic_hidden_norms) else 0.0,
            "hidden_reset_count": float(rollout_hidden_resets),
            "hidden_reset_count_total": float(self.hidden_reset_count),
            "context_progress_mean": float(np.concatenate(context_progress_rows).mean()),
            "context_remaining_horizon_mean": float(np.concatenate(context_horizon_rows).mean()),
            **{f"context_wave_{k}_fraction": float(transition_fraction[k-1]) for k in (1,2,3)},
            "critic_context_dim": float(self.trainer.critic.context_dim),
            "critic_parameter_count": float(sum(parameter.numel() for parameter in self.trainer.critic.parameters())),
            **{f"blue_deaths_index_{index}": float(rollout_death_indices[0, index]) for index in range(self.death_index_totals.shape[1])},
            **{f"red_deaths_index_{index}": float(rollout_death_indices[1, index]) for index in range(self.death_index_totals.shape[1])},
            **{name: float(rollout_causes[index]) for index, name in enumerate(self.death_cause_names)},
            "red_death_cause_unattributed": float(rollout_death_indices[1].sum() - rollout_causes[:3].sum()),
            "blue_death_cause_unattributed": float(rollout_death_indices[0].sum() - rollout_causes[3:].sum()),
            "natural_entry_count_wave2":float(natural_entry_count[0]),
            "natural_entry_survivor_sum_wave2":float(natural_entry_survivor_sum[0]),
            "natural_entry_count_wave3":float(natural_entry_count[1]),
            "natural_entry_survivor_sum_wave3":float(natural_entry_survivor_sum[1]),
        }
        if reward_rows:
            for key in reward_rows[0]:
                self.last_rollout_metrics[key] = float(np.mean([row[key] for row in reward_rows]))
        if self.trainer.recurrent.state_memory:
            self.last_rollout_metrics.update({
                f"state_memory_one_hot_alive_count_wave_{w}":float(np.asarray(storage["contexts"])[...,w-1].sum()) for w in (1,2,3)})
            self.last_rollout_metrics["state_memory_boundary_resets"] = float(np.asarray(storage["actor_recurrent_phase_reset_flags"]).sum())
        if self.trainer.wave_entry_curriculum.enabled:
            self.last_rollout_metrics.update(self.trainer.wave_entry_curriculum.diagnostics())
        kwargs = {key: (None if not values or values[0] is None else np.asarray(values)) for key, values in storage.items()}
        kwargs["iw_supervision_segments"] = completed_iw_segments
        kwargs["caiw_supervision_segments"] = completed_caiw_segments
        kwargs["brsc_supervision_boundaries"] = completed_brsc_boundaries
        kwargs["marc_success_segments"] = completed_marc_segments
        if hta_enabled:
            if not completed_hta_macros or any(row["duration"]<1 or row["duration"]>16 for row in completed_hta_macros):
                raise RuntimeError("HTA rollout produced invalid or missing closed macros")
            kwargs["hta_manager_transitions"]=ManagerTransitionBatch(
                observations=np.asarray([row["observation"] for row in completed_hta_macros],dtype=np.float32),
                alive_masks=np.asarray([row["alive_mask"] for row in completed_hta_macros],dtype=np.float32),
                options=np.asarray([row["options"] for row in completed_hta_macros],dtype=np.int64),
                old_log_probs=np.asarray([row["old_log_probs"] for row in completed_hta_macros],dtype=np.float32),
                rewards=np.asarray([row["reward"] for row in completed_hta_macros],dtype=np.float32),
                next_observations=np.asarray([row["next_observation"] for row in completed_hta_macros],dtype=np.float32),
                next_alive_masks=np.asarray([row["next_alive_mask"] for row in completed_hta_macros],dtype=np.float32),
                durations=np.asarray([row["duration"] for row in completed_hta_macros],dtype=np.int64),
                bootstrap_masks=np.asarray([row["bootstrap_mask"] for row in completed_hta_macros],dtype=np.float32),
                trace_masks=np.asarray([row["trace_mask"] for row in completed_hta_macros],dtype=np.float32),
                env_ids=np.asarray([row["env_id"] for row in completed_hta_macros],dtype=np.int64),
                end_reasons=np.asarray([row["end_reason"] for row in completed_hta_macros]),
                decision_reasons=np.asarray([row["decision_reason"] for row in completed_hta_macros]))
        if completed_iw_segments:
            kwargs.update({
                "iw_supervision_observations": np.concatenate([x["observations"] for x in completed_iw_segments]),
                "iw_supervision_alive_masks": np.concatenate([x["alive_masks"] for x in completed_iw_segments]),
                "iw_supervision_credit_waves": np.concatenate([np.full(len(x["horizons"]),x["credit_wave"]) for x in completed_iw_segments]),
                "iw_supervision_horizons": np.concatenate([x["horizons"] for x in completed_iw_segments]),
                "iw_supervision_targets": np.concatenate([np.full(len(x["horizons"]),x["target"]) for x in completed_iw_segments]),
                "iw_supervision_segment_ids": np.concatenate([np.full(len(x["horizons"]),x["segment_id"]) for x in completed_iw_segments]),
                "iw_supervision_boundary_flags": np.concatenate([x["boundary_flags"] for x in completed_iw_segments]),
            })
        batch = ModularRolloutBatch(**kwargs)
        if self.trainer.persistent_wave_trajectory_replay.enabled:
            self.trainer.persistent_wave_trajectory_replay.ingest_rollout(
                batch, self.trainer.ppo_update_count
            )
        return batch

    def checkpoint_extra(self, evaluation: dict[str, Any] | None = None) -> dict[str, Any]:
        network = self.algorithm_config["network"]
        value = {
            "environment_version": str(self.env_config.get("environment_version", ENVIRONMENT_VERSION)),
            "environment_variant": self.env_config.get("environment_variant", "direct_v2_3"),
            "observation_dim": int(network["observation_dim"]),
            "action_dim": int(network["action_dim"]),
            "num_agents": int(network["num_agents"]),
            "training_seed": self.seed, "training_gamma": self.trainer.gamma,
            "training_num_envs": self.num_envs,
            "training_total_sampled_steps": self.total_sampled_steps,
            "training_smoke": self.smoke,
            "environment_config_sha256": config_sha256(self.env_config),
            "declared_environment_config_sha256": self.environment_contract["declared"]["config_sha256"],
            "declared_total_waves": self.environment_contract["declared"]["total_waves"],
            "declared_max_steps": self.environment_contract["declared"]["max_steps"],
            "runtime_environment_config_sha256": self.environment_contract["effective_training"]["config_sha256"],
            "runtime_total_waves": self.environment_contract["effective_training"]["total_waves"],
            "runtime_max_steps": self.environment_contract["effective_training"]["max_steps"],
            "evaluation_environment_config_sha256": self.environment_contract["evaluation"]["config_sha256"],
            "evaluation_total_waves": self.environment_contract["evaluation"]["total_waves"],
            "evaluation_max_steps": self.environment_contract["evaluation"]["max_steps"],
            "curriculum_enabled": self.curriculum_enabled,
            **self.environment_provenance(),
            **self.method_identity(),
            "algorithm_config_sha256": config_sha256(self.algorithm_config),
            "environment_config": self.env_config,
            "runtime_environment_config": self.runtime_env_config,
            "algorithm_config": self.algorithm_config,
            "network_architecture": checkpoint_architecture(self.trainer),
            "curriculum_stage": self.current_stage,
            "current_total_waves": self.current_waves,
            "curriculum_config": self.algorithm_config.get("modules", {}).get("curriculum", {}),
            "wave_entry_curriculum_version": self.trainer.wave_entry_curriculum.version,
            "wave_entry_curriculum_config": deepcopy(
                self.algorithm_config.get("modules", {}).get("wave_entry_curriculum", {})
            ),
            "wave_entry_curriculum_state": self.trainer.wave_entry_curriculum.state_dict(),
            "episode_indices": self.vector.episode_indices.tolist(),
            "evaluation_history": self.evaluation_history,
            "best_evaluation": self.best_evaluation,
            "best_sampled_steps": self.best_sampled_steps,
            "transition_counts": self.transition_counts.tolist(),
            "alive_agent_counts": self.alive_agent_counts.tolist(),
            "wave_clear_transition_counts": self.wave_clear_transition_counts.tolist(),
            "reward_bonus_totals": self.reward_bonus_totals.tolist(),
            "wave_survival_pbrs_totals": self.pbrs_totals.tolist(),
            "paper_reward_totals": self.paper_reward_totals.tolist(),
            "death_index_totals": self.death_index_totals.tolist(),
            "death_cause_totals": self.death_cause_totals.tolist(),
            "hidden_reset_count": self.hidden_reset_count,
            "curriculum_transitions": self.curriculum_transitions,
            "resume_count": self.resume_count,
            "branch_provenance": self.branch_provenance,
            **self.runtime_source_checkpoint_provenance(),
            "rng_resume_metadata": deepcopy(self.trainer.rng_restore_metadata),
            "last_optimization_metrics": deepcopy(self.last_metrics),
            "iw_pending_segments_dropped_on_resume": self.iw_pending_segments_dropped_on_resume,
            "iw_pending_segment_counts": [{str(w):len(rows[w]) for w in (1,2)} for rows in self.iw_pending_episode],
            "caiw_completed_episode_counter":self.caiw_completed_episode_counter,
            "caiw_pending_segments_dropped_on_resume":self.caiw_pending_segments_dropped_on_resume,
            "caiw_pending_segment_counts":[{str(w):len(rows[w]) for w in (1,2)} for rows in self.caiw_pending_episode],
            "brsc_completed_episode_counter":self.brsc_completed_episode_counter,
            "brsc_pending_boundaries_dropped_on_resume":self.brsc_pending_boundaries_dropped_on_resume,
            "brsc_pending_boundary_counts":[{str(w):int(rows[w] is not None) for w in (1,2)} for rows in self.brsc_pending_episode],
            "marc_pending_dropped_on_resume":self.marc_pending_dropped_on_resume,
            "marc_pending_candidate_counts":[{str(w):len(rows[w]) for w in (1,2,3)} for rows in self.marc_pending_episode],
            "evaluation": evaluation,
        }
        if self.trainer.fbmr_enabled:
            mode=self.trainer.actor.entity_attention_mode;dual=mode=="frozen_base_dual_bounded_mean_residual"
            value.update({"entity_attention_mode":mode,"base_actor_frozen":True,
                          "entity_mean_residual_enabled":True,"max_mean_correction":self.trainer.actor.max_mean_correction if not dual else None,
                          "dual_bound_enabled":dual,"alpha_abs":self.trainer.actor.alpha_abs if dual else None,
                          "alpha_rel":self.trainer.actor.alpha_rel if dual else None,
                          "log_std_source":"frozen_baseline",**deepcopy(self.trainer.fbmr_branch_metadata),
                          "frozen_base_actor_sha256":self.trainer.frozen_actor_sha256()})
        if self.trainer.hierarchical_temporal_abstraction.enabled:
            value.update({"hta_pending_manager_actor_transitions":0,
                          "hta_manager_actor_updates":self.trainer.manager_actor_update_count,
                          "hta_manager_critic_updates":self.trainer.manager_critic_update_count,
                          "hta_manager_optimizer_steps":self.trainer.manager_optimizer_step_count,
                          "hta_option_usage_counts":self.trainer.hta_option_usage_counts.tolist(),
                          "hta_manager_decision_reason_counts":deepcopy(self.trainer.hta_decision_reason_counts)})
        return value

    def runtime_source_checkpoint_provenance(self) -> dict[str, Any]:
        """Compact runtime-source identity embedded in every checkpoint."""
        return {
            "runtime_source_manifest_sha256": self.runtime_source_manifest["runtime_source_manifest_sha256"],
            "runtime_source_manifest_file_count": self.runtime_source_manifest["runtime_source_manifest_file_count"],
        }

    def save_checkpoint(self, path: str | Path, evaluation: dict[str, Any] | None = None) -> None:
        self.trainer.save(path, self.checkpoint_extra(evaluation))

    def _evaluation_key(self, row: dict[str, Any]) -> tuple[float, ...]:
        return evaluation_selection_key(row, self.env_config.get("environment_variant", "direct_v2_3"))

    def _record_evaluation(self) -> dict[str, Any]:
        row = {"sampled_steps": self.trainer.sampled_steps,
               "evaluation_seed_base": self.eval_base,
               "evaluation_seed_end": self.eval_base + self.eval_episodes - 1,
               **evaluate_modular(
            self.trainer, self.evaluation_env_config, range(self.eval_base, self.eval_base + self.eval_episodes)
        )}
        self.latest_evaluation = row
        self.evaluation_history.append(row)
        path = self.output_dir / "evaluation_history.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.evaluation_history[0]))
            writer.writeheader(); writer.writerows(self.evaluation_history)
        print(self.evaluation_log_line(row,self.total_sampled_steps), flush=True)
        if self.best_evaluation is None or self._evaluation_key(row) > self._evaluation_key(self.best_evaluation):
            old = None if self.best_evaluation is None else self._evaluation_key(self.best_evaluation)
            self.best_evaluation = dict(row); self.best_sampled_steps = self.trainer.sampled_steps
            self.save_checkpoint(self.output_dir / "best_eval.pt", row)
            print(f"[BEST] old={old} | new={self._evaluation_key(row)} | sampled_steps={self.trainer.sampled_steps}", flush=True)
        return row

    def restore_best_from_disk(self, checkpoint_steps: int) -> None:
        path = self.output_dir / "evaluation_history.csv"
        rows: list[dict[str, Any]] = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    converted: dict[str, Any] = {}
                    for key, value in row.items():
                        if key is None:
                            continue
                        text = "" if value is None else value.strip()
                        if not text:
                            if key == "sampled_steps":
                                raise RuntimeError(
                                    "evaluation_history.csv contains an empty sampled_steps value"
                                )
                            converted[key] = None
                        else:
                            converted[key] = int(text) if key == "sampled_steps" else float(text)
                    if converted["sampled_steps"] <= checkpoint_steps:
                        rows.append(converted)
        self.evaluation_history = rows
        if rows:
            self.best_evaluation = max(rows, key=self._evaluation_key)
            self.best_sampled_steps = int(self.best_evaluation["sampled_steps"])
        best_path = self.output_dir / "best_eval.pt"
        if best_path.exists():
            state = torch.load(best_path, map_location="cpu", weights_only=False)
            if int(state.get("sampled_steps", -1)) <= checkpoint_steps:
                validate_modular_checkpoint(state, self.env_config, self.algorithm_config)
                stored = state.get("extra", {}).get("evaluation")
                if stored is not None and (self.best_evaluation is None or self._evaluation_key(stored) >= self._evaluation_key(self.best_evaluation)):
                    self.best_evaluation = stored; self.best_sampled_steps = int(state["sampled_steps"])

    def _restore_checkpoint(self, path: str | Path, *, branch: bool,
                            branch_intervention: str | None = None,
                            source_checkpoint_sha256: str | None = None) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not branch:
            validate_modular_checkpoint(state, self.env_config, self.algorithm_config, {
                "training_seed": self.seed, "training_num_envs": self.num_envs,
                "training_smoke": self.smoke,
            })
        if branch_intervention in {"frozen_base_mean_residual","frozen_base_dual_bounded_mean_residual"}:
            extra=self.trainer.load_fbmr_branch(path,source_checkpoint_sha256,restore_rng=False)
        else:
            extra = self.trainer.load(path, strict_protocol=not branch, restore_rng=False,
                                      defer_reference_attach=bool(branch and self.trainer.reference_variance.enabled))
        self.runtime_env_config = self.trainer.curriculum.runtime_config(self.env_config, self.trainer.sampled_steps)
        if self.curriculum_enabled:
            self.current_stage, _ = self.trainer.curriculum.stage(self.trainer.sampled_steps)
        else:
            self.current_stage = 0
        self.current_waves = int(self.runtime_env_config.get("persistent_waves", {}).get("total_waves", 1))
        self.environment_contract = validate_runtime_environment_contract(
            self.declared_env_config, self.runtime_env_config, self.evaluation_env_config,
            curriculum_enabled=self.curriculum_enabled,
        )
        previous = np.asarray(extra.get("episode_indices", [0] * self.num_envs), dtype=np.int64) + 1
        self.iw_pending_segments_dropped_on_resume = int(extra.get("iw_pending_segments_dropped_on_resume",0)) + sum(
            int(count) > 0 for rows in extra.get("iw_pending_segment_counts",[]) for count in rows.values())
        self.iw_pending_episode = [{1: [], 2: []} for _ in range(self.num_envs)]
        self.caiw_pending_segments_dropped_on_resume = int(extra.get("caiw_pending_segments_dropped_on_resume",0)) + sum(
            int(count)>0 for rows in extra.get("caiw_pending_segment_counts",[]) for count in rows.values())
        self.caiw_completed_episode_counter=int(extra.get("caiw_completed_episode_counter",0))
        self.caiw_pending_episode=[{1:[],2:[]} for _ in range(self.num_envs)]
        self.brsc_pending_boundaries_dropped_on_resume = int(extra.get("brsc_pending_boundaries_dropped_on_resume",0)) + sum(
            int(count)>0 for rows in extra.get("brsc_pending_boundary_counts",[]) for count in rows.values())
        self.brsc_completed_episode_counter=int(extra.get("brsc_completed_episode_counter",0))
        self.brsc_pending_episode=[{1:None,2:None} for _ in range(self.num_envs)]
        self._marc_discard_pending_after_resume(extra)
        self._make_vector(previous)
        if self.trainer.persistent_wave_trajectory_replay.enabled:
            self.trainer.persistent_wave_trajectory_replay.discard_pending_after_environment_restart()
        if self.trainer.wave_entry_curriculum.enabled:
            saved_wec = extra.get("wave_entry_curriculum_state")
            if saved_wec is None:
                raise RuntimeError("wave-entry curriculum checkpoint is missing its state")
            self.trainer.wave_entry_curriculum.load_state_dict(saved_wec)
        self.trainer.restore_rng_state(state)
        if branch and self.trainer.reference_variance.enabled:
            if not self.trainer.finalize_reference_variance_branch():
                raise RuntimeError("RV branch reference actor was not created after source RNG restoration")
        if branch_intervention in {"frozen_base_mean_residual","frozen_base_dual_bounded_mean_residual"}:
            restored=bool(self.trainer.rng_restore_metadata["rng_state_restored"])
            self.trainer.fbmr_branch_metadata["source_rng_restored"]=restored
            self.trainer.fbmr_branch_metadata["RNG_restored_from_source"]=restored
        self.transition_counts = np.asarray(extra.get("transition_counts", [0,0,0]), dtype=np.int64)
        self.alive_agent_counts = np.asarray(extra.get("alive_agent_counts", [0,0,0]), dtype=np.int64)
        self.wave_clear_transition_counts = np.asarray(extra.get("wave_clear_transition_counts", [0,0,0]), dtype=np.int64)
        self.reward_bonus_totals = np.asarray(extra.get("reward_bonus_totals", [0,0,0,0]), dtype=np.float64)
        self.pbrs_totals = np.asarray(extra.get("wave_survival_pbrs_totals", [0,0,0,0,0]), dtype=np.float64)
        self.paper_reward_totals = np.asarray(extra.get("paper_reward_totals", [0,0,0,0,0]), dtype=np.float64)
        self.death_index_totals = np.asarray(extra.get("death_index_totals", np.zeros((2,self.alive.shape[1])).tolist()), dtype=np.int64)
        self.death_cause_totals = np.asarray(extra.get("death_cause_totals", [0]*len(self.death_cause_names)), dtype=np.int64)
        self.hidden_reset_count = int(extra.get("hidden_reset_count", 0))
        self.curriculum_transitions = list(extra.get("curriculum_transitions", self.curriculum_transitions))
        self.resume_count = 0 if branch else int(extra.get("resume_count", 0)) + 1
        if not branch:self.restore_best_from_disk(self.trainer.sampled_steps)
        self.next_console = (self.trainer.sampled_steps // self.console_interval + 1) * self.console_interval
        self.next_evaluation = (self.trainer.sampled_steps // self.evaluation_interval + 1) * self.evaluation_interval
        self.next_checkpoint = (self.trainer.sampled_steps // self.checkpoint_interval + 1) * self.checkpoint_interval

    def resume(self, path: str | Path) -> None:
        self._restore_checkpoint(path, branch=False)

    def branch_from(self, path: str | Path, intervention: str | None = None,
                    source_checkpoint_sha256: str | None = None) -> None:
        self._restore_checkpoint(path, branch=True,branch_intervention=intervention,
                                 source_checkpoint_sha256=source_checkpoint_sha256)

    def startup_summary(self) -> dict[str, Any]:
        ids = self.environment_contract
        return {"algorithm":"modular_mappo","mode":"smoke" if self.smoke else "formal",
                **self.method_identity(),
                "device":self.device,"seed":self.seed,"num_envs":self.num_envs,
                "total_sampled_steps":self.total_sampled_steps,"rollout_steps":self.rollout_steps,
                "gamma":self.trainer.gamma,"enabled_modules":self.trainer.module_protocol()["enabled_modules"],
                "environment_variant":self.env_config.get("environment_variant","direct_v2_3"),
                "curriculum_enabled":self.curriculum_enabled,
                "declared_waves":ids["declared"]["total_waves"],
                "declared_max_steps":ids["declared"]["max_steps"],
                "declared_environment_config_sha256":ids["declared"]["config_sha256"],
                "effective_training_waves":ids["effective_training"]["total_waves"],
                "effective_training_max_steps":ids["effective_training"]["max_steps"],
                "runtime_environment_config_sha256":ids["effective_training"]["config_sha256"],
                "evaluation_waves":ids["evaluation"]["total_waves"],
                "evaluation_max_steps":ids["evaluation"]["max_steps"],
                "evaluation_environment_config_sha256":ids["evaluation"]["config_sha256"],
                "branch_mode":bool(self.branch_provenance)}

    @staticmethod
    def _compact_steps(value: int) -> str:
        return f"{value / 1_000_000:.2f}M" if value >= 1_000_000 else f"{value / 1_000:.0f}k"

    def startup_console_lines(self) -> tuple[str, str]:
        method=self.method_identity();modules=",".join(self.trainer.module_protocol()["enabled_modules"]) or "none"
        return (
            f"[START] method={method['development_method']} seed={self.seed} device={self.device} "
            f"envs={self.num_envs} steps={self.total_sampled_steps} rollout={self.rollout_steps}" +
            (f" factorial_variant={method['factorial_variant']} continuation_alpha={method['continuation_alpha']} wave_balance_temperature={method['wave_balance_temperature']}" if 'factorial_variant' in method else ''),
            f"[PROTOCOL] env={self.env_config.get('environment_variant','direct_v2_3')} "
            f"waves={self.current_waves} max_steps={self.runtime_env_config['simulation']['max_steps']} "
            f"actor_input={method['actor_input_dim']} actor_ctx={method['actor_context_dim']} critic_ctx={method['critic_context_dim']} "
            f"wave_context={method['wave_context_target']}/{method['wave_context_encoding']}/{method['mission_context_dim']}D "
            f"{('mission_film='+method['mission_film_mode']+'/'+str(method['mission_encoder_hidden_dim'])+'D/a'+str(method['mission_film_alpha'])+' ') if method.get('mission_film_enabled') else ''}"
            f"modules={modules} eval={self.eval_episodes}",
        )

    def train_log_line(self) -> str:
        rows = list(self.recent_episodes)
        mean = lambda key: float(np.mean([row[key] for row in rows])) if rows else float("nan")
        percent=100.0*self.trainer.sampled_steps/max(self.total_sampled_steps,1)
        line=(f"[TRAIN] {percent:.1f}% | steps={self._compact_steps(self.trainer.sampled_steps)}/{self._compact_steps(self.total_sampled_steps)} "
              f"| ep={self.completed_episode_count} | R={mean('team_raw_environment_return'):.2f} "
              f"| waves={mean('waves_cleared'):.2f} | red/blue={mean('red_losses'):.2f}/{mean('blue_losses'):.2f} "
              f"| actor={self.last_metrics.get('actor_loss',float('nan')):.4f} "
              f"| value={self.last_metrics.get('value_loss',float('nan')):.3f} "
              f"| H={self.last_metrics.get('entropy',float('nan')):.3f} "
              f"| KL={self.last_metrics.get('approx_kl',float('nan')):.4f} "
              f"| lr={self.last_metrics.get('actor_learning_rate',float('nan')):.1e}")
        extras=[]
        if self.trainer.recurrent.enabled:
            extras.append(f"recurrent_h={self.last_metrics.get('hidden_norm_mean',0):.3f} resets={self.last_metrics.get('hidden_reset_count',0):.0f} chunks={self.last_metrics.get('sequence_chunks',0):.0f} gru_grad={self.last_metrics.get('gru_gradient_norm',0):.3f}")
        if self.algorithm_config.get('development_method')=='jiao2025_core_3m':
            extras.append(f"rmb={self.last_metrics.get('recurrent_minibatches_per_epoch',0):.0f} optA/C={self.last_metrics.get('actor_optimizer_steps_this_update',0):.0f}/{self.last_metrics.get('critic_optimizer_steps_this_update',0):.0f} GRUA/C={self.last_metrics.get('actor_gru_grad_norm',0):.3f}/{self.last_metrics.get('critic_gru_grad_norm',0):.3f} hA/C={self.last_metrics.get('actor_hidden_norm',0):.3f}/{self.last_metrics.get('critic_hidden_norm',0):.3f}")
        if self.trainer.popart.enabled:extras.append(f"popart_std={self.last_metrics.get('popart_std',1):.3f}")
        if self.trainer.wave_balance.enabled:extras.append(f"wave_weight={self.last_metrics.get('effective_wave_weight_mean',1):.3f}")
        if self.trainer.reward_adapter.enabled:extras.append(f"bonus={self.reward_bonus_totals[0]:.2f}")
        if self.trainer.wave_survival_pbrs.enabled:extras.append(f"shaping={self.pbrs_totals[0]:.2f}")
        if self.trainer.anchor.enabled:extras.append(f"anchor_KL={self.last_metrics.get('anchor_kl',0):.4f}")
        if self.curriculum_enabled:extras.append(f"stage={self.current_stage} waves={self.current_waves}")
        wave_entry_curriculum = getattr(self.trainer, "wave_entry_curriculum", None)
        if wave_entry_curriculum is not None and wave_entry_curriculum.enabled:
            diagnostics=wave_entry_curriculum.diagnostics()
            extras.append(
                f"natural_waves={diagnostics['wec_recent_natural_aw']:.2f} "
                f"reset_mix={diagnostics['wec_reset_fraction_w1']:.2f}/"
                f"{diagnostics['wec_reset_fraction_w2']:.2f}/"
                f"{diagnostics['wec_reset_fraction_w3']:.2f}"
            )
        return line+(" | "+" | ".join(extras) if extras else "")

    def optimization_warning_line(self) -> str | None:
        keys=("actor_loss","value_loss","entropy","approx_kl")
        nonfinite=any(not math.isfinite(float(self.last_metrics.get(key,float("nan")))) for key in keys)
        kl=float(self.last_metrics.get("approx_kl",float("nan")))
        underflow=float(self.last_metrics.get("ratio_underflow_fraction",0.0))
        if not (nonfinite or (math.isfinite(kl) and kl>=.05) or underflow>0):return None
        return (f"[OPT_WARN] steps={self.trainer.sampled_steps} | KL={kl:.5f} "
                f"| H={self.last_metrics.get('entropy',float('nan')):.3f} "
                f"| logR=[{self.last_metrics.get('log_ratio_min',float('nan')):.2f},"
                f"{self.last_metrics.get('log_ratio_max',float('nan')):.2f}] | underflow={underflow:.4f}")

    @staticmethod
    def evaluation_log_line(row: dict[str, Any], total_sampled_steps: int | None = None) -> str:
        w1,w2,w3=(row.get(f"clear_wave_{k}_probability") for k in (1,2,3))
        q2=float("nan") if w1 in (None,0) or w2 is None else float(w2)/float(w1)
        q3=float("nan") if w2 in (None,0) or w3 is None else float(w3)/float(w2)
        prefix=(f"{100.0*int(row['sampled_steps'])/total_sampled_steps:.1f}% | " if total_sampled_steps else "")
        return (f"[EVAL] {prefix}steps={ModularMAPPOTrainingRunner._compact_steps(int(row['sampled_steps']))} | W1/W2/W3="
                f"{row.get('clear_wave_1_probability',0):.2f}/{row.get('clear_wave_2_probability',0):.2f}/{row.get('clear_wave_3_probability',0):.2f} "
                f"| Q2/Q3={q2:.2f}/{q3:.2f} | waves={row.get('average_waves_cleared',0):.2f} | R={row['average_return']:.2f} "
                f"| red/blue={row['average_red_loss']:.2f}/{row['average_blue_loss']:.2f} "
                f"| boundary={row['average_red_boundary_exits']:.2f} "
                f"| ground={row['average_red_ground_losses']:.2f}")

    def summary(self) -> dict[str, Any]:
        transition_fraction = self._fractions(self.transition_counts)
        alive_fraction = self._fractions(self.alive_agent_counts)
        pretraining = int(self.trainer.warm_start_provenance.get("pretraining_sampled_steps", 0))
        guard_updates = int(self.trainer.ppo_update_count)
        guard_epoch_total = int(self.trainer.actor_kl_guard_actor_epochs_total)
        return {
            "algorithm":"modular_mappo",
            **self.method_identity(),
            "modular_mappo_impl_version":MODULAR_MAPPO_IMPL_VERSION,
            "baseline_mappo_impl_version":MAPPO_IMPL_VERSION,
            "protocol": {**self.trainer.module_protocol(), "network_architecture":checkpoint_architecture(self.trainer),
                         "environment_config_sha256":config_sha256(self.env_config),
                         **self.environment_provenance(),
                         "algorithm_config_sha256":config_sha256(self.algorithm_config)},
            "sampled_steps": self.trainer.sampled_steps,
            "current_pw_training_sampled_steps": self.trainer.sampled_steps,
            "pretraining_sampled_steps": pretraining,
            "effective_total_experience_budget": pretraining + self.trainer.sampled_steps,
            "best_checkpoint_step": self.best_sampled_steps,
            "best_evaluation": self.best_evaluation,
            "latest_step": self.trainer.sampled_steps,
            "latest_evaluation": self.latest_evaluation,
            "completed_episodes": self.completed_episode_count,
            "wave_transition_counts": {f"wave_{k}":int(self.transition_counts[k-1]) for k in (1,2,3)},
            "wave_transition_fractions": {f"wave_{k}":float(transition_fraction[k-1]) for k in (1,2,3)},
            "wave_alive_agent_sample_counts": {f"wave_{k}":int(self.alive_agent_counts[k-1]) for k in (1,2,3)},
            "wave_alive_agent_sample_fractions": {f"wave_{k}":float(alive_fraction[k-1]) for k in (1,2,3)},
            "wave_clear_transition_counts": {f"wave_{k}":int(self.wave_clear_transition_counts[k-1]) for k in (1,2,3)},
            "reward_adapter_totals": {"reward_bonus_total":float(self.reward_bonus_totals[0]),**{f"reward_bonus_wave{k}":float(self.reward_bonus_totals[k]) for k in (1,2,3)}},
            "wave_survival_pbrs_totals": {"shaping_sum":float(self.pbrs_totals[0]),"shaping_abs_sum":float(self.pbrs_totals[1]),**{f"shaping_wave{k}":float(self.pbrs_totals[k+1]) for k in (1,2,3)}},
            "paper_R2_totals": {"blue_kill_component":float(self.paper_reward_totals[0]),
                                "red_loss_component":float(self.paper_reward_totals[1]),
                                **{f"wave_{k}":float(self.paper_reward_totals[k+1]) for k in (1,2,3)}},
            "death_index_totals": {
                **{f"blue_deaths_index_{index}":int(self.death_index_totals[0,index]) for index in range(self.death_index_totals.shape[1])},
                **{f"red_deaths_index_{index}":int(self.death_index_totals[1,index]) for index in range(self.death_index_totals.shape[1])},
            },
            "death_cause_totals": {name:int(self.death_cause_totals[index]) for index,name in enumerate(self.death_cause_names)},
            "hidden_reset_count": self.hidden_reset_count,
            "module_protocol": self.trainer.module_protocol(),
            "warm_start_provenance": self.trainer.warm_start_provenance,
            "anchor_provenance": self.trainer.anchor_provenance,
            "curriculum_transitions": self.curriculum_transitions,
            "wave_entry_curriculum": (
                self.trainer.wave_entry_curriculum.diagnostics()
                if self.trainer.wave_entry_curriculum.enabled else {"enabled": False}
            ),
            "resume_count": self.resume_count,
            "branch_provenance": self.branch_provenance,
            "rng_resume_metadata": deepcopy(self.trainer.rng_restore_metadata),
            "final_optimization_metrics": self.last_metrics,
            "actor_kl_guard_summary": {
                "enabled": bool(self.trainer.actor_kl_guard.enabled),
                "total_ppo_updates": guard_updates,
                "hard_stop_count": int(self.trainer.actor_kl_guard_hard_stop_count),
                "hard_stop_fraction": (float(self.trainer.actor_kl_guard_hard_stop_count / guard_updates)
                                       if guard_updates else 0.0),
                "mean_actor_epochs_used": (float(guard_epoch_total / guard_updates)
                                           if self.trainer.actor_kl_guard.enabled and guard_updates else None),
                "min_actor_epochs_used": (int(self.trainer.actor_kl_guard_actor_epochs_min)
                                          if self.trainer.actor_kl_guard_actor_epochs_min is not None else None),
            },
            **self.environment_provenance(),
        }

    def environment_provenance(self) -> dict[str, Any]:
        ids = self.environment_contract
        return {
            "curriculum_enabled": self.curriculum_enabled,
            "declared_environment_config_sha256": ids["declared"]["config_sha256"],
            "declared_total_waves": ids["declared"]["total_waves"],
            "declared_max_steps": ids["declared"]["max_steps"],
            "effective_training_environment_config_sha256": ids["effective_training"]["config_sha256"],
            "effective_training_total_waves": ids["effective_training"]["total_waves"],
            "effective_training_max_steps": ids["effective_training"]["max_steps"],
            "evaluation_environment_config_sha256": ids["evaluation"]["config_sha256"],
            "evaluation_total_waves": ids["evaluation"]["total_waves"],
            "evaluation_max_steps": ids["evaluation"]["max_steps"],
        }

    def method_identity(self) -> dict[str, Any]:
        architecture = checkpoint_architecture(self.trainer)
        result = {
            "development_method": self.algorithm_config.get("development_method", "modular_mappo"),
            "observation_schema": (
                "fire_ready_v1"
                if bool(getattr(self, "env_config", {}).get("observation", {}).get("include_own_fire_ready", False))
                else "legacy_paper_52d"
            ),
            "wave_context_target": self.trainer.wave_context.target if self.trainer.wave_context.enabled else "disabled",
            "wave_context_encoding": self.trainer.wave_context.encoding if self.trainer.wave_context.enabled else "disabled",
            "mission_context_dim": int(self.trainer.wave_context.context_dim),
            "actor_input_dim": int(architecture["actor_input_dim"]),
            "actor_context_dim": int(architecture["actor_context_dim"]),
            "critic_context_dim": int(architecture["critic_context_dim"]),
        }
        if self.algorithm_config.get("development_method")=="marc_core_factorial_v1":
            module=self.trainer.milestone_aware_retention_credit
            result.update(factorial_variant=self.algorithm_config["factorial_variant"],
                          continuation_alpha=module.continuation_alpha,wave_balance_temperature=module.wave_balance_temperature)
        if self.algorithm_config.get("development_method") in {"jiao2025_matched_plain_3m","jiao2025_core_3m"}:
            from .protocol import JIAO_3M_SOURCE_CLASSIFICATION
            result.update(jiao_transfer=deepcopy(self.algorithm_config["jiao_transfer"]),
                          jiao_source_classification=deepcopy(JIAO_3M_SOURCE_CLASSIFICATION),
                          recurrent_mode=self.trainer.recurrent.mode if self.trainer.recurrent.enabled else "disabled",
                          recurrent_sequence_length=self.trainer.recurrent.sequence_length if self.trainer.recurrent.enabled else 0,
                          hidden_reset_protocol="true_episode_only_dead_agent_mask_no_wave_reset",
                          popart_enabled=self.trainer.popart.enabled)
        if self.trainer.mission_film.enabled:
            result.update({key:architecture[key] for key in ("mission_film_enabled","mission_film_mode",
                "mission_encoder_hidden_dim","mission_film_alpha","mission_film_identity_init",
                "mission_film_augmented_residual")})
        if self.trainer.actor_kl_guard.enabled:
            result.update({"actor_kl_guard_enabled":True,
                           "actor_kl_guard_version":int(self.trainer.actor_kl_guard.version),
                           "actor_kl_guard_hard_kl":float(self.trainer.actor_kl_guard.hard_kl),
                           "actor_kl_guard_actor_early_stop":bool(self.trainer.actor_kl_guard.actor_early_stop)})
        if self.trainer.wave_entry_curriculum.enabled:
            result.update({
                "wave_entry_curriculum_enabled": True,
                "wave_entry_curriculum_version": int(self.trainer.wave_entry_curriculum.version),
                "wave_entry_curriculum_config": deepcopy(self.trainer.wave_entry_curriculum.config),
                "bank_capacity": int(self.trainer.wave_entry_curriculum.capacity_per_wave),
                "adaptive_target_reach_w2": float(self.trainer.wave_entry_curriculum.target_reach_w2),
                "adaptive_target_reach_w3": float(self.trainer.wave_entry_curriculum.target_reach_w3),
                "max_curriculum_fraction": float(self.trainer.wave_entry_curriculum.max_curriculum_fraction),
            })
        if self.trainer.hierarchical_temporal_abstraction.enabled:
            result.update({key:architecture[key] for key in (
                "hierarchical_temporal_abstraction_enabled","hta_version","manager_actor_class",
                "manager_critic_class","manager_observation_dim","num_options",
                "decision_interval_steps","worker_option_conditioning",
                "tactical_critic_option_conditioning","deployment_requires_manager")})
        if self.trainer.hta_worker_consolidation.enabled:
            result.update({key:architecture[key] for key in (
                "hta_worker_consolidation_enabled","hta_worker_consolidation_version",
                "worker_learning_timescale","worker_consolidation_start_step",
                "worker_consolidation_end_step","worker_final_lr_multiplier",
                "manager_learning_schedule","tactical_critic_learning_schedule")})
        if self.trainer.sequential_wave_gradient_projection.enabled:
            module=self.trainer.sequential_wave_gradient_projection
            result.update({"sequential_wave_gradient_projection_enabled":True,
                           "swgp_version":1,"swgp_mode":module.mode,"swgp_epsilon":module.epsilon,
                           "swgp_activation_start_step":module.activation_start_step,
                           "swgp_first_activation_sampled_steps":module.first_activation_sampled_steps,
                           "swgp_pre_activation_plain_update_count":module.pre_activation_plain_update_count,
                           "swgp_activation_update_count":module.activation_update_count,
                           "swgp_actor_only":True,"swgp_natural_wave_frequency":True,
                           "swgp_critic_unchanged":True})
        if self.trainer.team_mean_credit.enabled:
            module = self.trainer.team_mean_credit
            result.update({
                "team_mean_credit_enabled": True,
                "team_mean_credit_version": int(module.version),
                "team_mean_credit_mode": module.mode,
                "team_mean_credit_reward_scope": "training_credit_only",
                "team_mean_credit_sum_preserving": True,
            })
        if self.trainer.persistent_wave_trajectory_replay.enabled:
            module = self.trainer.persistent_wave_trajectory_replay
            result.update({
                "pwtr_enabled": True,
                "pwtr_version": int(module.version),
                "pwtr_fresh_wave_stratification": module.fresh_wave_stratification,
                "pwtr_replay_enabled": module.replay_enabled,
                "pwtr_replay_source": module.replay_source,
                "pwtr_priority_enabled": module.priority_enabled,
                "pwtr_bridge_enabled": module.bridge_enabled,
                "pwtr_natural_trajectory_only": True,
                "pwtr_reward_scope": "original_local_environment_reward",
            })
        if self.trainer.wave_specific_actor_isolation.enabled:
            module=self.trainer.wave_specific_actor_isolation
            result.update({"wave_specific_actor_isolation_enabled":True,"wsai_version":int(module.version),
                           "wave_actor_count":3,"wave_actor_parameter_sharing":False,
                           "wave_actor_routing":"environment_wave","shared_critic":True,
                           "wsai_natural_wave_weighting":True,
                           "wsai_global_actor_grad_clip":True})
        if self.trainer.wave_specific_mean_heads.enabled:
            module=self.trainer.wave_specific_mean_heads
            result.update({"wave_specific_mean_heads_enabled":True,"wsmh_version":int(module.version),
                           "wave_mean_head_count":3,"shared_backbone":True,"shared_log_std":True,
                           "wave_actor_routing":"environment_wave","shared_critic":True,
                           "wsmh_natural_wave_weighting":True,"wsmh_global_actor_grad_clip":True})
        if self.trainer.actor_gradient_clipping.enabled:
            module=self.trainer.actor_gradient_clipping
            result.update({"actor_gradient_clipping_enabled":True,"actor_grad_clip_version":int(module.version),
                           "actor_grad_clip_mode":module.mode,"actor_max_grad_norm":module.actor_max_grad_norm,
                           "critic_max_grad_norm":self.trainer.max_grad_norm,"actor_clip_only_intervention":True,
                           "network_topology_unchanged":True})
        return result

    def run(self) -> dict[str, Any]:
        for line in self.startup_console_lines():print(line,flush=True)
        try:
            while self.trainer.sampled_steps < self.total_sampled_steps:
                self._maybe_curriculum()
                remaining = math.ceil((self.total_sampled_steps - self.trainer.sampled_steps) / self.num_envs)
                rollout = self.collect_rollout(min(self.rollout_steps, remaining))
                self.last_metrics = {**self.trainer.update(rollout), **self.last_rollout_metrics,
                                     "curriculum_stage":float(self.current_stage),
                                     "current_total_waves":float(self.current_waves)}
                if self.last_metrics.get("swgp_activated_this_update",0.0)>0.5:
                    print(f"[SWGP_ACTIVATE] configured_step={int(self.last_metrics['swgp_activation_start_step'])} "
                          f"actual_step={int(self.last_metrics['swgp_first_activation_sampled_steps'])}",flush=True)
                if (self.trainer.actor_kl_guard.enabled and
                        self.last_metrics.get("kl_hard_stop_triggered", 0.0) > 0.5):
                    print(f"[KL_GUARD] steps={self.trainer.sampled_steps} | "
                          f"epoch_kl={self.last_metrics['epoch_kl_last']:.5f} | "
                          f"actor_epochs={int(self.last_metrics['actor_epochs_used'])}/{self.trainer.ppo_epochs}",
                          flush=True)
                warning=self.optimization_warning_line()
                if warning is not None:print(warning,flush=True)
                record = {"sampled_steps":self.trainer.sampled_steps,"rollout_update":self.trainer.ppo_update_count,**self.last_metrics}
                with (self.output_dir / "optimization_metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                if self.trainer.sampled_steps >= self.next_console:
                    print(self.train_log_line(), flush=True)
                    while self.next_console <= self.trainer.sampled_steps:self.next_console += self.console_interval
                if self.trainer.sampled_steps >= self.next_evaluation:
                    self._record_evaluation()
                    while self.next_evaluation <= self.trainer.sampled_steps:self.next_evaluation += self.evaluation_interval
                if self.trainer.sampled_steps >= self.next_checkpoint:
                    path = self.output_dir / f"checkpoint_{self.trainer.sampled_steps}.pt";self.save_checkpoint(path)
                    print(f"[CHECKPOINT] path={path.name} | sampled_steps={self.trainer.sampled_steps}", flush=True)
                    while self.next_checkpoint <= self.trainer.sampled_steps:self.next_checkpoint += self.checkpoint_interval
            if self.latest_evaluation is None or int(self.latest_evaluation["sampled_steps"]) != self.trainer.sampled_steps:
                self._record_evaluation()
            self.save_checkpoint(self.output_dir / "latest.pt")
            self.save_checkpoint(self.output_dir / "final.pt")
            result = self.summary()
            (self.output_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"[DONE] sampled_steps={self.trainer.sampled_steps} | best={self.best_sampled_steps} | latest={self.trainer.sampled_steps}", flush=True)
            return result
        finally:
            self.vector.close()


__all__ = ["ModularMAPPOTrainingRunner", "environment_runtime_identity",
           "validate_runtime_environment_contract"]
