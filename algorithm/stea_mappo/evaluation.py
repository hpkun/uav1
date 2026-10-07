"""Deterministic recurrent evaluation with strict holdout provenance."""
from pathlib import Path
import numpy as np
import torch
from env.factory import make_combat_environment
from algorithm.common.evaluator import aggregate_combat_records, episode_return_metrics
from algorithm.common.protocol import config_sha256
from .factory import build_stea_mappo_trainer
from .protocol import require_cuda, validate_checkpoint
from .trainer import STEA_MAPPO_IMPL_VERSION


def evaluate(trainer, env_config, seeds):
    require_cuda(trainer.device)
    records, diagnostics = [], []
    for seed in seeds:
        env = make_combat_environment(env_config)
        observation,_ = env.reset(int(seed))
        hidden = np.zeros((env.team_size,trainer.actor.gru_hidden_dim),dtype=np.float32)
        start = np.array(1.,dtype=np.float32)
        returns = np.zeros(env.team_size)
        while True:
            actions,_,_,hidden,stats = trainer.act(observation,env.red_alive_mask,
                hidden,start,deterministic=True,return_diagnostics=True)
            diagnostics.append(stats)
            observation,reward,terminated,truncated,info = env.step(actions)
            returns += reward
            hidden *= env.red_alive_mask[:,None]
            start = np.array(0.,dtype=np.float32)
            if terminated or truncated:
                hidden.fill(0.)
                team,agent = episode_return_metrics(returns)
                records.append({"episode_return":team,"mean_agent_episode_return":agent,**info})
                break
    result = aggregate_combat_records(records)
    if diagnostics:
        result.update({key:float(np.mean([row[key] for row in diagnostics])) for key in diagnostics[0]})
    return result


def evaluate_stea_mappo_checkpoint(checkpoint_path, algorithm_config, environment_config,
                                  device, evaluation_seeds):
    require_cuda(device)
    seeds = [int(seed) for seed in evaluation_seeds]
    if not seeds or any(b != a+1 for a,b in zip(seeds,seeds[1:])):
        raise ValueError("evaluation seeds must be non-empty, increasing and contiguous")
    state = torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    extra = validate_checkpoint(state,environment_config,algorithm_config)
    training_end = int(extra["training_seed"])+int(extra["training_total_sampled_steps"])+int(extra["training_num_envs"])
    if any(int(extra["training_seed"]) <= seed <= training_end for seed in seeds):
        raise ValueError("holdout seeds overlap the conservative training seed range")
    trainer = build_stea_mappo_trainer(algorithm_config,device,
        seed=extra["training_seed"],smoke=extra["training_smoke"])
    trainer.load(checkpoint_path)
    result = evaluate(trainer,environment_config,seeds)
    result.update({"algorithm":"STEA-MAPPO","stea_mappo_impl_version":STEA_MAPPO_IMPL_VERSION,
        "network_architecture":dict(trainer.network_architecture),
        "checkpoint":str(Path(checkpoint_path).resolve()),"checkpoint_sampled_steps":int(state["sampled_steps"]),
        "checkpoint_training_seed":extra["training_seed"],"checkpoint_training_gamma":extra["training_gamma"],
        "checkpoint_training_num_envs":extra["training_num_envs"],
        "checkpoint_training_total_sampled_steps":extra["training_total_sampled_steps"],
        "checkpoint_training_smoke":extra["training_smoke"],"checkpoint_effective_hidden_dim":extra["effective_hidden_dim"],
        "checkpoint_environment_config_sha256":extra["environment_config_sha256"],
        "checkpoint_algorithm_config_sha256":extra["algorithm_config_sha256"],
        "evaluation_environment_config_sha256":config_sha256(environment_config),
        "provided_algorithm_config_sha256":config_sha256(algorithm_config),"protocol_complete":True,
        "checkpoint_environment_version":extra["environment_version"],
        "evaluation_environment_version":environment_config["environment_version"],
        "holdout_seed_base":seeds[0],"holdout_seed_end":seeds[-1],"evaluation_episodes":len(seeds),
        "device":str(device),"observation_dim":trainer.observation_dim,"action_dim":trainer.action_dim,"num_agents":trainer.num_agents})
    return result
