"""Explicit short technical run: formal networks/hyperparameters, limited budget."""
from pathlib import Path
import argparse
import sys
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.navigation3d import load_config
from algorithm.navigation3d.runner import NavigationRunner
from algorithm.navigation3d.evaluation import write_json, evaluate, load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-config', default=str(ROOT/'configs/navigation3d_v1.yaml'))
    p.add_argument('--algorithm-config', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--sampled-steps', type=int, default=1024)
    p.add_argument('--num-envs', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=['cuda'], default='cuda')
    args = p.parse_args()
    if not 0 < args.sampled_steps <= 8192 or args.num_envs <= 0 or args.sampled_steps % args.num_envs:
        p.error('technical run budget must be <=8192, positive and divisible by num-envs')
    if args.seed < 0 or args.seed+args.sampled_steps+args.num_envs >= 10_000_000:
        p.error('technical training seeds must stay below the evaluation range')
    torch.set_num_threads(1)
    env_config = load_config(args.env_config)
    config = yaml.safe_load(Path(args.algorithm_config).read_text(encoding='utf-8'))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (('env_config', env_config), ('algorithm_config', config)):
        (output/(name+'.yaml')).write_text(yaml.safe_dump(value, sort_keys=False), encoding='utf-8')
    runner = NavigationRunner(env_config, config, args.num_envs, args.seed, args.device)
    try:
        trainer = runner.trainer
        initial = {k: v.clone() for k, v in trainer.actor.state_dict().items()}
        metrics = []
        while trainer.sampled_steps < args.sampled_steps:
            remaining = (args.sampled_steps-trainer.sampled_steps)//args.num_envs
            batch = runner.collect_rollout(min(runner.rollout_steps, remaining))
            row = trainer.update(batch)
            metrics.append(dict(sampled_steps=trainer.sampled_steps, **row))
            print(f'{trainer.kind} sampled_steps={trainer.sampled_steps} actor_updates={trainer.actor_update_count} KL={row["approx_kl"]:.6g}', flush=True)
        trainer.save(output/'final.pt', dict(runtime_num_envs=args.num_envs, runtime_budget=args.sampled_steps,
            used_training_seeds=sorted(runner.vector.used_training_seeds), technical_smoke=True))
        bank, mask = runner.observations.copy(), runner.vector.current_alive_masks.copy()
        expected = trainer.act(bank, mask, deterministic=True)
        reloaded = load_checkpoint(output/'final.pt')
        actual = reloaded.act(bank, mask, deterministic=True)
        if not np.array_equal(expected, actual):
            raise AssertionError('navigation checkpoint deterministic action mismatch')
        if not any(not torch.equal(initial[k], trainer.actor.state_dict()[k]) for k in initial):
            raise AssertionError('navigation actor did not update')
        evaluation = evaluate(env_config, reloaded, range(10_000_000, 10_000_005))
        summary = dict(algorithm=trainer.kind, environment_id='navigation3d-v1', device=str(trainer.device),
            gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, observation_dim=trainer.observation_dim,
            action_dim=3, num_agents=trainer.num_agents, actor_parameter_count=sum(p.numel() for p in trainer.actor.parameters()),
            critic_parameter_count=sum(p.numel() for p in trainer.critic.parameters()),
            sampled_steps=trainer.sampled_steps, ppo_updates=trainer.ppo_update_count,
            actor_updates=trainer.actor_update_count, critic_updates=trainer.critic_update_count,
            network_width=256, runtime_num_envs=args.num_envs, worker_pids=runner.vector.worker_pids,
            parallel_backend=runner.vector.backend, checkpoint_actions_equal=True,
            actor_changed=True, completed_training_episodes=len(runner.completed), evaluation=evaluation)
        write_json(output/'optimization_metrics.json', metrics)
        write_json(output/'summary.json', summary)
        print(f'done: {trainer.kind}, CUDA, {trainer.sampled_steps} samples, eval_success={evaluation["success_rate"]}', flush=True)
    finally:
        runner.close()


if __name__ == '__main__':
    main()
