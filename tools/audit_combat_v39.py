"""100 initialization seeds and bounded fixed-policy diagnostics; no learning."""
import json
from pathlib import Path
import numpy as np
import torch
from env.combat_env import MultiUAVCombatEnv
from env.sensor import sensor_geometry
from env.fixed_policy import NearestTargetPursuitPolicy
from env.config import load_config
from tools.record_combat_episode import load_recording_policy

SEEDS = (1, 31, 10000000)


def check_decomposition(reward, info, accumulated=None):
    paper = sum((info[f'r{i}_rewards'] for i in range(1, 5)), np.zeros(5))
    np.testing.assert_allclose(info['paper_rewards'], paper, rtol=0, atol=1e-6)
    np.testing.assert_allclose(info['extended_rewards'], paper + info['safety_rewards'], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(reward, info['extended_rewards'])
    np.testing.assert_array_equal(reward, info['local_rewards'])
    np.testing.assert_allclose(info['episode_paper_reward_total'],
        sum(info[f'episode_r{i}_total'] for i in range(1, 5)), rtol=0, atol=1e-6)
    np.testing.assert_allclose(info['episode_extended_reward_total'],
        info['episode_paper_reward_total'] + info['episode_safety_total'], rtol=0, atol=1e-6)
    if accumulated is not None:
        np.testing.assert_allclose(accumulated, info['episode_extended_reward_rewards'], rtol=0, atol=1e-6)


def initial_visibility():
    rows = []
    for seed in range(100):
        e = MultiUAVCombatEnv('configs/combat_environment_v39.yaml')
        e.reset(seed)
        r = sensor_geometry(e.red, e.blue, e.config['sensor'])[0]
        b = sensor_geometry(e.blue, e.red, e.config['sensor'])[0]
        rows.append(dict(seed=seed, red_visible_pairs=int(r.sum()), blue_visible_pairs=int(b.sum()),
            red_visible_blue_counts=r.sum(axis=1).tolist(), blue_visible_red_counts=b.sum(axis=1).tolist()))
    result = dict(seeds=list(range(100)), rows=rows)
    for side in ('red', 'blue'):
        pairs = [row[f'{side}_visible_pairs'] for row in rows]
        counts = np.asarray([row[f'{side}_visible_{"blue" if side == "red" else "red"}_counts'] for row in rows])
        result[side] = dict(total_pairs=sum(pairs), minimum_pairs=min(pairs), mean_pairs=float(np.mean(pairs)),
            maximum_pairs=max(pairs), episodes_all_25_visible=pairs.count(25),
            per_agent_minimum=counts.min(axis=0).tolist(), per_agent_mean=counts.mean(axis=0).tolist(),
            per_agent_maximum=counts.max(axis=0).tolist())
    return result


def trial(version, policy, seed):
    e = MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml')
    e.reset(seed)
    row = dict(version=e.environment_version, policy=policy, seed=seed,
        first_detection_step=None, first_r3_step=None, first_r4_step=None)
    pursuit = NearestTargetPursuitPolicy(e.config['blue_policy'], e.config['action'])
    def detected():
        return any(sensor_geometry(a, b, e.config['sensor'])[0].any()
            for a, b in ((e.red, e.blue), (e.blue, e.red)))
    if detected():
        row['first_detection_step'] = 0
    accumulated = np.zeros(5)
    while True:
        action = np.zeros((5, 3)) if policy == 'ZERO' else pursuit.team_actions(e.red, e.blue)
        _, reward, term, trunc, info = e.step(action)
        accumulated += reward
        if row['first_detection_step'] is None and detected():
            row['first_detection_step'] = e.steps
        for name in ('r3', 'r4'):
            if row[f'first_{name}_step'] is None and np.any(info[f'{name}_rewards'] != 0):
                row[f'first_{name}_step'] = e.steps
        if version == '39':
            check_decomposition(reward, info, accumulated)
        if term or trunc:
            break
    for name in ('fire_window', 'attempt', 'kill'):
        candidates = [info[f'{side}_first_{name}_step'] for side in ('red', 'blue')
            if info[f'{side}_first_{name}_step'] is not None]
        row[f'first_{name}_step'] = min(candidates) if candidates else None
    row.update(episode_length=e.steps, termination_reason=info['termination_reason'],
        red_boundary_exits=info['red_boundary_exits'], red_ground_losses=info['red_ground_losses'],
        red_combat_losses=info['blue_attack_kills'], red_survivors=info['red_survivors'],
        blue_survivors=info['blue_survivors'])
    for name in ('r1', 'r2', 'r3', 'r4', 'safety'):
        row[f'episode_{name}_total'] = info[f'episode_{name}_total']
    if version == '39':
        for name in ('paper_reward', 'extended_reward'):
            row[f'episode_{name}_total'] = info[f'episode_{name}_total']
    return row


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for checkpoint contract audit')
    out = Path('outputs/v39_validation')
    out.mkdir(parents=True, exist_ok=True)
    visibility = initial_visibility()
    (out / 'initial_visibility.json').write_text(json.dumps(visibility, indent=2))
    checkpoint = Path('outputs/mappo_v38_seed1_3m/checkpoint_3000000.pt')
    status = dict(checkpoint=str(checkpoint), cuda_device=torch.cuda.get_device_name(), evaluated_in_v39=False)
    if checkpoint.exists():
        algorithm = load_config_algorithm(checkpoint.parent / 'algorithm_config.yaml')
        try:
            load_recording_policy(checkpoint, load_config('configs/combat_environment_v39.yaml'), algorithm, 'cuda')
        except RuntimeError as error:
            status.update(contract_rejected=True, reason=str(error))
        else:
            raise AssertionError('v3.8 checkpoint must not pass v3.9 environment contract')
    else:
        status['reason'] = 'checkpoint absent locally'
    (out / 'old_checkpoint_contract.json').write_text(json.dumps(status, indent=2))
    rows = [trial(v, p, s) for p in ('ZERO', 'PURSUIT') for s in SEEDS for v in ('38', '39')]
    (out / 'fixed_policy_comparison.json').write_text(json.dumps(dict(rows=rows, seeds=SEEDS,
        first_detection_includes_reset=True, first_reward_is_actual_step_not_reset=True,
        reward_decomposition_atol=1e-6, training_started=False), indent=2))
    print(json.dumps(dict(visibility={s: visibility[s] for s in ('red', 'blue')}, checkpoint=status, rows=rows), indent=2))


def load_config_algorithm(path):
    import yaml
    return yaml.safe_load(path.read_text())


if __name__ == '__main__':
    main()
