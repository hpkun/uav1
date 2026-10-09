"""Read-only validation of the short MADDPG CUDA smoke and source preservation."""
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from algorithm.maddpg.protocol import require_cuda, validate_checkpoint
from algorithm.maddpg.factory import build_maddpg_trainer


def all_finite(value):
    if isinstance(value, dict):
        return all(all_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(all_finite(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def main():
    require_cuda('cuda')
    directory = ROOT/'outputs/maddpg_implementation_validation'
    smoke = directory/'smoke_8v8'
    summary = json.loads((smoke/'run_summary.json').read_text())
    reloaded = json.loads((smoke/'reloaded_evaluation.json').read_text())
    metrics = [json.loads(line) for line in (smoke/'optimization_metrics.jsonl').read_text().splitlines()]
    assert len(metrics) == summary['actor_update_count'] == summary['critic_update_count'] == 57
    assert summary['sampled_steps'] == summary['replay_size'] == 1024 and summary['vector_steps'] == 64
    assert summary['actor_updates_per_agent'] == summary['critic_updates_per_agent'] == [57]*8
    assert all(all_finite(row) for row in metrics)
    assert all(reloaded[key] == value for key, value in summary['latest_evaluation'].items() if key != 'sampled_steps')
    env = yaml.safe_load((smoke/'env_config.yaml').read_text())
    config = yaml.safe_load((smoke/'algorithm_config.yaml').read_text())
    state = torch.load(smoke/'final.pt', map_location='cpu', weights_only=False)
    validate_checkpoint(state, env, config)
    t = build_maddpg_trainer(config, 'cuda', state['training_seed'])
    t.load(smoke/'final.pt', env, config)
    for name in ('actors', 'critics', 'target_actors', 'target_critics'):
        for module, saved in zip(getattr(t, name), state[name]):
            assert all(torch.equal(value.cpu(), saved[key]) for key, value in module.state_dict().items())
    before = json.loads((directory/'preservation_before.json').read_text(encoding='utf-8-sig'))
    for windows_path, digest in before.items():
        posix = windows_path.replace('\\', '/')
        path = Path('/mnt/c/'+posix[3:]) if sys.platform != 'win32' else Path(windows_path)
        assert hashlib.sha256(path.read_bytes()).hexdigest().upper() == digest
    counts = {}
    for n in (5, 8):
        formal = yaml.safe_load((ROOT/f'configs/maddpg_{n}v{n}.yaml').read_text())
        counts[f'{n}v{n}'] = build_maddpg_trainer(formal, 'cuda').parameter_counts()
        assert formal['training']['total_sampled_steps'] == 3000000
        assert formal['training']['learning_starts'] == 10000 and formal['training']['replay_capacity'] == 1000000
    report = {'cuda': torch.cuda.get_device_name(0), 'sampled_steps': 1024, 'vector_steps': 64,
        'parallel_workers': len(set(summary['protocol']['worker_pids'])), 'gradient_steps': 57,
        'actor_updates_per_agent': [57]*8, 'critic_updates_per_agent': [57]*8,
        'all_losses_gradients_q_targets_finite': True, 'checkpoint_all_online_and_target_tensors_roundtrip': True,
        'two_episode_reloaded_evaluation_identical': True, 'formal_yaml_values_preserved': True,
        'existing_algorithm_environment_config_files_unchanged': len(before), 'parameter_counts': counts,
        'evaluation': summary['latest_evaluation']}
    (directory/'validation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'evaluation'}, indent=2))


if __name__ == '__main__':
    main()
