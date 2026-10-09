"""Independent classic MADDPG training entry point; --smoke never changes YAML."""
import argparse
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from algorithm.maddpg.runner import MADDPGTrainingRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-config', default='configs/combat_environment_v24.yaml')
    parser.add_argument('--algorithm-config', default='configs/maddpg_8v8.yaml')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda', choices=('cuda',))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--total-sampled-steps', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    def resolved(path):
        value = Path(path)
        return value if value.is_absolute() else ROOT/value
    env = yaml.safe_load(resolved(args.env_config).read_text(encoding='utf-8'))
    config = yaml.safe_load(resolved(args.algorithm_config).read_text(encoding='utf-8'))
    runner = MADDPGTrainingRunner(env, config, args.num_envs, args.total_sampled_steps,
        args.device, args.seed, resolved(args.output_dir), args.smoke)
    runner.run()


if __name__ == '__main__':
    main()
