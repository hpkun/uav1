"""Strict noise-free MADDPG checkpoint evaluation on development seeds."""
import argparse
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from algorithm.maddpg.evaluation import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--env-config')
    parser.add_argument('--algorithm-config')
    parser.add_argument('--device', default='cuda', choices=('cuda',))
    parser.add_argument('--seed-base', type=int)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--output')
    args = parser.parse_args()
    def resolved(path):
        value = Path(path)
        return value if value.is_absolute() else ROOT/value
    checkpoint = resolved(args.checkpoint)
    env = yaml.safe_load(resolved(args.env_config or checkpoint.parent/'env_config.yaml').read_text(encoding='utf-8'))
    config = yaml.safe_load(resolved(args.algorithm_config or checkpoint.parent/'algorithm_config.yaml').read_text(encoding='utf-8'))
    base = config['implementation']['evaluation_seed_base'] if args.seed_base is None else args.seed_base
    episodes = config['training']['evaluation_episodes'] if args.episodes is None else args.episodes
    result = evaluate_checkpoint(checkpoint, env, config, args.device, range(base, base+episodes))
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        output = resolved(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
