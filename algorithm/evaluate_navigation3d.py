"""Independent deterministic navigation checkpoint evaluator."""
from pathlib import Path
import argparse
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.navigation3d.evaluation import load_checkpoint, evaluate, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--seed-base', type=int, default=10_000_000)
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.episodes <= 0 or args.seed_base < 10_000_000:
        p.error('positive episode count and separate evaluation seeds required')
    torch.set_num_threads(1)
    trainer = load_checkpoint(args.checkpoint)
    result = evaluate(trainer.env_config, trainer, range(args.seed_base, args.seed_base+args.episodes))
    write_json(args.output, result)
    print(f'episodes={result["episodes"]}, success_rate={result["success_rate"]}')


if __name__ == '__main__':
    main()
