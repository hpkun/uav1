"""CUDA stateless deterministic holdout evaluation entry."""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))
import argparse
import json
import yaml
from algorithm.ea_mappo.evaluation import evaluate_ea_mappo_checkpoint


def resolved(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT/path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--env-config",default="configs/combat_environment_v25.yaml")
    parser.add_argument("--algorithm-config",default="configs/ea_mappo_5v5.yaml")
    parser.add_argument("--seed-base",type=int,required=True)
    parser.add_argument("--episodes",type=int,required=True)
    parser.add_argument("--device",choices=["cuda"],default="cuda")
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    env,config = (yaml.safe_load(resolved(value).read_text(encoding="utf-8"))
                  for value in (args.env_config,args.algorithm_config))
    result = evaluate_ea_mappo_checkpoint(resolved(args.checkpoint),config,env,args.device,
        range(args.seed_base,args.seed_base+args.episodes))
    output = resolved(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
