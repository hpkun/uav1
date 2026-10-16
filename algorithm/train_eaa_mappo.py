"""Train fixed-v3.38 EA/EAA structural ablations with the unchanged MAPPO PPO."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.eaa_mappo.entry import run_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-config',default='configs/combat_environment_v338.yaml')
    parser.add_argument('--algorithm-config',default='configs/eaa_mappo_4v4_v338.yaml')
    parser.add_argument('--device',choices=['cuda','cpu'])
    parser.add_argument('--seed',type=int)
    parser.add_argument('--num-envs',type=int)
    parser.add_argument('--total-sampled-steps',type=int)
    parser.add_argument('--output-dir')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume')
    run_variant(parser.parse_args())


if __name__=='__main__':main()
