"""Run bounded, matched three-seed navigation ablations."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import argparse
import yaml
from algorithm.etm_mappo.benchmark import run_benchmark

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/etm_mappo.yaml'))
    p.add_argument('--output-dir',required=True)
    p.add_argument('--budget',type=int,default=8192)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--reuse-completed-from',help='Optional directory of completed, strictly matching runs; partial runs are never resumed')
    a=p.parse_args()
    run_benchmark(yaml.safe_load(Path(a.config).read_text(encoding='utf-8')),a.output_dir,a.seeds,a.budget,a.reuse_completed_from)
