"""Verify pre-change complete historical trajectories/rewards/RNG bit for bit."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.audit_v342_v343_history import PROBE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    probe=PROBE.replace("'336','341'", "'336','341','342','343','344'")
    probe=probe.replace('digest.update(bytes((term,trunc)))', '''
                if hasattr(e,'dwell'):
                    for side in ('red','blue'):
                        for name in ('targets','counts','sampled_targets','sampled_counts','completed_this_step'):
                            digest.update(getattr(e.dwell[side],name).tobytes())
                digest.update(bytes((term,trunc)))'''.strip())
    current=json.loads(subprocess.check_output([sys.executable,'-c',probe],cwd=ROOT,text=True))
    before=json.loads(subprocess.check_output([sys.executable,'-c',probe],cwd=args.baseline_root.resolve(),text=True))
    assert current==before,'historical trajectory/reward/RNG changed'
    result=dict(episodes=len(current),complete_trajectory_observation_reward_rng_dwell_exact=True,results=current)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'{len(current)} complete historical episodes match pre-change snapshot exactly',flush=True)


if __name__=='__main__':main()
