"""Compare fixed, non-holdout observations and optional canonical trajectories."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import yaml
import numpy as np
from algorithm.ippo.diagnostics import compare_checkpoints, trace_diagnostics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--observation-bank',required=True)
    p.add_argument('--env-config',default='configs/combat_environment_v341.yaml')
    p.add_argument('--algorithm-config',default='configs/ippo_4v4_v341.yaml')
    p.add_argument('--recordings',nargs='*',help='Optional canonical recording directories with the same scene seed')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    ec=yaml.safe_load(Path(args.env_config).read_text(encoding='utf-8'))
    ac=yaml.safe_load(Path(args.algorithm_config).read_text(encoding='utf-8'))
    result=compare_checkpoints(args.checkpoints,args.observation_bank,ec,ac)
    previous=None;scene_seed=None;scenes=[]
    from tools.combat_trace import read_trace
    for directory in args.recordings or []:
        directory=Path(directory)
        meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        if not 20_000_000 <= meta['episode_seed'] < 30_000_000:
            raise ValueError('scene comparisons require recording seeds')
        if scene_seed is not None and meta['episode_seed']!=scene_seed:
            raise ValueError('trajectory comparisons require equal scenario seeds')
        scene_seed=meta['episode_seed']
        trace=read_trace(directory/'episode_trace.npz',meta)
        states=trace['red_kinematics'];alive=trace['red_alive']
        delta=None;overlap=0
        if previous is not None:
            length=min(len(previous[0]),len(states))
            mask=previous[1][:length]&alive[:length]
            overlap=int(mask.sum())
            if overlap:delta=float(np.linalg.norm(states[:length,:,:3]-previous[0][:length,:,:3],axis=-1)[mask].mean())
        scenes.append(dict(recording=str(directory),episode_seed=scene_seed,
            previous_mean_position_delta_m=delta,overlap_alive_frames=overlap,
            flight_diagnostics=trace_diagnostics(trace,meta)))
        previous=(states,alive)
    result['fixed_scene_comparisons']=scenes
    output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(output.resolve())

if __name__=='__main__':main()
