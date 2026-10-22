"""Four CUDA 4096-sample/full-hyperparameter integrations; no formal run loop."""
from pathlib import Path
from unittest.mock import patch
import argparse,json,sys
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import verify_v342_v343_readiness as baseline
from env.config import load_config
from algorithm.common.checkpoint import validate_checkpoint_for_resume,validate_checkpoint_for_evaluation
from algorithm.ippo.protocol import validate_checkpoint as validate_ippo
from tools.combat_replay_report import export_analysis


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('fresh smoke output required')
    args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);results=[]
    for version in ('344','345'):
        for algo in ('mappo','ippo'):
            created=[]
            original=baseline.MAPPOTrainingRunner if algo=='mappo' else baseline.IPPOTrainingRunner
            def factory(*args,**kwargs):
                runner=original(*args,**kwargs);created.append(runner);return runner
            name='MAPPOTrainingRunner' if algo=='mappo' else 'IPPOTrainingRunner'
            out=args.output/f'{algo}_v{version}'
            with patch.object(baseline,name,factory):result=baseline.run(algo,version,out)
            runner=created[0]
            assert runner.vector._closed and all(not process.is_alive() for process in runner.vector._processes)
            state=torch.load(out/'diagnostic_4096.pt',map_location='cuda',weights_only=False)
            other='345' if version=='344' else '344';ec=load_config(ROOT/f'configs/combat_environment_v{other}.yaml')
            ac=baseline.algorithm_config(algo,other)
            for check in ((validate_checkpoint_for_resume,validate_checkpoint_for_evaluation) if algo=='mappo' else (validate_ippo,)):
                try:check(state,ec,ac)
                except (ValueError,RuntimeError):pass
                else:raise AssertionError('cross-angle-version checkpoint silently accepted')
            metadata=json.loads((out/'recording/metadata.json').read_text(encoding='utf-8'))
            evaluation=json.loads((out/'evaluation.json').read_text(encoding='utf-8'))
            assert evaluation['event_semantics']==metadata['event_semantics']=='continuous_dwell_neutralization'
            assert not any(e['type']=='fire_attempt' for e in metadata['events'])
            for side in ('red','blue'):
                assert metadata[side+'_dwell_neutralizations']==metadata[side+'_attack_kills']
                assert metadata[side+'_dwell_completed']==sum(e['type']=='dwell_complete' and e['side']==side for e in metadata['events'])
                assert metadata[side+'_fire_attempts']==metadata[side+'_weapon_hits']==0
            export_analysis(out/'recording/episode_trace.npz',out/'recording/metadata.json',out/'recording',('csv','report'))
            result.update(workers_closed=True,event_semantics=metadata['event_semantics'],
                red_dwell_completed=metadata['red_dwell_completed'],blue_dwell_completed=metadata['blue_dwell_completed'],
                red_neutralizations=metadata['red_dwell_neutralizations'],blue_neutralizations=metadata['blue_dwell_neutralizations'],
                other_dwell_version_rejected=other,diagnostic_replay_and_event_counts_verified=True)
            (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');results.append(result)
    (args.output/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
