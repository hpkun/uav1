"""Three bounded CUDA audits with unchanged formal algorithm hyperparameters.

MAPPO/IPPO: 4096 samples, one formal PPO update. MAAC: one full replay cycle.
No formal training run; includes historical checkpoint and retired-version audit.
"""
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import sys
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import verify_v342_v343_readiness as ppo
from tools.verify_maac_v344_v346 import formal_update
from env.config import load_config
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation as validate_mappo
from algorithm.ippo.protocol import validate_checkpoint as validate_ippo
from algorithm.maac.protocol import validate_checkpoint as validate_maac
from algorithm.common.evaluator import evaluate
from tools.record_combat_episode import load_recording_policy
from tools.combat_replay_report import export_analysis

VALIDATORS=dict(mappo=validate_mappo,ippo=validate_ippo,maac=validate_maac)


def configs(algo,v):
    return load_config(ROOT/f'configs/combat_environment_v{v}.yaml'),ppo.algorithm_config(algo,v)


def historical_checkpoints(out):
    specs=[('mappo','343','outputs/v342_v343_validation_20261021/cuda_smoke/mappo_v343/diagnostic_4096.pt'),
           ('ippo','343','outputs/v342_v343_validation_20261021/cuda_smoke/ippo_v343/diagnostic_4096.pt'),
           ('mappo','344','outputs/v344_v345_validation_20261022/cuda_smoke/mappo_v344/diagnostic_4096.pt'),
           ('ippo','344','outputs/v344_v345_validation_20261022/cuda_smoke/ippo_v344/diagnostic_4096.pt'),
           ('maac','344','outputs/maac_v344_v345_validation_20261022/cuda_verified/344_formal_update/diagnostic_bounded.pt')]
    results=[]
    for algo,v,rel in specs:
        path=ROOT/rel
        if not path.is_file():raise FileNotFoundError('historical checkpoint required: '+str(path))
        digest=hashlib.sha256(path.read_bytes()).hexdigest();ec,ac=configs(algo,v)
        trainer,state,_=load_recording_policy(path,ec,ac,'cuda')
        evaluation=evaluate(trainer,ec,range(10_000_000,10_000_002))
        try:VALIDATORS[algo](state,*configs(algo,'346'))
        except (ValueError,RuntimeError):pass
        else:raise AssertionError('historical checkpoint accepted as v3.46')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        results.append(dict(algorithm=algo,version=v,checkpoint=str(path),sha256=digest,
            read_only=True,loaded_evaluated_on_cuda=True,v346_rejected=True,evaluation=evaluation))
    # Read an actual retired checkpoint; never instantiate or relabel it.
    path=ROOT/'outputs/v344_v345_validation_20261022/cuda_smoke/mappo_v345/diagnostic_4096.pt'
    state=torch.load(path,map_location='cuda',weights_only=False)
    assert state['extra']['environment_version']=='3.45'
    for v in ('344','346'):
        try:validate_mappo(state,*configs('mappo',v))
        except RuntimeError:pass
        else:raise AssertionError('retired checkpoint silently relabeled')
    results.append(dict(retired_checkpoint=str(path),version='3.45',v344_and_v346_rejected=True))
    (out/'historical_checkpoints.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--history-only',action='store_true',help='Complete historical checks after all three bounded CUDA summaries exist')
    args=parser.parse_args();out=args.output
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    torch.set_num_threads(1)
    if args.history_only:
        results=[json.loads((out/f'{algo}_v346/summary.json').read_text()) for algo in ('mappo','ippo','maac')]
        trainer,_,_=load_recording_policy(out/'maac_v346/diagnostic_bounded.pt',*configs('maac','346'),'cuda')
        first=evaluate(trainer,configs('maac','346')[0],range(10_000_000,10_000_002))
        assert first==evaluate(trainer,configs('maac','346')[0],range(10_000_000,10_000_002))
        results[2]['deterministic_evaluation_repeat_exact']=True
        (out/'maac_v346/summary.json').write_text(json.dumps(results[2],indent=2,allow_nan=False),encoding='utf-8')
        historical=historical_checkpoints(out)
        (out/'summary.json').write_text(json.dumps(dict(cuda=results,historical=historical),indent=2,allow_nan=False),encoding='utf-8')
        print('Three bounded CUDA audits and historical checkpoint checks complete',flush=True)
        return
    if out.exists() and any(out.iterdir()):raise FileExistsError('fresh audit output required')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);results=[]
    for algo in ('mappo','ippo'):
        created=[];name='MAPPOTrainingRunner' if algo=='mappo' else 'IPPOTrainingRunner'
        original=getattr(ppo,name)
        def factory(*args,**kwargs):
            runner=original(*args,**kwargs);created.append(runner);return runner
        directory=out/f'{algo}_v346'
        with patch.object(ppo,name,factory):result=ppo.run(algo,'346',directory)
        runner=created[0]
        assert runner.vector._closed and all(not p.is_alive() for p in runner.vector._processes)
        state=torch.load(directory/'diagnostic_4096.pt',map_location='cuda',weights_only=False)
        try:VALIDATORS[algo](state,*configs(algo,'344'))
        except (ValueError,RuntimeError):pass
        else:raise AssertionError('new checkpoint accepted as v3.44')
        export_analysis(directory/'recording/episode_trace.npz',directory/'recording/metadata.json',directory/'recording',('csv','report'))
        trainer,_,_=load_recording_policy(directory/'diagnostic_4096.pt',*configs(algo,'346'),'cuda')
        first=evaluate(trainer,configs(algo,'346')[0],range(10_000_000,10_000_002))
        second=evaluate(trainer,configs(algo,'346')[0],range(10_000_000,10_000_002))
        assert first==second,'deterministic evaluation changed'
        result.update(workers_closed=True,all_workers_exited=True,v344_rejected=True,
                      deterministic_evaluation_repeat_exact=True,analysis_replay_exported=True)
        (directory/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        results.append(result)
    print('MAAC v346: formal bounded replay audit',flush=True)
    results.append(formal_update('346',out/'maac_v346'))
    historical=historical_checkpoints(out)
    (out/'summary.json').write_text(json.dumps(dict(cuda=results,historical=historical),indent=2,allow_nan=False),encoding='utf-8')
    print('All three formal-parameter bounded CUDA audits passed; no formal training',flush=True)


if __name__=='__main__':main()
