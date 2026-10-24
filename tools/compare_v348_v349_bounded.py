"""Matched 6144-sample MAAC runs; explicit diagnostic budget, never 3M."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from env.config import load_config
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.geometry import engagement_geometry


def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')


def frozen(out):
    result=[]
    for distance in (500,1500,2000,3000,4000,5000):
        for heading in (0,15,30,60,90,180):
            for blue_heading in (0,180):
                envs=[MultiUAVCombatEnv(load_config(ROOT/f'configs/combat_environment_v{v}.yaml')) for v in ('348','349')]
                values=[]
                for e in envs:
                    e.reset(1);e.red=[AircraftState(0,0,-3000,225,0,np.deg2rad(heading))]+[AircraftState(0,0,-3000,225,0,0,False) for _ in range(3)]
                    e.blue=[AircraftState(distance,0,-3000,225,0,np.deg2rad(blue_heading))]+[AircraftState(0,0,-3000,225,0,0,False) for _ in range(3)]
                    reward,diagnostic=e.dense_combat_reward();values.append(float(reward[0]))
                g=engagement_geometry(envs[0].red[0],envs[0].blue[0])
                result.append(dict(distance=distance,red_heading=heading,blue_heading=blue_heading,alpha_deg=np.rad2deg(g.off_boresight),aa_deg=np.rad2deg(g.aa),v348_r4=values[0],v349_r4=values[1]))
    dump(out/'frozen_reward_grid.json',result)


def main():
    # Windows spawn imports this driver in every environment worker. Keep CUDA
    # DLL imports parent-only; no algorithm/worker sampling behavior changes.
    import torch
    from algorithm.maac.runner import MAACTrainingRunner
    from algorithm.maac.protocol import validate_checkpoint
    from tools.record_combat_episode import load_recording_policy,record
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory')
    if args.output.exists():raise FileExistsError(args.output)
    out=args.output;out.mkdir(parents=True);torch.set_num_threads(1);frozen(out)
    results={};initial=None
    for version in ('348','349'):
        d=out/version;d.mkdir();ec=load_config(ROOT/f'configs/combat_environment_v{version}.yaml')
        ac=yaml.safe_load((ROOT/f'configs/maac_4v4_v{version}.yaml').read_text())
        for n,c in [('env_config.yaml',ec),('algorithm_config.yaml',ac)]: (d/n).write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
        runner=MAACTrainingRunner(ec,ac,d,total_sampled_steps=6144,seed=1,smoke=False)
        runner.eval_seeds=[10000000,10000001,10000002,10000003,10000004]
        state={name:{k:v.detach().clone() for k,v in getattr(runner.trainer,name).state_dict().items()} for name in ('actor','critic')}
        if initial is None:initial=state
        else:
            for name in initial:
                for k,v in initial[name].items():assert torch.equal(v,state[name][k])
        dump(d/'diagnostic_protocol.json',dict(seed=1,budget=6144,num_envs=12,formal_hyperparameters=True,training_smoke=False,
             evaluation_seeds=runner.eval_seeds,initial_networks_equal=True,no_formal_3m_training=True))
        result=runner.run()
        t=runner.trainer
        assert t.sampled_steps==6144 and t.actor_updates>0 and t.critic_updates>0
        for module in (t.actor,t.critic):assert all(torch.isfinite(p).all() for p in module.parameters())
        path=d/'final.pt';trainer,saved,provenance=load_recording_policy(path,ec,ac,'cuda');validate_checkpoint(saved,ec,ac)
        record(trainer,ec,20000349,d/'recording',provenance)
        other='349' if version=='348' else '348'
        try:validate_checkpoint(saved,load_config(ROOT/f'configs/combat_environment_v{other}.yaml'),yaml.safe_load((ROOT/f'configs/maac_4v4_v{other}.yaml').read_text()))
        except (ValueError,RuntimeError):pass
        else:raise AssertionError('cross-version checkpoint accepted')
        changes={name:float(sum((v-state[name][k]).square().sum() for k,v in getattr(t,name).state_dict().items()).sqrt()) for name in state}
        assert all(x>0 for x in changes.values())
        result.update(parameter_change_l2=changes,cross_version_rejected=True,checkpoint_reload=True,formal_3m_training_started=False)
        dump(d/'bounded_summary.json',result);results[version]=result
        del runner,trainer,t;torch.cuda.empty_cache()
    dump(out/'comparison.json',results)


if __name__=='__main__':main()
