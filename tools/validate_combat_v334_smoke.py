"""Validate saved minimal CUDA smokes, reload, evaluate and reject identity drift."""
import json
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.config import load_config
from env.combat_env import MultiUAVCombatEnv
from tools.record_combat_episode import load_recording_policy
from algorithm.common.evaluator import evaluate
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation
from algorithm.madsac.protocol import validate_madsac_checkpoint
from algorithm.maac.protocol import validate_checkpoint


def finite(value):
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite(v)
    elif isinstance(value,float):assert np.isfinite(value)


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    ec=load_config(ROOT/'configs/combat_environment_v334.yaml');out=ROOT/'outputs/v334_validation'
    checks={'mappo':validate_checkpoint_for_evaluation,'madsac':validate_madsac_checkpoint,'maac':validate_checkpoint}
    results={}
    for name,check in checks.items():
        ac=yaml.safe_load((ROOT/f'configs/{name}_4v4_v334.yaml').read_text())
        folder=out/f'{name}_smoke';summary=json.loads((folder/'run_summary.json').read_text())
        assert summary['sampled_steps']==32
        actor=summary.get('actor_updates',summary.get('actor_update_count',0))
        critic=summary.get('critic_updates',summary.get('critic_update_count',0))
        assert actor>0 and critic>0
        if name!='mappo':assert summary['replay_size']==32
        trainer,state,provenance=load_recording_policy(folder/'latest.pt',ec,ac,'cuda')
        finite(state);finite(summary)
        assert state['sampled_steps']==32
        e=MultiUAVCombatEnv(ec);obs,_=e.reset(41)
        actions=trainer.act(obs[None],e.red_alive_mask[None],deterministic=True)
        assert actions.shape==(1,4,3) and np.isfinite(actions).all()
        e.step(actions[0])
        evaluation=evaluate(trainer,ec,[10000002])
        finite(evaluation)
        rejected=[]
        for old in ('330','331','332','333'):
            prior=torch.load(ROOT/f'outputs/v{old}_validation/{name}_smoke/latest.pt',map_location='cuda',weights_only=False)
            try:check(prior,ec,ac)
            except RuntimeError as error:
                assert 'environment_version' in str(error);rejected.append(old)
            else:raise AssertionError('historical checkpoint accepted')
        for field in ('environment_config_sha256','algorithm_config_sha256'):
            altered=deepcopy(state);altered['extra'][field]='invalid'
            try:check(altered,ec,ac)
            except RuntimeError:pass
            else:raise AssertionError('config hash drift accepted')
        results[name]=dict(checkpoint=provenance,actor_updates=actor,critic_updates=critic,
            replay_size=summary.get('replay_size'),reset_rollout_update_save_load=True,
            finite_checkpoint_and_metrics=True,evaluation=evaluation,
            rejected_historical_versions=rejected,config_hash_drift_rejected=True)
        print(f'{name}: CUDA checkpoint reload/evaluation passed',flush=True)
    (out/'smoke_validation.json').write_text(json.dumps(results,indent=2,allow_nan=False))


if __name__=='__main__':main()
