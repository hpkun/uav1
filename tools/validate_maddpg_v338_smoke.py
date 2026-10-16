"""Audit an existing MADDPG-v3 reference (g=1) CUDA smoke; never starts training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from algorithm.maddpg.protocol import require_cuda,validate_checkpoint
from algorithm.maddpg.factory import build_maddpg_trainer
from algorithm.maddpg.evaluation import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-dir',default='outputs/maddpg_v338_validation')
    args=parser.parse_args();directory=ROOT/args.validation_dir;smoke=directory/'smoke'
    require_cuda('cuda');torch.set_num_threads(1)
    summary=json.loads((smoke/'run_summary.json').read_text())
    ec,ac=(yaml.safe_load((smoke/name).read_text()) for name in ('env_config.yaml','algorithm_config.yaml'))
    state=torch.load(smoke/'latest.pt',map_location='cuda',weights_only=False)
    extra=validate_checkpoint(state,ec,ac)
    trainer=build_maddpg_trainer(ac,'cuda',extra['training_seed'])
    initial=[p.detach().clone() for p in trainer.target_actors.parameters()]
    trainer.load(smoke/'latest.pt',ec,ac)
    assert all(p.is_cuda for modules in (trainer.actors,trainer.critics,trainer.target_actors,trainer.target_critics) for p in modules.parameters())
    assert any(not torch.equal(p,q) for p,q in zip(initial,trainer.target_actors.parameters()))
    for name in ('actors','critics','target_actors','target_critics'):
        for module,saved in zip(getattr(trainer,name),state[name]):
            assert all(torch.equal(v,saved[k]) for k,v in module.state_dict().items())
    events,remainder=divmod(summary['sampled_steps']-ac['training']['learning_starts'],ac['training']['sampled_steps_per_update'])
    assert summary['actor_update_count']==summary['critic_update_count']==events==80
    assert summary['actor_updates_per_agent']==summary['critic_updates_per_agent']==[events]*4
    assert summary['transitions_since_update']==remainder==64
    assert summary['vector_steps']==512 and summary['sampled_steps']==8192
    metrics=[json.loads(x) for x in (smoke/'optimization_metrics.jsonl').read_text().splitlines()]
    training=[json.loads(x) for x in (smoke/'training_metrics.jsonl').read_text().splitlines()]
    assert len(metrics)==events and len(training)==512
    assert [row['actor_update_count'] for row in metrics]==list(range(1,events+1))
    assert all(row['actor_update_count']==0 for row in training if row['sampled_steps']<=128)
    assert sum(row['completed_episodes_this_step'] for row in training)>0
    for row in metrics:
        for k in ('actor_losses','critic_losses','actor_grad_norms','critic_grad_norms','q_means'):
            assert len(row[k])==4 and np.isfinite(row[k]).all()
        assert np.isfinite([row[k] for k in ('actor_loss','critic_loss','target_mean','target_abs_max')]).all()
    reloaded=evaluate(trainer,ec,range(10000000,10000002))
    assert all(reloaded[k]==v for k,v in summary['latest_evaluation'].items() if k in reloaded)
    (smoke/'reloaded_evaluation.json').write_text(json.dumps(reloaded,indent=2))
    preservation=json.loads((directory/'preservation_before.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in preservation.items())
    formal=yaml.safe_load((ROOT/'configs/maddpg_4v4_v338.yaml').read_text())
    assert formal['training']['learning_starts']==10000 and formal['training']['minibatch_size']==1024
    report={'status':'PASS','device':'cuda','gpu':torch.cuda.get_device_name(),
        'sampled_steps':8192,'vector_steps':512,'theoretical_update_events':events,
        'actual_actor_update_events':trainer.actor_update_count,'actual_critic_update_events':trainer.critic_update_count,
        'actor_updates_per_agent':trainer.actor_updates_per_agent,'critic_updates_per_agent':trainer.critic_updates_per_agent,
        'remainder':remainder,'optimization_records':len(metrics),'completed_training_episodes':sum(r['completed_episodes_this_step'] for r in training),
        'all_networks_cuda':True,'all_metrics_finite':True,'online_target_state_roundtrip':'exact',
        'reloaded_deterministic_evaluation':'identical','preserved_files':len(preservation),
        'parameter_counts':trainer.parameter_counts(),'evaluation':reloaded}
    (directory/'smoke_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='evaluation'},indent=2))


if __name__=='__main__':main()
