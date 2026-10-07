"""Read-only preflight for the frozen Jiao recipe matched pair."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_jiao2025_3m_config,checkpoint_architecture
ENV=ROOT/'configs/persistent_wave_v2_blue433_environment.yaml'
CONFIGS={variant:ROOT/f'configs/dev_jiao2025_{variant}_3m.yaml' for variant in ('matched_plain','core')}
OUTPUTS={variant:ROOT/f'outputs/dev433_jiao2025_{variant}_seed5301_3m' for variant in CONFIGS}
ENV_SHA='50d42e965443ab9896eb3d2de99c4216009daacb2d54fac05633a4b5b18b8985'
# New protocol's current frozen source identity, NOT a replacement historical lock.
ENV_SOURCE_SHA='721baf2a3180068593a3f9ccbbd2f3c619758f1b532e8ebd96976f28253b946d'

def env_source_sha():
    digest=hashlib.sha256()
    for path in sorted((ROOT/'env').rglob('*.py')):
        digest.update(path.relative_to(ROOT).as_posix().encode());digest.update(b'\0')
        digest.update(path.read_bytes());digest.update(b'\0')
    return digest.hexdigest()

def validate_pair(plain,core,env):
    for cfg in (plain,core):
        validate_jiao2025_3m_config(cfg,env)
        if cfg['training']['seed']!=5301:raise ValueError('formal development seed must be5301')
    a,b=deepcopy(plain),deepcopy(core)
    b['development_method']=a['development_method']
    for key in ('wave_context','recurrent_memory','popart'):b['modules'][key]=deepcopy(a['modules'][key])
    if a!=b:raise ValueError('pair differs outside scalar-F/joint-GRU/PopArt package')
    return True

def preflight():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    if hashlib.sha256(ENV.read_bytes()).hexdigest()!=ENV_SHA:raise RuntimeError('frozen environment SHA mismatch')
    if env_source_sha()!=ENV_SOURCE_SHA:raise RuntimeError('current frozen environment source tree mismatch')
    for output in OUTPUTS.values():
        if output.exists():raise RuntimeError(f'refuse occupied formal output: {output}')
    configs={k:load_config(v) for k,v in CONFIGS.items()}
    validate_pair(configs['matched_plain'],configs['core'],load_config(ENV));rows=[]
    for variant,cfg in configs.items():
        trainer=build_modular_mappo_trainer(cfg,'cuda');arch=checkpoint_architecture(trainer)
        core=variant=='core'
        assert arch['actor_input_dim']==(53 if core else 52)
        assert arch['actor_context_dim']==arch['critic_context_dim']==int(core)
        assert trainer.actor.recurrent_hidden_dim==trainer.critic.recurrent_hidden_dim==(128 if core else 0)
        enabled=trainer.module_protocol()['enabled_modules']
        assert set(enabled)==({'wave_context','recurrent_memory','popart'} if core else set())
        for step in (0,600000,900000,1500000,3000000):
            assert trainer.actor_lr_decay.apply(trainer.actor_optimizer,step,.0005)==.0005
            assert trainer.actor_optimizer.param_groups[0]['lr']==.0005
        rows.append(dict(variant=variant,actor_input=arch['actor_input_dim'],critic_context=arch['critic_context_dim'],
            actor_GRU=trainer.actor.recurrent_hidden_dim,critic_GRU=trainer.critic.recurrent_hidden_dim,
            F='1/2/3 actor+critic' if core else 'OFF',PopArt=trainer.popart.enabled,enabled_modules=enabled,
            sequence=32 if core else None,shared_training=cfg['training'],
            evaluation='44000000..44000049 deterministic50/100k',future_holdout_executed=False))
    return dict(status='JIAO2025_CORE_3M_PREFLIGHT_PASS',cells=rows,formal_outputs_present=0,
        same_training_seed=True,same_environment_initialization_protocol=True,
        full_parameter_matched_initialization_claimed=False,
        environment_sha256=ENV_SHA,environment_source_sha256=ENV_SOURCE_SHA,formal_training_started=False)

if __name__=='__main__':
    report=preflight()
    print('variant | Actor input | F | Actor/Critic GRU | PopArt | modules | BPTT | validation | holdout executed')
    for row in report['cells']:print(f"{row['variant']} | {row['actor_input']} | {row['F']} | {row['actor_GRU']}/{row['critic_GRU']} | {row['PopArt']} | {row['enabled_modules']} | {row['sequence']} | {row['evaluation']} | False")
    print(json.dumps(report,indent=2))
