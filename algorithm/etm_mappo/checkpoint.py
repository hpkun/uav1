"""Fail-closed standalone algorithm identity and deterministic evaluation reload."""
from pathlib import Path
import torch
from .config import IMPLEMENTATION_VERSION, fingerprint, require_cuda, validate_config


def checkpoint_state(trainer,run_state=None):
    return dict(algorithm='ETM-MAPPO',implementation_version=IMPLEMENTATION_VERSION,
        environment_version=trainer.config['environment']['version'],
        config=trainer.config,config_sha256=fingerprint(trainer.config),
        environment_config_sha256=fingerprint(trainer.config['environment']),
        network_architecture=trainer.config['network'],parameter_counts=trainer.parameter_counts(),
        training_seed=trainer.config['training']['seed'],
        actor=trainer.actor.state_dict(),critic=trainer.critic.state_dict(),
        actor_optimizer=trainer.actor_optimizer.state_dict(),critic_optimizer=trainer.critic_optimizer.state_dict(),
        sampled_steps=trainer.sampled_steps,vector_steps=trainer.vector_steps,rollout_updates=trainer.rollout_updates,
        actor_optimizer_steps=trainer.actor_optimizer_steps,critic_optimizer_steps=trainer.critic_optimizer_steps,
        run_state=run_state or {},resume_capability='deterministic_evaluation_reload',bit_exact_training_resume=False)


def save_checkpoint(trainer,path,run_state=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(checkpoint_state(trainer,run_state),path)


def validate_checkpoint(state,config):
    validate_config(config)
    if state.get('algorithm')!='ETM-MAPPO' or state.get('implementation_version')!=IMPLEMENTATION_VERSION:
        raise RuntimeError('incompatible ETM-MAPPO checkpoint identity/version')
    for key,expected in [('environment_version',config['environment']['version']),
        ('config_sha256',fingerprint(config)),('environment_config_sha256',fingerprint(config['environment'])),
        ('config',config),('network_architecture',config['network']),('training_seed',config['training']['seed']),
        ('bit_exact_training_resume',False),('resume_capability','deterministic_evaluation_reload')]:
        if state.get(key)!=expected:raise RuntimeError(f'checkpoint {key} mismatch')
    counts={name:sum(value.numel() for value in state.get(name,{}).values()) for name in ('actor','critic')}
    counts['total']=sum(counts.values())
    if state.get('parameter_counts')!=counts:raise RuntimeError('checkpoint parameter count mismatch')
    for key in ('actor_optimizer','critic_optimizer','sampled_steps','vector_steps','rollout_updates',
                'actor_optimizer_steps','critic_optimizer_steps','run_state'):
        if key not in state:raise RuntimeError(f'checkpoint missing {key}')


def load_checkpoint(path,device='cuda',config=None):
    from .trainer import ETMMAPPOTrainer
    require_cuda(device)
    state=torch.load(path,map_location=device,weights_only=False)
    config=state.get('config') if config is None else config
    if not isinstance(config,dict):raise RuntimeError('checkpoint has no complete config')
    validate_checkpoint(state,config)
    trainer=ETMMAPPOTrainer(config,device)
    if trainer.parameter_counts()!=state['parameter_counts']:raise RuntimeError('actual network parameter count mismatch')
    trainer.actor.load_state_dict(state['actor'],strict=True)
    trainer.critic.load_state_dict(state['critic'],strict=True)
    trainer.actor_optimizer.load_state_dict(state['actor_optimizer'])
    trainer.critic_optimizer.load_state_dict(state['critic_optimizer'])
    for key in ('sampled_steps','vector_steps','rollout_updates','actor_optimizer_steps','critic_optimizer_steps'):
        setattr(trainer,key,int(state[key]))
    return trainer,state

