"""Preflight native network/Adam states before any live parameters are replaced."""
import math
from contextlib import contextmanager
from copy import deepcopy
import random
import numpy as np
import torch

def validate_model_state(model,saved):
    reference=model.state_dict()
    if not isinstance(saved,dict) or reference.keys()!=saved.keys():raise ValueError('Checkpoint network architecture keys mismatch')
    for key,expected in reference.items():
        actual=saved[key]
        if not isinstance(actual,torch.Tensor) or actual.shape!=expected.shape or actual.dtype!=expected.dtype:
            raise ValueError(f'Checkpoint network architecture/dtype mismatch: {key}')
        if not torch.isfinite(actual).all():raise ValueError(f'Nonfinite checkpoint model tensor: {key}')

def validate_optimizer_state(optimizer,saved):
    if not isinstance(saved,dict) or not isinstance(saved.get('state'),dict) or not isinstance(saved.get('param_groups'),list):
        raise ValueError('Invalid checkpoint optimizer structure')
    if len(saved['param_groups'])!=len(optimizer.param_groups):raise ValueError('Checkpoint optimizer group count mismatch')
    parameters={}
    for actual,expected in zip(saved['param_groups'],optimizer.param_groups):
        if not isinstance(actual,dict) or len(actual.get('params',[]))!=len(expected['params']):raise ValueError('Checkpoint optimizer parameter count mismatch')
        for key in ('lr','betas','eps','weight_decay','amsgrad'):
            if key in expected and actual.get(key)!=expected[key]:raise ValueError(f'Checkpoint optimizer hyperparameter mismatch: {key}')
        for index,parameter in zip(actual['params'],expected['params']):
            if type(index) is not int or index in parameters:raise ValueError('Invalid checkpoint optimizer parameter IDs')
            parameters[index]=parameter
    for index,state in saved['state'].items():
        if index not in parameters or not isinstance(state,dict):raise ValueError('Invalid checkpoint optimizer state mapping')
        parameter=parameters[index]
        if state and not {'step','exp_avg','exp_avg_sq'}<=state.keys():raise ValueError('Incomplete checkpoint Adam optimizer state')
        for key,value in state.items():
            if isinstance(value,torch.Tensor):
                if not torch.isfinite(value).all():raise ValueError('Nonfinite checkpoint optimizer state')
                if key=='step':
                    if value.ndim!=0 or value.item()<0:raise ValueError('Invalid checkpoint optimizer step')
                elif value.shape!=parameter.shape or value.dtype!=parameter.dtype:
                    raise ValueError('Checkpoint optimizer tensor shape/dtype mismatch')
            elif key!='step' or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('Invalid checkpoint optimizer state value')

def validate_counter(value,name):
    if type(value) is not int or value<0:raise ValueError(f'Invalid checkpoint counter: {name}')

def validate_torch_rng(state,*,device='cpu'):
    if not isinstance(state,torch.Tensor) or state.dtype!=torch.uint8 or state.ndim!=1:
        raise ValueError('Invalid checkpoint Torch RNG state')
    try:
        probe=torch.Generator(device=device)
        probe.set_state(state.cpu())
    except (RuntimeError,TypeError,ValueError) as error:
        raise ValueError(f'Invalid checkpoint Torch RNG state for {device}') from error

def validate_cuda_rng(states,device):
    if not isinstance(states,(list,tuple)):raise ValueError('Invalid checkpoint CUDA RNG states')
    for state in states:
        if not isinstance(state,torch.Tensor) or state.dtype!=torch.uint8 or state.ndim!=1 or state.numel()==0:
            raise ValueError('Invalid checkpoint CUDA RNG state')
    # CPU loading does not restore CUDA states and must not initialize CUDA just
    # to inspect another machine's device topology. Empty states are valid for
    # checkpoints saved on CPU-only machines, as in the original format.
    if torch.device(device).type=='cuda' and states:
        if len(states)!=torch.cuda.device_count():raise ValueError('Checkpoint CUDA RNG device count mismatch')
        for i,state in enumerate(states):validate_torch_rng(state,device=f'cuda:{i}')

def validate_numpy_rng(state):
    if not isinstance(state,(tuple,list)) or len(state)!=5:raise ValueError('Invalid checkpoint NumPy RNG state')
    kind,keys,position,has_gauss,cached=state
    if (kind!='MT19937' or not isinstance(keys,np.ndarray) or keys.shape!=(624,) or keys.dtype!=np.uint32
            or type(position) is not int or not 0<=position<=624
            or type(has_gauss) is not int or has_gauss not in (0,1)
            or not isinstance(cached,(int,float,np.floating)) or not math.isfinite(cached)):
        raise ValueError('Invalid checkpoint NumPy RNG state')
    try:np.random.RandomState(0).set_state(state)
    except (ValueError,TypeError,OverflowError) as error:raise ValueError('Invalid checkpoint NumPy RNG state') from error

def validate_python_rng(state):
    if not isinstance(state,tuple) or len(state)!=3:raise ValueError('Invalid checkpoint Python RNG state')
    cached=state[2]
    if cached is not None and (not isinstance(cached,(float,int)) or not math.isfinite(cached)):
        raise ValueError('Invalid checkpoint Python RNG Gaussian cache')
    try:random.Random(0).setstate(state)
    except (ValueError,TypeError,OverflowError) as error:raise ValueError('Invalid checkpoint Python RNG state') from error

def validate_generator_rng(generator,state):
    try:
        probe=type(generator.bit_generator)(0)
        probe.state=deepcopy(state)
    except (ValueError,TypeError,KeyError,OverflowError) as error:
        raise ValueError('Invalid checkpoint learner RNG state') from error

def _rng_snapshot():
    return (torch.random.get_rng_state().clone(),
        [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [],
        deepcopy(np.random.get_state()),random.getstate())

def _restore_rng(snapshot):
    cpu,cuda,numpy,python=snapshot
    # Use the underlying generators, not potentially failing public setters or
    # optimizer/module load hooks that caused the transaction to abort.
    torch.default_generator.set_state(cpu)
    for i,state in enumerate(cuda):torch.cuda.default_generators[i].set_state(state)
    np.random.mtrand._rand.set_state(numpy)
    random._inst.setstate(python)

@contextmanager
def loading_transaction(models,optimizers,*,owner,counters=(),generator_attr=None,inference=False):
    """Restore the caller on recoverable preflight/commit failure, without hooks.

    This is a native-loader transaction, not a training resume mechanism. It
    covers initialized CUDA devices; process termination or an unusable GPU
    context cannot be repaired by an in-process rollback.
    """
    rng=_rng_snapshot()
    networks=[(model,{k:v.detach().clone() for k,v in model.state_dict().items()},
        [(p,p.requires_grad,None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()],
        [(m,m.training) for m in model.modules()]) for model in models]
    optimizer_states=[]
    for optimizer in optimizers:
        optimizer_states.append((optimizer,optimizer.state,{p:deepcopy(s) for p,s in optimizer.state.items()},
            optimizer.param_groups,[(g,{k:list(v) if k=='params' else deepcopy(v) for k,v in g.items()}) for g in optimizer.param_groups],
            deepcopy(optimizer.defaults)))
    missing=object()
    attributes={name:deepcopy(getattr(owner,name,missing)) if hasattr(owner,name) else missing for name in counters}
    generator=getattr(owner,generator_attr) if generator_attr else None
    generator_state=deepcopy(generator.bit_generator.state) if generator is not None else None
    try:
        yield
    except BaseException:
        with torch.no_grad():
            for model,saved,parameters,modes in networks:
                for name,value in model.state_dict(keep_vars=True).items():value.copy_(saved[name])
                for parameter,requires_grad,gradient in parameters:
                    parameter.requires_grad_(requires_grad);parameter.grad=gradient
                for module,training in modes:module.training=training
        for optimizer,state,values,groups,group_values,defaults in optimizer_states:
            state.clear();state.update(values);optimizer.state=state
            groups[:]=[group for group,_ in group_values]
            for group,values in group_values:group.clear();group.update(values)
            optimizer.param_groups=groups;optimizer.defaults.clear();optimizer.defaults.update(defaults)
        for name,value in attributes.items():
            if value is missing:
                if hasattr(owner,name):object.__delattr__(owner,name)
            else:object.__setattr__(owner,name,value)
        if generator is not None:
            object.__setattr__(owner,generator_attr,generator);generator.bit_generator.state=generator_state
        _restore_rng(rng)
        raise
    else:
        if inference:_restore_rng(rng)
