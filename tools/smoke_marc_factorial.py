"""Four tiny CUDA rollouts and real ten-epoch PPO updates; no evaluation."""
import json,tempfile
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.preflight_marc_factorial import config,ENV,load_config,MARC_FACTORIAL_CELLS,captured_update,state_memory_rollout
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner

def smoke():
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory')
    reports=[]
    for index,cell in enumerate(MARC_FACTORIAL_CELLS):
        with tempfile.TemporaryDirectory(prefix='marc_factorial_') as directory:
            runner=ModularMAPPOTrainingRunner(load_config(ENV),config(cell),num_envs=2,device='cuda',
                seed=88723001+index,output_dir=directory,smoke=True)
            try:
                for line in runner.startup_console_lines():print(line)
                real=runner.collect_rollout(4);metrics,_=captured_update(runner.trainer,real)
                mixed,_=captured_update(runner.trainer,state_memory_rollout(runner.trainer))
                assert not runner.trainer.milestone_aware_retention_credit.retention_active
                reports.append(dict(variant=cell,environment_steps=8,real_PPO_epochs=metrics['ppo_epochs_executed'],
                    mixed_wave_PPO_epochs=mixed['ppo_epochs_executed'],finite=True,retention_active=False))
            finally:runner.vector.close()
    return dict(status='MARC_FACTORIAL_TINY_CUDA_SMOKE_PASS',cells=reports,formal_training_started=False,
                evaluation_started=False,future_holdout_executed=False)
if __name__=='__main__':print(json.dumps(smoke(),indent=2))
