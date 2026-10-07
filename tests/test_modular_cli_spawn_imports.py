"""Environment spawn must not duplicate training/CUDA imports in every worker."""
from pathlib import Path
import subprocess,sys

ROOT=Path(__file__).resolve().parents[1]

def test_spawn_bootstrap_does_not_import_training_stack():
    code="""
import runpy,sys
runpy.run_path('algorithm/train_modular_mappo.py',run_name='__mp_main__')
assert 'torch' not in sys.modules
assert 'algorithm.modular_mappo.trainer' not in sys.modules
assert 'algorithm.modular_mappo.runner' not in sys.modules
from env.process_worker import combat_environment_worker
assert 'torch' not in sys.modules
print('LIGHTWEIGHT_SPAWN_PASS')
"""
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert 'LIGHTWEIGHT_SPAWN_PASS' in result.stdout

def test_normal_library_import_preserves_cli_api():
    from algorithm.train_modular_mappo import load_config,ModularMAPPOTrainingRunner
    config=load_config(ROOT/'configs/dev_marc_state_memory_v1_2m.yaml')
    assert config['training']['num_train_envs']==24
    assert config['training']['total_sampled_steps']==2000000
    assert callable(ModularMAPPOTrainingRunner)
