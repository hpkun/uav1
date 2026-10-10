"""Capture unchanged v3.6 before introducing v3.7 extension hooks."""
import hashlib,json
from pathlib import Path
import yaml
from tools._prepare_v36 import capture35

def capture36():
    # Reuse the capture harness without modifying historical files.
    import tools._prepare_v36 as harness
    original=harness.MultiUAVCombatEnv
    harness.MultiUAVCombatEnv=lambda _:original('configs/combat_environment_v36.yaml')
    try:return capture35()
    finally:harness.MultiUAVCombatEnv=original

def main():
    out=Path('outputs/v37_validation');out.mkdir(parents=True,exist_ok=True)
    (out/'before_hashes.json').write_text(json.dumps({p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('env','algorithm','configs') for p in Path(folder).rglob('*') if p.suffix in ('.py','.yaml')},indent=2))
    Path('tests/v37_v36_reference.json').write_text(json.dumps(capture36(),indent=2))
    c=yaml.safe_load(Path('configs/combat_environment_v36.yaml').read_text());c['environment_version']='3.7'
    c['weapon'].pop('hit_probability')
    c['weapon'].update(effective_hit_distance=1116.2212531024945,ata_noise_scale=.9984711359155588,ha_noise_scale=.9984711359155588)
    for k in ('team_casualty_penalty','win_reward','lose_penalty','draw_reward'):c['reward'][k]=0.
    Path('configs/combat_environment_v37.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    for name in ('mappo','rmappo','ea_mappo','stea_mappo'):
        Path(f'configs/{name}_5v5_v37.yaml').write_bytes(Path(f'configs/{name}_5v5_v36.yaml').read_bytes())

if __name__=='__main__':main()
