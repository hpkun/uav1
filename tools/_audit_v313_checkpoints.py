"""Read-only CUDA validation: old 56D artifacts remain valid and cannot cross-load."""
from pathlib import Path
import json
import torch
import yaml
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation
from algorithm.rmappo.protocol import validate_checkpoint as rm_check, require_cuda
from algorithm.ea_mappo.protocol import validate_checkpoint as ea_check
from algorithm.stea_mappo.protocol import validate_checkpoint as stea_check
from env.config import load_config


def main():
    require_cuda('cuda')
    old = load_config('configs/combat_environment_v312.yaml')
    new = load_config('configs/combat_environment_v313.yaml')
    rows = {}
    for name, stem, checker in (('mappo','mappo',validate_checkpoint_for_evaluation),
        ('rmappo','rmappo',rm_check),('ea-mappo','ea_mappo',ea_check),('stea-mappo','stea_mappo',stea_check)):
        checkpoint = Path(f'outputs/v312_validation/cuda_smoke/{name}/latest.pt')
        if not checkpoint.exists():
            raise RuntimeError(f'old bounded validation checkpoint unavailable: {checkpoint}')
        state = torch.load(checkpoint,map_location='cuda',weights_only=False)
        cfg_old = yaml.safe_load(Path(f'configs/{stem}_5v5_v312.yaml').read_text())
        cfg_new = yaml.safe_load(Path(f'configs/{stem}_5v5_v313.yaml').read_text())
        checker(state,old,cfg_old)
        try:
            checker(state,new,cfg_new)
        except RuntimeError as error:
            assert 'environment_version' in str(error)
            rows[name] = dict(old_checkpoint_valid=True, v312_to_v313_rejected=True,
                reason=str(error), old_observation_dim=state['extra']['observation_dim'], new_observation_dim=56)
        else:
            raise AssertionError('cross-version checkpoint was accepted')
    Path('outputs/v313_validation/checkpoint_isolation.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
