"""Bounded CUDA updates and read-only historical regression for v3.47."""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import io
import json
import subprocess
import sys
import zipfile
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.verify_v342_v343_readiness import run, algorithm_config
from tools.verify_maac_v344_v346 import formal_update
from tools.verify_v346_readiness import VALIDATORS, configs
from tools.audit_v342_v343_history import PROBE
from tools.record_combat_episode import load_recording_policy
from algorithm.common.evaluator import evaluate


def dump(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


def historical(out):
    baseline = out / 'baseline_head'
    baseline.mkdir()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    archive = subprocess.check_output(
        ['git', 'archive', '--format=zip', 'HEAD', 'env', 'configs', 'algorithm'], cwd=ROOT)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        z.extractall(baseline)
    probe = PROBE.replace("'31','32','37','39','330','332','336','341'", "'344','346'")
    probe = probe.replace('digest.update(bytes((term,trunc)))', '''
                for side in ('red','blue'):
                    for name in ('targets','counts','sampled_targets','sampled_counts','completed_this_step'):
                        digest.update(getattr(e.dwell[side],name).tobytes())
                for key in sorted(info):
                    if key.endswith('_rewards'):
                        digest.update(info[key].tobytes())
                digest.update(bytes((term,trunc)))'''.strip())
    current = json.loads(subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True))
    before = json.loads(subprocess.check_output([sys.executable, '-c', probe], cwd=baseline, text=True))
    assert current == before, 'historical complete trajectories changed'
    dump(out / 'historical_trajectories.json', dict(baseline_revision=revision,
        complete_episodes=len(current), states_observations_rewards_rng_dwell_exact=True, results=current))
    results = []
    for algo in ('mappo', 'ippo', 'maac'):
        for v in ('344', '346'):
            directory = ROOT / f'outputs/{algo}_v{v}_seed1_3m'
            candidates = [directory / 'best_eval.pt']
            if not candidates[0].exists():
                candidates = sorted((ROOT / 'outputs').glob(f'**/{algo}_v{v}/diagnostic_4096.pt'))
            if not candidates:
                raise FileNotFoundError(f'{algo} v{v} historical checkpoint required')
            path = candidates[0]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            ec, ac = configs(algo, v)
            trainer, state, _ = load_recording_policy(path, ec, ac, 'cuda')
            ev = evaluate(trainer, ec, [10_000_000])
            try:
                VALIDATORS[algo](state, *configs(algo, '347'))
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError('historical checkpoint accepted as v3.47')
            assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
            results.append(dict(algorithm=algo, version=v, checkpoint=str(path), sha256=digest,
                read_only=True, cuda_load_evaluation_pass=True, v347_rejected=True, evaluation=ev))
    dump(out / 'historical_checkpoints.json', results)
    print('Historical trajectories and six CUDA checkpoint evaluations PASS', flush=True)


def cuda(out):
    results = [run(algo, '347', out / f'{algo}_v347') for algo in ('mappo', 'ippo')]
    results.append(formal_update('347', out / 'maac_v347'))
    checks = checkpoint_checks(out)
    dump(out / 'cuda_summary.json', dict(updates=results, checkpoint_checks=checks,
        formal_3m_training_started=False))
    print('All three bounded CUDA updates and checkpoint contracts PASS', flush=True)


def checkpoint_checks(out):
    checks = []
    for algo, filename in (('mappo', 'diagnostic_4096.pt'), ('ippo', 'diagnostic_4096.pt'),
                           ('maac', 'diagnostic_bounded.pt')):
        state = torch.load(out / f'{algo}_v347' / filename, map_location='cuda', weights_only=False)
        ec, ac = configs(algo, '347')
        VALIDATORS[algo](state, ec, ac)
        for v in ('338', '341', '344', '346'):
            # IPPO begins at v3.41.
            if algo == 'ippo' and v == '338':
                continue
            try:
                VALIDATORS[algo](state, *configs(algo, v))
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError('v3.47 checkpoint accepted under historical identity')
        for key in ('environment_config_sha256', 'algorithm_config_sha256',
                    'observation_dim', 'action_dim', 'num_agents'):
            if key not in state['extra']:
                continue
            bad = deepcopy(state)
            bad['extra'][key] = 'invalid'
            try:
                VALIDATORS[algo](bad, ec, ac)
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError('corrupt checkpoint identity accepted: ' + key)
        checks.append(dict(algorithm=algo, historical_versions_rejected=True, corrupt_identity_rejected=True))
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('history', 'cuda', 'checkpoints'), required=True)
    p.add_argument('--input', type=Path, help='Completed bounded CUDA directory for checkpoint rechecks')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory; no CPU fallback')
    if args.output.exists():
        raise FileExistsError('A fresh diagnostic output directory is required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    if args.mode == 'checkpoints':
        if args.input is None:
            p.error('--input is required for checkpoint rechecks')
        checks = checkpoint_checks(args.input)
        results = [json.loads((args.input / f'{algo}_v347/summary.json').read_text())
                   for algo in ('mappo', 'ippo', 'maac')]
        dump(args.output / 'cuda_summary.json', dict(updates=results, checkpoint_checks=checks,
             formal_3m_training_started=False))
        print('Saved bounded CUDA results and checkpoint rechecks PASS', flush=True)
    else:
        (historical if args.mode == 'history' else cuda)(args.output)


if __name__ == '__main__':
    main()
