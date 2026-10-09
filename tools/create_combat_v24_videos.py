"""Twenty shared diagnostic seeds, symmetric representative selection, actual MP4s.

This tool writes only its output directory. It never trains or selects checkpoints.
"""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from algorithm.common.protocol import config_sha256
from env.geometry import engagement_geometry
from env.models import AircraftState
from tools.combat_visualization import checkpoint_sha256, dump_metadata, read_trace, ensure_fresh_output
from tools.record_combat_episode import load_recording_policy, record
from tools.render_combat_episode import render

RUNS = {
    'MAPPO': ('mappo', 'mappo_8v8_formal.yaml'),
    'RMAPPO': ('rmappo', 'rmappo_8v8.yaml'),
    'EA-MAPPO': ('ea_mappo', 'ea_mappo_8v8.yaml'),
    'STEA-MAPPO': ('stea_mappo', 'stea_mappo_8v8_formal.yaml'),
}
SEEDS = list(range(66000000, 66000020))


def protected_hashes():
    """Prove algorithms, environment and formal configs were not modified."""
    paths = [p for directory in ('algorithm', 'env', 'configs')
             for p in (ROOT / directory).rglob('*') if p.suffix in ('.py', '.yaml')]
    return {str(p.relative_to(ROOT)): checkpoint_sha256(p) for p in sorted(paths)}


def inspect_final(algorithm, run_dir, device):
    checkpoint = run_dir / 'checkpoint_3000000.pt'
    if not checkpoint.is_file():
        return None, {'algorithm': algorithm, 'eligible': False, 'reason': 'missing final checkpoint_3000000.pt'}
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    extra = state.get('extra', {})
    inventory = {'algorithm': algorithm, 'checkpoint': str(checkpoint),
        'actual_algorithm': state.get('algorithm'), 'sampled_steps': state.get('sampled_steps'),
        **{key: extra.get(key) for key in ('training_seed', 'environment_version', 'observation_dim',
            'action_dim', 'num_agents', 'training_smoke', 'environment_config_sha256', 'algorithm_config_sha256')},
        'critic_type': state.get('critic_type'), 'checkpoint_sha256': checkpoint_sha256(checkpoint)}
    try:
        env = yaml.safe_load((run_dir / 'env_config.yaml').read_text())
        config = yaml.safe_load((run_dir / 'algorithm_config.yaml').read_text())
        formal_config = yaml.safe_load((ROOT / 'configs' / RUNS[algorithm][1]).read_text())
        current_env = yaml.safe_load((ROOT / 'configs/combat_environment_v24.yaml').read_text())
        expected = {'environment_version': '2.4', 'observation_dim': 104, 'action_dim': 3,
            'num_agents': 8, 'training_seed': 1, 'training_smoke': False}
        if state.get('algorithm') != algorithm or state.get('sampled_steps') != 3000000:
            raise RuntimeError('final algorithm/step identity mismatch')
        if any(extra.get(key) != value for key, value in expected.items()):
            raise RuntimeError('final seed/environment/dimension/formal-mode metadata mismatch')
        if config_sha256(config) != config_sha256(formal_config):
            raise RuntimeError('snapshot differs from current formal algorithm protocol (legacy run excluded)')
        if config_sha256(env) != config_sha256(current_env):
            raise RuntimeError('snapshot differs from current v2.4 environment')
        # The official validator, factory and trainer load are all mandatory.
        trainer, _, provenance = load_recording_policy(checkpoint, env, config, device)
        inventory['eligible'] = True
        return (trainer, env, provenance), inventory
    except (ValueError, RuntimeError, KeyError, FileNotFoundError) as error:
        inventory.update(eligible=False, reason=str(error))
        return None, inventory


def analyze_episode(trace, metadata, env_config):
    """Describe observed geometry and events, without inferring actor intentions."""
    dt, events = float(metadata['dt']), metadata['events']
    result = {key: value for key, value in metadata.items() if key not in ('events', 'trace_shapes', 'feature_names')}
    per_agent, illegal = {}, []
    weapon = env_config['weapon']
    for side in ('red', 'blue'):
        fires = [e for e in events if e['type'] == 'fire_attempt' and e['side'] == side]
        kills = [e for e in events if e['type'] == 'attack_kill' and e['side'] == side]
        result[f'time_to_first_{side}_fire'] = min((e['step']*dt for e in fires), default=None)
        result[f'time_to_first_{side}_kill'] = min((e['step']*dt for e in kills), default=None)
        rows = []
        for agent in range(trace[f'{side}_alive'].shape[1]):
            own = [e for e in fires if e['attacker'] == agent]
            credited = [e for e in kills if agent in e['attackers']]
            rows.append({'agent': agent+1, 'fire_attempts': len(own), 'hits': sum(e['hit'] for e in own),
                'kills': len(credited), 'shared_kill_events': sum(len(e['attackers']) > 1 for e in credited),
                'fractional_kill_credit': sum(1/len(e['attackers']) for e in credited),
                'final_alive': bool(trace[f'{side}_alive'][-1, agent])})
        per_agent[side] = rows
        first = min((e['step'] for e in kills), default=None)
        result[f'{side}_kills_within_5s_of_first_kill'] = sum(e['step'] <= first+5/dt for e in kills) if first is not None else 0
        result[f'{side}_first_to_last_kill_s'] = (max(e['step'] for e in kills)-first)*dt if first is not None else None
        result[f'{side}_distinct_firing_agents'] = len({e['attacker'] for e in fires})
        result[f'{side}_distinct_fire_targets'] = len({e['target'] for e in fires})
        result[f'{side}_shared_kill_events'] = sum(len(e['attackers']) > 1 for e in kills)
        if side == 'red':
            for event in fires:
                attacker, target = AircraftState(*event['attacker_state']), AircraftState(*event['target_state'])
                g = engagement_geometry(attacker, target)
                valid = (weapon['range_min'] <= g.distance <= weapon['range_max']
                    and g.off_boresight <= weapon['off_boresight_angle_max']
                    and g.target_aspect <= weapon['target_aspect_angle_max'])
                if not valid:
                    illegal.append({'step': event['step'], 'attacker': event['attacker'], 'target': event['target'],
                        'distance': g.distance, 'off_boresight': g.off_boresight, 'target_aspect': g.target_aspect})
            result['red_fire_distance_mean'] = float(np.mean([e['distance'] for e in fires])) if fires else None
            result['red_fire_off_boresight_deg_max'] = float(np.rad2deg(max((e['off_boresight'] for e in fires), default=0)))
            result['red_fire_target_aspect_deg_max'] = float(np.rad2deg(max((e['target_aspect'] for e in fires), default=0)))
            # Separated rear-side entries within 5s, an observable pincer proxy, not proof of intent.
            pincer_targets = set()
            for i, event in enumerate(fires):
                vector = np.asarray(event['start'])-np.asarray(event['end'])
                bearing = np.arctan2(vector[1], vector[0])
                for other in fires[:i]:
                    if (other['target'] != event['target'] or other['attacker'] == event['attacker']
                            or abs(other['step']-event['step'])*dt > 5):
                        continue
                    ov = np.asarray(other['start'])-np.asarray(other['end'])
                    delta = (bearing-np.arctan2(ov[1], ov[0])+np.pi)%(2*np.pi)-np.pi
                    if abs(delta) >= np.pi/4:
                        pincer_targets.add(event['target'])
            result['red_separated_rear_entry_targets_5s'] = len(pincer_targets)
    result['per_agent'] = per_agent
    result['illegal_red_fire_count'] = len(illegal)
    result['illegal_red_fires'] = illegal
    red, blue = trace['red_kinematics'], trace['blue_kinematics']
    ra, ba = trace['red_alive'], trace['blue_alive']
    concentrations, lure_counts = [], []
    for frame in range(len(ra)):
        r, b = np.flatnonzero(ra[frame]), np.flatnonzero(ba[frame])
        if len(r) and len(b):
            distances = np.linalg.norm(red[frame, r, :3][:, None]-blue[frame, b, :3][None, :], axis=-1)
            if len(r) >= 2 and len(b) >= 2:
                counts = np.bincount(distances.argmin(axis=1), minlength=len(b))
                concentrations.append(counts.max()/len(r))
            if 5 <= trace['time_s'][frame] <= 15:
                lure_counts.append(int(np.bincount(distances.argmin(axis=0), minlength=len(r)).max()))
    result['mean_red_nearest_target_concentration_multi_target'] = float(np.mean(concentrations)) if concentrations else None
    result['max_blue_nearest_pursuers_one_red_5_to_15s'] = max(lure_counts, default=0)
    radius = np.linalg.norm(red[:, :, :2], axis=-1)
    result['red_alive_time_fraction_near_boundary_90pct'] = float(np.mean(radius[ra] >= .9*metadata['arena_radius']))
    action_alive = ra[:-1]
    actions = trace['red_actions']
    result['red_action_saturation_fraction'] = np.mean(np.abs(actions[action_alive]) > .95, axis=0).tolist()
    adjacent_alive = ra[:-2] & ra[1:-1]
    result['red_action_mean_absolute_step_change'] = np.mean(np.abs(np.diff(actions, axis=0))[adjacent_alive], axis=0).tolist() if adjacent_alive.any() else [0., 0., 0.]
    path_ratios, turning = [], []
    for agent in range(ra.shape[1]):
        deaths = np.flatnonzero(~ra[:, agent])
        last = int(deaths[0]) if len(deaths) else len(ra)-1
        path = red[:last+1, agent]
        distance = np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1).sum()
        chord = np.linalg.norm(path[-1, :3]-path[0, :3])
        path_ratios.append(float(distance/max(chord, 100.)))
        delta = (np.diff(path[:, 5])+np.pi)%(2*np.pi)-np.pi
        turning.append(float(np.abs(delta).sum()/(2*np.pi)))
    result['red_mean_path_length_over_displacement_floor100m'] = float(np.mean(path_ratios))
    result['red_mean_absolute_heading_turns'] = float(np.mean(turning))
    timeline = []
    for frame in sorted(set(range(0, len(ra), max(1, round(5/dt)))) | {len(ra)-1}):
        r, b = np.flatnonzero(ra[frame]), np.flatnonzero(ba[frame])
        distances = np.linalg.norm(red[frame, r, :3][:, None]-blue[frame, b, :3][None, :], axis=-1)
        timeline.append({'time_s': float(trace['time_s'][frame]),
            'red_alive': r.tolist(), 'blue_alive': b.tolist(),
            'red_nearest_blue': {int(agent)+1: int(b[target])+1 for agent, target in zip(r, distances.argmin(axis=1))} if len(b) else {},
            'blue_nearest_red': {int(agent)+1: int(r[target])+1 for agent, target in zip(b, distances.argmin(axis=0))} if len(r) else {},
            'red_centroid': np.mean(red[frame, r, :3], axis=0).tolist() if len(r) else None,
            'blue_centroid': np.mean(blue[frame, b, :3], axis=0).tolist() if len(b) else None,
            'mean_red_nearest_blue_distance_m': float(distances.min(axis=1).mean()) if distances.size else None})
    result['timeline_5s'] = timeline
    return result


def select_representative_seeds(records):
    """Name-independent, deterministic rules; ties always use the lower seed."""
    grouped = {}
    for row in records:
        grouped.setdefault(int(row['episode_seed']), []).append(row)
    if len(grouped) < 3:
        raise ValueError('at least three common seeds are required')
    algorithms = {r['algorithm'] for r in records}
    if any(len(rows) != len(algorithms) or {r['algorithm'] for r in rows} != algorithms for rows in grouped.values()):
        raise ValueError('each seed must contain every algorithm exactly once')
    # Sort numeric values before aggregation to avoid algorithm-order floating point ties.
    means = {seed: float(np.mean(sorted(r['episode_steps'] for r in rows))) for seed, rows in grouped.items()}
    median = float(np.median(list(means.values())))
    typical = min(means, key=lambda seed: (abs(means[seed]-median), seed))
    def hard_key(seed):
        rows = grouped[seed]
        outcomes = {(bool(r['red_win']), bool(r['blue_win']), bool(r['draw']), r['termination_reason']) for r in rows}
        losses = [8-r['red_survivors'] for r in rows]
        lengths = [r['episode_steps'] for r in rows]
        returns = [r['episode_return'] for r in rows]
        return (len(outcomes)-1, max(losses)-min(losses),
                (max(lengths)-min(lengths))/1000+(max(returns)-min(returns))/160, -seed)
    hard = max((s for s in means if s != typical), key=hard_key)
    fast = min((s for s in means if s not in (typical, hard)), key=lambda s: (means[s], s))
    return {'typical_seed': typical, 'hard_seed': hard, 'fast_seed': fast,
        'rules': {'typical': 'closest to median cross-algorithm mean episode steps; lower-seed tie',
            'hard': 'outcome-category disparity, then Red-loss range, then length range/1000 + return range/160; lower-seed tie; exclude typical',
            'fast': 'minimum cross-algorithm mean steps; lower-seed tie; exclude typical/hard'},
        'mean_steps_by_seed': means}


def probe_video(path):
    command = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_read_frames,codec_name:format=duration', '-of', 'json', str(path)]
    data = json.loads(subprocess.check_output(command, text=True))
    stream = data['streams'][0]
    result = {'duration_s': float(data['format']['duration']), 'width': int(stream['width']),
        'height': int(stream['height']), 'frame_count': int(stream['nb_read_frames']), 'codec': stream['codec_name']}
    if any(result[key] <= 0 for key in ('duration_s', 'width', 'height', 'frame_count')):
        raise RuntimeError(f'invalid MP4: {path}')
    return result


def collect(output, device, resume=False):
    if resume:
        if not (output / 'checkpoint_inventory.json').is_file():
            raise RuntimeError('resume requires an existing diagnostic inventory')
        previous_inventory = json.loads((output / 'checkpoint_inventory.json').read_text())
    else:
        ensure_fresh_output(output)
        previous_inventory = []
    before = protected_hashes()
    inventory, records = [], []
    for algorithm, (slug, _) in RUNS.items():
        run_dir = ROOT / f'outputs/{slug}_v24_8v8_seed1_3m'
        policy, item = inspect_final(algorithm, run_dir, device)
        previous = next((row for row in previous_inventory if row['algorithm'] == algorithm), None)
        if previous and previous.get('checkpoint_sha256') != item.get('checkpoint_sha256'):
            raise RuntimeError('resume checkpoint differs from original diagnostic inventory')
        inventory.append(item)
        dump_metadata(output / 'checkpoint_inventory.json', inventory)
        print(f"[CHECKPOINT] {algorithm} eligible={item['eligible']} {item.get('reason', '')}", flush=True)
        if policy is None:
            continue
        trainer, config, provenance = policy
        for seed in SEEDS:
            directory = output / 'diagnostics' / slug / f'seed_{seed}'
            if resume and (directory / 'metadata.json').exists():
                metadata = json.loads((directory / 'metadata.json').read_text())
                if (metadata['episode_seed'] != seed or metadata['checkpoint_sha256'] != provenance['checkpoint_sha256']
                        or metadata['environment_config_sha256'] != config_sha256(config)
                        or metadata['algorithm_config_sha256'] != provenance['algorithm_config_sha256']):
                    raise RuntimeError('cached trace provenance mismatch')
            else:
                metadata = record(trainer, config, seed, directory, provenance)
            analysis = analyze_episode(read_trace(directory / 'episode_trace.npz'), metadata, config)
            dump_metadata(directory / 'behavior_analysis.json', analysis)
            if analysis['illegal_red_fire_count']:
                raise RuntimeError(f'illegal Red attack: {directory}; STOP; see behavior_analysis.json')
            records.append(analysis)
            print(f"[EPISODE] {algorithm} seed={seed} steps={metadata['episode_steps']} win={metadata['red_win']} Red={metadata['red_survivors']} Blue={metadata['blue_survivors']} illegal=0", flush=True)
        del trainer, policy
        torch.cuda.empty_cache()
    if not records:
        raise RuntimeError('no eligible formal final checkpoint')
    selected = select_representative_seeds(records)
    dump_metadata(output / 'selected_seeds.json', selected)
    dump_metadata(output / 'summary.json', records)
    scalars = [key for key, value in records[0].items() if not isinstance(value, (dict, list))]
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=scalars, extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    after = protected_hashes()
    if before != after:
        raise RuntimeError('protected source/config changed during diagnostics')
    for item in inventory:
        if item.get('checkpoint') and checkpoint_sha256(item['checkpoint']) != item['checkpoint_sha256']:
            raise RuntimeError('original checkpoint changed during diagnostics')
    dump_metadata(output / 'preservation_check.json', {'protected_sources_and_configs_unchanged': True,
        'checkpoint_hashes_unchanged': True, 'source_hashes': after})
    print('[SELECTED] '+json.dumps({f'{mode}_seed': selected[f'{mode}_seed'] for mode in ('typical', 'hard', 'fast')}), flush=True)


def render_selected(output, overwrite=False):
    records = json.loads((output / 'summary.json').read_text())
    selected = json.loads((output / 'selected_seeds.json').read_text())
    probes = {}
    for algorithm in sorted({r['algorithm'] for r in records}):
        slug = RUNS[algorithm][0]
        for mode in ('typical', 'hard', 'fast'):
            seed = selected[f'{mode}_seed']
            source = output / 'diagnostics' / slug / f'seed_{seed}'
            directory = output / slug / mode
            directory.mkdir(parents=True, exist_ok=True)
            for filename in ('episode_trace.npz', 'metadata.json', 'behavior_analysis.json'):
                target = directory / filename
                if overwrite or not target.exists():
                    shutil.copy2(source / filename, target)
            trace, metadata = directory / 'episode_trace.npz', directory / 'metadata.json'
            if overwrite or not (directory / 'trajectory.png').exists():
                render(trace, metadata, directory / 'trajectory.png')
            print(f'[RENDER] {algorithm} {mode} seed={seed}', flush=True)
            if overwrite or not (directory / 'video.mp4').exists():
                render(trace, metadata, directory / 'video.mp4', stride=4, fps=20)
            probes[f'{slug}/{mode}'] = probe_video(directory / 'video.mp4')
            dump_metadata(output / 'video_validation.json', probes)
            print(f'[VIDEO] {slug}/{mode} {probes[f"{slug}/{mode}"]}', flush=True)
    write_gallery(output, records, selected)


def write_gallery(output, records, selected):
    """A local page lets the user watch the common seeds side by side."""
    sections = []
    realtime = True
    for mode in ('typical', 'hard', 'fast'):
        seed = selected[f'{mode}_seed']
        cards = []
        for row in sorted((r for r in records if r['episode_seed'] == seed), key=lambda r: r['algorithm']):
            slug = RUNS[row['algorithm']][0]
            prefix = f'{slug}/{mode}'
            movie = 'video_realtime.mp4' if (output/prefix/'video_realtime.mp4').is_file() else 'video.mp4'
            realtime = realtime and movie == 'video_realtime.mp4'
            interactive = (f' · <a href="{prefix}/interactive.html">可操控视角 HTML</a>'
                           if (output/prefix/'interactive.html').is_file() else '')
            cards.append(f'<article><h3>{html.escape(row["algorithm"])}</h3>'
                f'<p>{row["episode_steps"]} steps · return {row["episode_return"]:.2f} · '
                f'Red {row["red_survivors"]}/8 · Blue {row["blue_survivors"]}/8</p>'
                f'<video controls preload="metadata" poster="{prefix}/trajectory.png" src="{prefix}/{movie}"></video>'
                f'<p><a href="{prefix}/{movie}">MP4</a> · <a href="{prefix}/trajectory.png">完整轨迹图</a> · '
                f'<a href="{prefix}/behavior_analysis.json">事件与行为数据</a>{interactive}</p></article>')
        sections.append(f'<section><h2>{mode.title()} · seed {seed}</h2><div class="cards">'+''.join(cards)+'</div></section>')
    (output / 'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>v2.4 8v8 对局录像</title><style>body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#20242a}'
        '.cards{display:flex;flex-wrap:wrap;gap:20px}article{flex:1;min-width:340px;padding:16px;background:white;border-radius:8px}'
        'video{width:100%}a{color:#1657a0}</style><h1>v2.4 8v8 · 共同 seed 对局录像</h1>'
        f'<p>deterministic · seed1 final 3M · {"10 fps / stride 1 · 约 1 倍模拟时间播放" if realtime else "20 fps / stride 4 · 约 8 倍模拟时间播放"}。'
        '绿色虚线为命中，橙色为未命中，x 为死亡位置。可用播放器控制暂停查看。</p>'
        +''.join(sections)+'</html>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/combat_v24_8v8_videos')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage', choices=('collect', 'render', 'all'), default='all')
    parser.add_argument('--resume', action='store_true', help='reuse completed diagnostic traces with matching provenance')
    parser.add_argument('--rerender', action='store_true', help='refresh only generated representative artifacts')
    args = parser.parse_args()
    if args.stage in ('collect', 'all'):
        if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('CUDA is mandatory for checkpoint diagnostics')
        collect(args.output, args.device, args.resume)
    if args.stage in ('render', 'all'):
        render_selected(args.output, args.rerender)


if __name__ == '__main__':
    main()
