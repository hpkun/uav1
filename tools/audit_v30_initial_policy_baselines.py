"""Read-only step-zero chance-win audit with batched CUDA actors and CPU workers.

No trainer, optimizer, PPO update, checkpoint selection or environment mutation.
Episode policy RNG is independent of the environment RNG and worker scheduling.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from collections import Counter, deque
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import time
import numpy as np
import torch
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config
from env.sensor import sensor_geometry
from algorithm.mappo.networks import SharedMAPPOActor
from algorithm.rmappo.protocol import require_cuda
from tools.audit_combat_v30 import summarize, verify_transition


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def wilson(wins, n):
    if not n:
        return None
    p, z = wins / n, 1.959963984540054
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [center-half, center+half]


def conditional(rows, predicate):
    subset = [r for r in rows if predicate(r)]
    wins = sum(r['red_win'] for r in subset)
    return {'episodes': len(subset), 'red_wins': wins,
            'red_win_probability': wins/len(subset) if subset else None,
            'wilson_95': wilson(wins, len(subset))}


def first_order(row, event):
    red, blue = row[f'red_first_{event}_step'], row[f'blue_first_{event}_step']
    if red is None and blue is None:
        return 'neither'
    if red is None:
        return 'blue_first'
    if blue is None:
        return 'red_first'
    return 'red_first' if red < blue else 'blue_first' if blue < red else 'tie'


def snapshot(e):
    return {side: [[s.x, s.y, s.z, s.psi, s.theta, s.v, s.alive] for s in getattr(e, side)]
            for side in ('red', 'blue')}


def pair_trace(frame, i, j):
    from env.models import AircraftState
    from env.geometry import engagement_geometry
    a, b = frame['red'][i], frame['blue'][j]
    red = AircraftState(a[0], a[1], a[2], a[5], a[4], a[3], a[6])
    blue = AircraftState(b[0], b[1], b[2], b[5], b[4], b[3], b[6])
    g = engagement_geometry(red, blue)
    reverse = engagement_geometry(blue, red)
    return {'step': frame['step'], 'distance': g.distance,
            'red_off_boresight_deg': math.degrees(g.off_boresight),
            'blue_off_boresight_deg': math.degrees(reverse.off_boresight),
            'relative_heading_deg': math.degrees(math.atan2(math.sin(a[3]-b[3]), math.cos(a[3]-b[3]))),
            'red_position': a[:3], 'blue_position': b[:3]}


class EpisodeProbe:
    """Observe the exact pre-combat entry geometry without altering decisions."""
    def __init__(self, e):
        self.e = e
        self.first_detection = dict(red=None, blue=None)
        self.entries = dict(red=0, blue=0)
        self.window = None
        self.history = deque(maxlen=11)
        self.actions_abs = np.zeros(3)
        self.action_count = 0
        self.original = e._entry_attempts
        e._entry_attempts = self.entry
        self.observe_detection()

    def observe_detection(self):
        for side, other in (('red', 'blue'), ('blue', 'red')):
            visible = sensor_geometry(getattr(self.e, side), getattr(self.e, other), self.e.config['sensor'])[0]
            if visible.any() and self.first_detection[side] is None:
                self.first_detection[side] = self.e.steps

    def entry(self, attackers, targets, fire_states, side):
        e = self.e
        current, distance = e.weapon.eligibility(attackers, targets, getattr(e, f'{side}_ammo'))
        self.entries[side] += int((current & ~fire_states.previous_eligible).sum())
        if side == 'red':
            self.observe_detection()
            frame = {'step': e.steps, **snapshot(e)}
            self.history.append(frame)
            blue = e.weapon.eligibility(e.blue, e.red, e.blue_ammo)[0]
            if self.window is None and (current.any() or blue.any()):
                eligible = current | blue.T
                pairs = np.argwhere(eligible)
                i, j = min(pairs, key=lambda p: (distance[tuple(p)], int(p[0]), int(p[1])))
                i, j = int(i), int(j)
                self.window = {'step': e.steps, 'red_eligible_pairs': int(current.sum()),
                    'blue_eligible_pairs': int(blue.sum()), 'pair': [i, j],
                    'distance': float(distance[i, j]),
                    'trace_last_10_steps_plus_entry': [pair_trace(f, i, j) for f in self.history],
                    'mean_abs_red_action_before_entry': (self.actions_abs/max(1, self.action_count)).tolist()}
        return self.original(attackers, targets, fire_states, side)

    def before_step(self, actions):
        if self.window is None:
            values = np.abs(actions[self.e.red_alive_mask.astype(bool)])
            self.actions_abs += values.sum(axis=0)
            self.action_count += len(values)

    def fields(self):
        return {**{f'{side}_first_detection_step': self.first_detection[side] for side in ('red', 'blue')},
                **{f'{side}_eligible_pair_entries': self.entries[side] for side in ('red', 'blue')},
                'first_weapon_window': self.window}


def worker(connection, config):
    torch.set_num_threads(1)
    e = probe = None
    try:
        while True:
            op, value = connection.recv()
            if op == 'close':
                break
            if op == 'reset':
                e = MultiUAVCombatEnv(config)
                obs, _ = e.reset(value)
                probe = EpisodeProbe(e)
                total = np.zeros(5)
                connection.send(('ok', (obs, e.red_alive_mask)))
            elif op == 'step':
                actions = e.fixed_policy.team_actions(e.red, e.blue) if value is None else value
                probe.before_step(actions)
                obs, reward, terminated, truncated, info = e.step(actions)
                verify_transition(e, obs, reward, info)
                assert info['red_success'] == info['red_win']
                assert not (terminated and truncated)
                total += reward
                record = None
                if terminated or truncated:
                    assert sum(bool(info[k]) for k in ('red_win', 'blue_win', 'draw')) == 1
                    assert truncated == info['timeout']
                    assert np.isclose(total.sum(), sum(info[f'episode_{k}_total'] for k in ('event','outcome','adv','safe')), atol=2e-4)
                    record = jsonable({**info, **probe.fields(), 'episode_return': float(total.sum()),
                                       'mean_agent_episode_return': float(total.mean())})
                connection.send(('ok', (obs, e.red_alive_mask, record)))
    except BaseException:
        import traceback
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()


def receive(connection):
    status, result = connection.recv()
    if status != 'ok':
        raise RuntimeError(result)
    return result


class Moments:
    def __init__(self):
        self.n = 0
        self.values = {}

    def add(self, **values):
        self.n += len(next(iter(values.values())))
        for k, a in values.items():
            a = np.asarray(a, dtype=float)
            row = self.values.setdefault(k, [np.zeros(3), np.zeros(3), np.zeros(3)])
            row[0] += a.sum(axis=0)
            row[1] += (a*a).sum(axis=0)
            row[2] += np.abs(a).sum(axis=0)

    def result(self):
        return {'alive_agent_decisions': self.n, 'dimensions': ['heading','pitch','speed'],
                **{k: {'mean': (v[0]/self.n).tolist(),
                       'std': np.sqrt(np.maximum(0, v[1]/self.n-(v[0]/self.n)**2)).tolist(),
                       'mean_absolute': (v[2]/self.n).tolist()} for k, v in self.values.items()}}


def summarize_baseline(rows):
    result = summarize(rows)
    result['red_win_95_ci'] = wilson(result['red_win_episodes'], len(rows))
    counts = Counter(r['termination_reason'] for r in rows)
    result['termination_reasons'] = {k: counts[k] for k in (
        'red_win_elimination','blue_win_elimination','draw_mutual_destruction',
        'red_win_timeout_survivors','blue_win_timeout_survivors','draw_timeout_equal_survivors')}
    for side in ('red', 'blue'):
        result[f'{side}_distributions'] = {key: dict(sorted(Counter(r[f'{side}_{key}'] for r in rows).items()))
            for key in ('survivors','ammo_used','weapon_hits','attack_kills','fire_attempts','eligible_pair_entries')}
        attempts = sum(r[f'{side}_fire_attempts'] for r in rows)
        result[f'{side}_empirical_hit_rate'] = sum(r[f'{side}_weapon_hits'] for r in rows)/attempts if attempts else None
        result[f'{side}_mean_eligible_pair_entries'] = float(np.mean([r[f'{side}_eligible_pair_entries'] for r in rows]))
        result[f'{side}_kill_count_rates'] = {k: sum(r[f'{side}_attack_kills']==k for r in rows)/len(rows) for k in range(6)}
        result[f'{side}_per_aircraft_ammo_used_distribution'] = dict(sorted(Counter(
            6-ammo for r in rows for ammo in r[f'{side}_ammo']).items()))
        result[f'{side}_ammo_exhaustion_episode_rate'] = sum(0 in r[f'{side}_ammo'] for r in rows)/len(rows)
        for event in ('detection','fire_window','attempt','hit','kill'):
            times = [r[f'{side}_first_{event}_step'] for r in rows if r[f'{side}_first_{event}_step'] is not None]
            result[f'{side}_first_{event}'] = {'episode_count':len(times), 'mean_step':float(np.mean(times)) if times else None,
                                            'median_step':float(np.median(times)) if times else None}
        attempters = [r for r in rows if r[f'{side}_fire_attempts'] > 0]
        result[f'{side}_hit_given_attempt'] = sum(r[f'{side}_weapon_hits'] > 0 for r in attempters)/len(attempters) if attempters else None
        result[f'{side}_kill_given_attempt'] = sum(r[f'{side}_attack_kills'] > 0 for r in attempters)/len(attempters) if attempters else None
    result['first_event_advantage'] = {event: {order: conditional(rows, lambda r: first_order(r,event)==order)
        for order in ('red_first','blue_first','tie','neither')} for event in ('attempt','hit','kill')}
    result['red_win_given_red_kills'] = {k: conditional(rows, lambda r: r['red_attack_kills']==k) for k in range(6)}
    windows = [r['first_weapon_window'] for r in rows if r['first_weapon_window'] is not None]
    result['first_weapon_window_summary'] = {key: float(np.mean([w[key] for w in windows])) if windows else None
                                            for key in ('step','distance','red_eligible_pairs','blue_eligible_pairs')}
    closing = []
    for w in windows:
        trace = w['trace_last_10_steps_plus_entry']
        if len(trace)<2:
            continue
        first, last = trace[0], trace[-1]
        red0,blue0 = np.asarray(first['red_position']),np.asarray(first['blue_position'])
        los = (blue0-red0)/max(1e-12,np.linalg.norm(blue0-red0))
        red_closing = float(np.dot(np.asarray(last['red_position'])-red0,los))
        blue_closing = float(-np.dot(np.asarray(last['blue_position'])-blue0,los))
        closing.append([red_closing,blue_closing,float(last['distance']<first['distance'])])
    result['pre_entry_closing_summary'] = {
        'episodes':len(closing), 'mean_red_displacement_toward_blue_m':float(np.mean([c[0] for c in closing])) if closing else None,
        'mean_blue_displacement_toward_red_m':float(np.mean([c[1] for c in closing])) if closing else None,
        'distance_decreasing_episode_rate':float(np.mean([c[2] for c in closing])) if closing else None,
        'red_mean_absolute_pre_entry_action':np.mean([w['mean_abs_red_action_before_entry'] for w in windows],axis=0).tolist() if windows else None}
    result['red_win_elimination_count'] = sum(r['termination_reason']=='red_win_elimination' for r in rows)
    result['red_win_timeout_count'] = sum(r['termination_reason']=='red_win_timeout_survivors' for r in rows)
    result['critical_noncombat_win'] = bool(result['red_win_without_red_attack_kills_count'] or result['red_win_blue_majority_noncombat_death_count'])
    return result


def scene_audit(config, seed_base, n=10000):
    e = MultiUAVCombatEnv(config)
    values = {s: [] for s in ('red', 'blue')}
    for seed in range(seed_base, seed_base+n):
        e.reset(seed)
        for side, other in (('red','blue'),('blue','red')):
            states, opponents = getattr(e,side), getattr(e,other)
            center = np.mean([[s.x,s.y] for s in states],axis=0)
            nominal = math.atan2(-center[1], -center[0])
            _, distance, ata, _ = sensor_geometry(states, opponents, config['sensor'])
            errors = [abs(math.atan2(math.sin(s.psi-nominal),math.cos(s.psi-nominal))) for s in states]
            values[side].append([np.mean([s.v for s in states]),np.mean([s.altitude for s in states]),
                                np.mean(errors),distance.mean(),distance.min(axis=1).mean(),
                                np.mean(config['sensor']['range_max']-distance.min(axis=1)),
                                np.mean(config['sensor']['off_boresight_angle_max']-np.arccos(ata).min(axis=1))])
    red, blue = np.asarray(values['red']), np.asarray(values['blue'])
    labels = ['speed_mps','altitude_m','abs_heading_error_rad','pair_distance_m','nearest_enemy_distance_m',
              'nearest_detection_range_margin_m','best_detection_angle_margin_rad']
    return {'reset_seeds': n, 'seed_base':seed_base, 'metrics': {k: {'red_mean':float(red[:,i].mean()),
        'blue_mean':float(blue[:,i].mean()), 'paired_red_minus_blue':float((red[:,i]-blue[:,i]).mean()),
        'paired_difference_standard_error':float((red[:,i]-blue[:,i]).std(ddof=1)/math.sqrt(n))} for i,k in enumerate(labels)}}


def build_actor(config, seed, device):
    n, i = config['network'], config['implementation']
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    return SharedMAPPOActor(n['observation_dim'],n['action_dim'],n['actor_hidden_layers'][0],
        i['log_std_min'],i['log_std_max'],i['actor_activation'],i['policy_std_mode'],
        i['log_std_init'],i['mean_head_init_gain']).to(device).eval()


def review_runtime_semantics():
    """Read existing logs, never load/update a trained policy."""
    cfg=yaml.safe_load((ROOT/'configs/mappo_5v5_v30.yaml').read_text())
    review={'recent_episode_window':cfg['runtime_logging']['recent_episode_window'],
        'train_win':'mean red_success over last min(100, completed episodes)',
        'train_fire':'episode has at least one red_first_fire_window_step',
        'train_kill':'episode has at least one red_first_kill_step; not elimination rate',
        'rollout_deterministic':False,'formal_eval_deterministic':True,
        'sampled_steps_per_rollout':16*cfg['training']['rollout_steps'],
        'rollout_updates_at_20480':20480//(16*cfg['training']['rollout_steps']),
        'timeout_logging_issue':{'runner_summary':'legacy red_failure_timeout comparison misses v3.0 timeout reasons',
            'step_metrics':'aggregate_combat_records overwrites legacy value when episodes complete; correct timeout flag',
            'formal_evaluator':'correct info.timeout with legacy fallback',
            'impact':'timeout summary field only; no effect on win/outcome or selection logic'},
        'eval_zero_wins_in_20_wilson_95':wilson(0,20)}
    run=ROOT/'outputs/mappo_v30_5v5_seed1_1m'
    if (run/'run_summary.json').exists():
        summary=json.loads((run/'run_summary.json').read_text())
        updates=[]
        with (run/'optimization_metrics.jsonl').open() as stream:
            for _,line in zip(range(5),stream):
                r=json.loads(line)
                updates.append({k:r[k] for k in ('sampled_steps','rollout_update','effective_ppo_epochs')})
        completed, actual_timeout = 0, 0.
        with (run/'training_metrics.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                n=row.get('evaluation_episodes',0)
                completed+=n
                actual_timeout+=n*(row.get('timeout_rate') or 0)
        import re
        first_log=next(line for line in (run/'train.log').read_text().splitlines() if '[TRAIN] steps=20480/' in line)
        episode_count=int(re.search(r'\beps=(\d+)',first_log)[1])
        win_rate=float(re.search(r'win=([\d.]+)',first_log)[1])
        review['existing_seed1_run']={'path':str(run),'first_five_updates':updates,
            'sum_effective_epochs_first_5':sum(r['effective_ppo_epochs'] for r in updates),
            'episodes_at_20480':episode_count,'console_recent_wins_at_20480':round(win_rate*min(100,episode_count)),
            'completed_episodes_summary':summary['completed_episodes'],
            'completed_episodes_sum_logged_batches':completed,
            'counts_match':completed==summary['completed_episodes'],
            'logged_completed_timeout_count':int(round(actual_timeout)),
            'correct_timeout_rate_from_batches':actual_timeout/completed if completed else None,
            'summary_reported_timeout_rate':summary['timeout_rate']}
    return review


def paired_compare(groups):
    names = list(groups)
    by_seed = {k: {r['seed']: r for r in v} for k,v in groups.items()}
    common = sorted(set.intersection(*(set(v) for v in by_seed.values())))
    comparisons = {}
    for a in range(1,6):
        det, st = f'init_det_seed{a}', f'init_stoch_seed{a}'
        pairs = [(by_seed[det][s],by_seed[st][s]) for s in common]
        comparisons[str(a)] = {'episodes':len(pairs), 'win_rate_gap_stoch_minus_det':float(np.mean([y['red_win']-x['red_win'] for x,y in pairs])),
            'det_only_win':sum(x['red_win'] and not y['red_win'] for x,y in pairs),
            'stoch_only_win':sum(y['red_win'] and not x['red_win'] for x,y in pairs),
            'both_win':sum(x['red_win'] and y['red_win'] for x,y in pairs)}
    table = [{'seed':s, 'red_win_by_policy':{k:by_seed[k][s]['red_win'] for k in names},
              'red_win_policy_count':sum(by_seed[k][s]['red_win'] for k in names)} for s in common]
    return {'common_environment_episodes':len(common),'actor_mode_pairs':comparisons,
            'zero_random_stoch1_all_win':sum(all(by_seed[k][s]['red_win'] for k in ('zero','uniform','init_stoch_seed1')) for s in common),
            'easy_seeds_by_win_count':sorted(table,key=lambda r:(-r['red_win_policy_count'],r['seed']))}


def run_group(connections, mode, actor, actor_seed, count, seed_base, policy_seed, output):
    rows, moments, active = [], Moments(), {}
    next_i = 0
    def reset_slot(slot):
        nonlocal next_i
        seed = seed_base+next_i
        next_i += 1
        connections[slot].send(('reset',seed))
        obs, alive = receive(connections[slot])
        generator = torch.Generator(device='cuda').manual_seed(policy_seed+seed-seed_base)
        active[slot] = [seed, obs, alive, generator, np.random.default_rng(policy_seed+seed-seed_base)]
    for slot in range(min(len(connections),count)):
        reset_slot(slot)
    raw_path = output/f'{mode}_seed{actor_seed}_episodes.jsonl.gz'
    with gzip.open(raw_path,'wt',encoding='utf-8') as stream, torch.inference_mode():
        while active:
            slots = sorted(active)
            masks = np.stack([active[s][2] for s in slots]).astype(bool)
            if actor is not None:
                obs = torch.as_tensor(np.stack([active[s][1] for s in slots]),device='cuda')
                distribution = actor.distribution(obs)
                raw = distribution.mean
                if mode == 'init_stoch':
                    noise = torch.stack([torch.randn((5,3),device='cuda',generator=active[s][3]) for s in slots])
                    raw = raw+distribution.scale*noise
                action = torch.tanh(raw).cpu().numpy()*masks[...,None]
                moments.add(mu=distribution.mean.cpu().numpy()[masks], sigma=distribution.scale.cpu().numpy()[masks],
                    log_std=distribution.scale.log().cpu().numpy()[masks], raw_action=raw.cpu().numpy()[masks],
                    action=action[masks], deterministic_action=torch.tanh(distribution.mean).cpu().numpy()[masks])
            elif mode == 'uniform':
                action = np.stack([active[s][4].uniform(-1,1,(5,3)).astype(np.float32) for s in slots])*masks[...,None]
            else:
                action = np.zeros((len(slots),5,3),dtype=np.float32)
            for j, slot in enumerate(slots):
                connections[slot].send(('step',None if mode=='mirror' else action[j]))
            for slot in slots:
                obs, alive, record = receive(connections[slot])
                if record is None:
                    active[slot][1:3] = [obs, alive]
                    continue
                record.update(seed=active[slot][0],policy=mode,actor_seed=actor_seed,
                              policy_sampling_seed=policy_seed+active[slot][0]-seed_base)
                rows.append(record)
                stream.write(json.dumps(record)+'\n')
                del active[slot]
                if next_i < count:
                    reset_slot(slot)
                if len(rows)%100 == 0:
                    print(f'[AUDIT] {mode} actor={actor_seed} completed={len(rows)}/{count}',flush=True)
    rows.sort(key=lambda r:r['seed'])
    assert len(rows)==count and len({r['seed'] for r in rows})==count
    return rows, moments.result() if actor is not None else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--actor-episodes',type=int,default=500)
    p.add_argument('--baseline-episodes',type=int,default=1000)
    p.add_argument('--seed-base',type=int,default=31_000_000)
    p.add_argument('--policy-seed',type=int,default=41_000_000)
    p.add_argument('--summarize-only',action='store_true',help='Rebuild summaries from existing raw records without rerunning episodes')
    args = p.parse_args()
    if args.summarize_only:
        output=Path(args.output)
        result=json.loads((output/'summary.json').read_text())
        groups={}
        for path in sorted(output.glob('*_episodes.jsonl.gz')):
            with gzip.open(path,'rt',encoding='utf-8') as stream:
                rows=[json.loads(line) for line in stream]
            row=rows[0]
            name=f'{row["policy"]}_seed{row["actor_seed"]}' if row['actor_seed'] else row['policy']
            groups[name]=rows
        result['baselines']={name:summarize_baseline(rows) for name,rows in groups.items()}
        paired=paired_compare(groups)
        result['paired_actor_modes']=paired['actor_mode_pairs']
        (output/'paired_comparison.json').write_text(json.dumps(paired,indent=2))
        (output/'summary.json').write_text(json.dumps(result,indent=2))
        (output/'source_review.json').write_text(json.dumps(review_runtime_semantics(),indent=2))
        return
    if args.actor_episodes<500 or args.baseline_episodes<1000 or args.workers<1:
        raise ValueError('requires >=500 episodes/actor, >=1000 other baselines')
    if not (31_000_000<=args.seed_base and args.seed_base+max(10000,args.baseline_episodes,args.actor_episodes)<70_000_000):
        raise ValueError('use a new diagnostic block below the 70M formal holdout')
    require_cuda('cuda')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config_path = ROOT/'configs/combat_environment_v30.yaml'
    algorithm_path = ROOT/'configs/mappo_5v5_v30.yaml'
    config = load_config(config_path)
    algorithm = yaml.safe_load(algorithm_path.read_text())
    assert algorithm['implementation']['log_std_init']==-.5
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if list(output.iterdir()):
        raise RuntimeError('audit output must be fresh')
    protected = list((ROOT/'env').glob('*.py')) + list((ROOT/'algorithm').rglob('*.py')) + list((ROOT/'configs').glob('*.yaml'))
    fingerprints = {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected}
    (output/'source_hashes_before.json').write_text(json.dumps(fingerprints,indent=2))
    (output/'env_config.yaml').write_bytes(config_path.read_bytes())
    (output/'algorithm_config.yaml').write_bytes(algorithm_path.read_bytes())
    start = time.monotonic()
    scene = scene_audit(config,args.seed_base)
    (output/'initial_scene.json').write_text(json.dumps(scene,indent=2))
    context = mp.get_context('spawn')
    connections, processes = [], []
    groups, summary, distributions = {}, {}, {}
    try:
        for _ in range(args.workers):
            parent, child = context.Pipe()
            process = context.Process(target=worker,args=(child,config),daemon=True)
            process.start(); child.close()
            connections.append(parent); processes.append(process)
        for mode in ('zero','uniform','mirror','init_det','init_stoch'):
            for actor_seed in (range(1,6) if mode.startswith('init') else [0]):
                name = f'{mode}_seed{actor_seed}' if actor_seed else mode
                actor = build_actor(algorithm,actor_seed,'cuda') if actor_seed else None
                if actor is not None:
                    state_path = output/f'actor_init_seed{actor_seed}.pt'
                    if not state_path.exists():
                        torch.save({'actor':actor.state_dict(),'seed':actor_seed,'ppo_updates':0},state_path)
                    else:
                        state = torch.load(state_path,weights_only=False,map_location='cuda')['actor']
                        assert all(torch.equal(v,state[k]) for k,v in actor.state_dict().items())
                rows, stats = run_group(connections,mode,actor,actor_seed,
                    args.actor_episodes if actor_seed else args.baseline_episodes,args.seed_base,args.policy_seed,output)
                groups[name], summary[name], distributions[name] = rows, summarize_baseline(rows), stats
                (output/'progress_summary.json').write_text(json.dumps(summary,indent=2))
                print(f'[DONE] {name} win={summary[name]["red_win_rate"]:.4f} timeout={summary[name]["timeout_rate"]:.4f}',flush=True)
                del actor
        paired = paired_compare(groups)
        (output/'paired_comparison.json').write_text(json.dumps(paired,indent=2))
    finally:
        for connection in connections:
            connection.send(('close',None))
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate();process.join()
        for connection in connections:
            connection.close()
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in fingerprints.items())
    result = {'environment_version':'3.0','cuda_device':torch.cuda.get_device_name(),'seed_base':args.seed_base,
        'policy_seed':args.policy_seed,'actor_seeds':list(range(1,6)),'workers':args.workers,
        'initial_scene':scene,'baselines':summary,'action_statistics':distributions,
        'paired_actor_modes':paired['actor_mode_pairs'],'common_environment_episodes':paired['common_environment_episodes'],
        'protected_source_hashes_unchanged':True,'ppo_updates':0,'formal_training_started':False,
        'parameter_sweep':False,'elapsed_seconds':time.monotonic()-start}
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print('[COMPLETE]',json.dumps({'episodes':sum(len(v) for v in groups.values()),'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    main()
