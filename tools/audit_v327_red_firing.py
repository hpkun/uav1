"""Read-only Red firing audit at the real pre-combat attempt hook; no extra RNG."""
import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.combat_v327 import CombatEnvironmentV327
from env.geometry import engagement_geometry
from env.v36_geometry import angle_within
from tools.record_combat_episode import load_recording_policy


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8'); return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('fresh audit directory required')
    out.mkdir(parents=True, exist_ok=True)
    protected_paths = list((ROOT/'env').glob('*.py')) + list((ROOT/'configs').glob('*.yaml'))
    protected_paths += list((ROOT/'algorithm').rglob('*.py'))
    protected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_paths}
    source = args.checkpoint
    before = source.stat(); snapshot = out/'checkpoint_snapshot.pt'
    shutil.copy2(source, snapshot); after = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(snapshot.read_bytes()).digest()
    env_config = yaml.safe_load((source.parent/'env_config.yaml').read_text())
    config = yaml.safe_load((source.parent/'algorithm_config.yaml').read_text())
    trainer, state, provenance = load_recording_policy(snapshot, env_config, config, 'cuda')
    assert env_config['environment_version'] == '3.27'
    original_entry = CombatEnvironmentV327._entry_attempts
    original_resolve = CombatEnvironmentV327._resolve_combat
    mro = CombatEnvironmentV327.__mro__
    static = dict(mro=[c.__name__ for c in mro], entry_method=original_entry.__qualname__,
        entry_file=inspect.getsourcefile(original_entry), entry_source=inspect.getsource(original_entry),
        red_super_method=next(c.__dict__['_entry_attempts'].__qualname__ for c in mro[mro.index(next(c for c in mro if '_entry_attempts' in c.__dict__))+1:] if '_entry_attempts' in c.__dict__),
        resolve_method=original_resolve.__qualname__, weapon=env_config['weapon'],fire_control=env_config['fire_control'])
    (out/'static_chain.json').write_text(json.dumps(static,indent=2),encoding='utf-8')

    def run(seed, instrument):
        e = MultiUAVCombatEnv(env_config); obs, _ = e.reset(seed)
        agent_rows, pair_rows, violations, current = [], [], [], []
        digest = hashlib.sha256()

        def entry(self, attackers, targets, fire_states, side):
            nonlocal current
            if side != 'red':
                return original_entry(self, attackers, targets, fire_states, side)
            ammo = self.red_ammo.copy(); cooldown_before = self.red_fire_cooldown_steps.copy()
            eligible, distances = self.weapon.eligibility(attackers, targets, ammo)
            actual_decrement = []
            maximum = np.maximum

            def observe_maximum(*a, **kw):
                value = maximum(*a, **kw)
                if kw.get('out') is self.red_fire_cooldown_steps:
                    actual_decrement.append(value.copy())
                return value

            current = []
            local_pairs = []
            for i, own in enumerate(attackers):
                counts = [0, 0, 0, 0]; nearest = None; pairs = []
                for j, target in enumerate(targets):
                    g = engagement_geometry(own, target)
                    live = own.alive and target.alive
                    range_ok = live and self.weapon.range_min <= g.distance <= self.weapon.range_max
                    az_ok = bool(range_ok and angle_within(abs(g.boresight_azimuth_error), self.weapon.boresight_azimuth_max))
                    full = bool(az_ok and angle_within(abs(g.boresight_elevation_error), self.weapon.boresight_elevation_max))
                    counts = [x+int(y) for x,y in zip(counts,(live,range_ok,az_ok,full))]
                    if target.alive:
                        nearest = g.distance if nearest is None else min(nearest,g.distance)
                    pairs.append(dict(step=self.steps,red=i,blue=j,live=bool(live),distance=g.distance,
                        azimuth_deg=float(np.degrees(g.boresight_azimuth_error)),
                        elevation_deg=float(np.degrees(g.boresight_elevation_error)),range_ok=bool(range_ok),
                        range_az_ok=az_ok,full_geometry=full,eligible=bool(eligible[i,j]),
                        ammo_before=int(ammo[i]),cooldown_before=int(cooldown_before[i])))
                local_pairs.extend(pairs)
                current.append(dict(step=self.steps,red=i,alive=bool(own.alive),live_blue_count=sum(s.alive for s in targets),
                    nearest_live_blue_distance=nearest,range_count=counts[1],range_az_count=counts[2],full_geometry_count=counts[3],
                    eligible_count=int(eligible[i].sum()),eligible_indices=json.dumps(np.flatnonzero(eligible[i]).tolist()),
                    ammo_before=int(ammo[i]),cooldown_before=int(cooldown_before[i]),cooldown_after_decrement=-1,
                    should_fire=False,actual_attempts=0,actual_target=-1,hit=False,kill=False,kill_targets='[]'))
            # Observe the actual decrement in the original hook, without mutation
            # or another hit call. This temporary wrapper is local to this process.
            with patch.object(np,'maximum',observe_maximum):
                attempts = original_entry(self, attackers, targets, fire_states, side)
            assert len(actual_decrement) == 1
            decremented = actual_decrement[0]
            for i,row in enumerate(current):
                own_attempts = [attempt for attempt in attempts if attempt[0]==i]
                row.update(cooldown_after_decrement=int(decremented[i]),
                    should_fire=bool(eligible[i].any() and decremented[i]==0),actual_attempts=len(own_attempts),
                    actual_target=own_attempts[0][1] if own_attempts else -1,
                    hit=bool(own_attempts[0][2]) if own_attempts else False)
                expected_target = min(np.flatnonzero(eligible[i]),key=lambda j:(distances[i,j],j)) if row['should_fire'] else -1
                if (len(own_attempts)!=int(row['should_fire']) or row['actual_target']!=expected_target
                        or self.red_ammo[i]!=ammo[i]-len(own_attempts)):
                    violations.append(dict(**row,pairs=[p for p in local_pairs if p['red']==i]))
                for p in local_pairs:
                    if p['red']==i:p['cooldown_after_decrement']=int(decremented[i])
            agent_rows.extend(current);pair_rows.extend(local_pairs)
            return attempts

        def resolve(self, red_attempts, blue_attempts):
            result=original_resolve(self,red_attempts,blue_attempts)
            for row in current:
                kills=[j for j,attackers in result[0].items() if row['red'] in attackers]
                row['kill']=bool(kills);row['kill_targets']=json.dumps(kills)
            return result

        def loop():
            nonlocal obs
            while True:
                action=trainer.act(obs[None],e.red_alive_mask[None],deterministic=True)[0]
                obs,reward,terminated,truncated,info=e.step(action)
                for values in (np.asarray([s.as_array() for s in e.red+e.blue]),obs,reward,e.red_ammo,e.blue_ammo,e.red_fire_cooldown_steps,e.blue_fire_cooldown_steps):
                    digest.update(np.ascontiguousarray(values).tobytes())
                if violations or terminated or truncated:break
            digest.update(json.dumps(e.rng.bit_generator.state,sort_keys=True).encode())
            return info
        if instrument:
            with patch.object(CombatEnvironmentV327,'_entry_attempts',entry),patch.object(CombatEnvironmentV327,'_resolve_combat',resolve):info=loop()
        else:info=loop()
        if not instrument:return digest.hexdigest(),None,None,None
        live=[r for r in agent_rows if r['alive']]
        funnel=dict(A=sum(r['live_blue_count']>0 for r in live),B=sum(r['range_count']>0 for r in live),
            C=sum(r['range_az_count']>0 for r in live),D=sum(r['full_geometry_count']>0 for r in live),
            E=sum(r['eligible_count']>0 for r in live),F=sum(r['should_fire'] for r in live),
            G=info['red_fire_attempts'],H=info['red_weapon_hits'],I=info['red_attack_kills'])
        pair=dict(total_live=sum(p['live'] for p in pair_rows),range=sum(p['range_ok'] for p in pair_rows),
            range_az=sum(p['range_az_ok'] for p in pair_rows),full_geometry=sum(p['full_geometry'] for p in pair_rows),
            weapon_eligible=sum(p['eligible'] for p in pair_rows))
        summary=dict(seed=seed,episode_length=e.steps,red_agent_steps=len(live),scheduled_agent_steps=len(agent_rows),
            funnel=funnel,pair_funnel=pair,invariant_violation_count=len(violations),
            red_fire_window_pair_steps=pair['weapon_eligible'],red_fire_window_agent_steps=funnel['E'],
            red_attempts=funnel['G'],red_hits=funnel['H'],red_kills=funnel['I'],red_ammo_used=info['red_ammo_used'],
            red_ground_losses=info['red_ground_losses'],red_boundary_losses=info['red_boundary_exits'],
            blue_attack_kills=info['blue_attack_kills'],termination_reason=info['termination_reason'])
        return digest.hexdigest(),summary,agent_rows,pair_rows

    baseline,_,_,_=run(10000000,False)
    summaries=[]
    for seed in range(10000000,10000020):
        digest,summary,agents,pairs=run(seed,True);summaries.append(summary)
        if seed==10000000:
            assert digest==baseline,'Instrumentation changed transitions or RNG'
            summary['instrumentation_rng_and_transition_hash_equal']=True
            write_csv(out/'primary_agent_steps.csv',agents);write_csv(out/'primary_pair_steps.csv',pairs)
            # One candidate per Red agent-step with no actual attempt. Rank
            # in-range first, then excess angle beyond the 30-degree window.
            candidates=[]
            by_step={(r['step'],r['red']):r for r in agents}
            for p in pairs:
                r=by_step[p['step'],p['red']]
                if not p['live'] or r['actual_attempts']:continue
                bad=[]
                if not p['range_ok']:bad.append('distance')
                if abs(p['azimuth_deg'])>30:bad.append('azimuth')
                if abs(p['elevation_deg'])>30:bad.append('elevation')
                if r['ammo_before']==0:bad.append('ammo')
                if r['cooldown_after_decrement']>0:bad.append('cooldown')
                if bad:candidates.append(dict(**p,failed_conditions='+'.join(bad)))
            candidates.sort(key=lambda p:(not p['range_ok'],max(abs(p['azimuth_deg'])-30,0)+max(abs(p['elevation_deg'])-30,0),max(p['distance']-1000,0)))
            selected=[];seen=set()
            for p in candidates:
                key=(p['step'],p['red'])
                if key not in seen:selected.append(p);seen.add(key)
                if len(selected)==10:break
            write_csv(out/'near_misses.csv',selected)
            (out/'primary_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        (out/'batch_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
        print(json.dumps(summary),flush=True)
        if summary['invariant_violation_count']:
            raise RuntimeError('Red firing invariant violation; stopping further experiments')
    flat=[dict(seed=r['seed'],episode_length=r['episode_length'],**{k:r[k] for k in ('red_fire_window_pair_steps','red_fire_window_agent_steps','red_attempts','red_hits','red_kills','red_ammo_used','red_ground_losses','red_boundary_losses','blue_attack_kills','termination_reason')}) for r in summaries]
    write_csv(out/'batch_summary.csv',flat)
    totals={k:sum(r['funnel'][k] for r in summaries) for k in 'ABCDEFGHI'}
    rates={name:sum(r['funnel'][key]>0 for r in summaries)/len(summaries) for name,key in [('range','B'),('range_az','C'),('geometry','D'),('attempt','G'),('hit','H'),('kill','I')]}
    result=dict(checkpoint_provenance=provenance,source=str(source.resolve()),checkpoint_sampled_steps=state['sampled_steps'],
        primary=summaries[0],batch_totals=totals,batch_episode_rates=rates,
        invariant_violation_count=sum(r['invariant_violation_count'] for r in summaries),
        instrumentation_rng_neutral=True,protected_sources_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items()))
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert result['protected_sources_unchanged']
    print('AUDIT_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
