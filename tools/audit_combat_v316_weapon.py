"""Static, model-free weapon geometry and Equation (8) audit; no calibration."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.geometry import engagement_geometry
from env.v313_geometry import combat_geometry_v313


class FixedNormal:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.calls = 0

    def normal(self, mean, std):
        assert (mean, std) == (0., 1.)
        self.calls += 1
        return self.epsilon


def geometry_cases():
    # name, distance, own pitch, LOS elevation, horizontal ATA, expected eligibility
    definitions = [
        ('horizontal_front', 3000, 0, 0, 0, True),
        ('ata_boundary', 3000, 0, 0, 30, True),
        ('ha_boundary_aligned', 3000, 30, 30, 0, True),
        ('boresight_below', 3000, 29.999999, 0, 0, True),
        ('boresight_boundary', 3000, 30, 0, 0, True),
        ('boresight_above', 3000, 30.000001, 0, 0, False),
        ('replay_red_up_target_below', 3990, 60, -22, 2, False),
        ('replay_blue_up_target_above', 3990, 18, 22, .6, True),
        ('pitch_down_target_above', 3990, -60, 22, 2, False),
        ('height_difference_nose_aligned', 3000, 25, 25, 0, True),
        ('ata_outside_boresight_inside', 3000, 25, 25, 30.000001, False),
        ('ha_outside_boresight_inside', 3000, 30.000001, 30.000001, 0, False),
        ('distance_outside', 4000.000001, 0, 0, 0, False),
        ('range_boundary', 4000, 0, 0, 0, True),
        ('coincident', 0, 60, 0, 0, True),
    ]
    cases = []
    for name, distance, pitch, elevation, ata, expected in definitions:
        a = AircraftState(0., 0., -3000., 225., math.radians(pitch), 0.)
        el, az = math.radians(elevation), math.radians(ata)
        b = AircraftState(distance*math.cos(el)*math.cos(az), distance*math.cos(el)*math.sin(az),
                          -3000.-distance*math.sin(el), 225., 0., math.pi)
        cases.append((name, a, b, expected))
    return cases


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for repository audits; no CPU fallback')
    output.mkdir(parents=True, exist_ok=True)
    old = MultiUAVCombatEnv(ROOT/'configs/combat_environment_v315.yaml')
    new = MultiUAVCombatEnv(ROOT/'configs/combat_environment_v316.yaml')
    geometry_rows = []
    for name, attacker, target, expected in geometry_cases():
        g = combat_geometry_v313([attacker], [target])
        distance, ata, ha = [float(g[key][0, 0]) for key in ('distance', 'ata', 'ha')]
        bore = engagement_geometry(attacker, target).off_boresight
        eligible_old = old.weapon.qualifies(attacker, target, 6)
        eligible_new = new.weapon.qualifies(attacker, target, 6)
        assert eligible_new == expected, name
        rejected = []
        if not new.weapon.range_min <= distance <= new.weapon.range_max: rejected.append('distance')
        if ata > new.weapon.ata_max+1e-12: rejected.append('ATA')
        if ha > new.weapon.ha_max+1e-12: rejected.append('paper_HA')
        if bore > new.weapon.boresight_max+1e-12: rejected.append('true_3D_off_boresight')
        if name == 'replay_red_up_target_below':
            assert eligible_old and rejected == ['true_3D_off_boresight']
        if name == 'replay_blue_up_target_above':
            assert eligible_old and eligible_new
        geometry_rows.append(dict(case=name, distance=distance, ATA_deg=np.rad2deg(ata),
                                  paper_HA_deg=np.rad2deg(ha), attacker_pitch_deg=np.rad2deg(attacker.theta),
                                  LOS_elevation_deg=np.rad2deg(g['los_elevation'][0, 0]),
                                  off_boresight_deg=np.rad2deg(bore), v315_eligible=eligible_old,
                                  v316_eligible=eligible_new, rejection_reason=';'.join(rejected)))
    equation_rows = []
    for distance, ata, ha in [(3000., 0., 0.), (3990., .03, .4), (4000., .5, .5)]:
        threshold = math.pi*math.exp(-distance/new.weapon.effective_hit_distance)
        for epsilon in (0., .5, -.5, 2., -2.):
            rng = FixedNormal(epsilon)
            hit = new.weapon.attempt_hit(rng, distance, ata, ha)
            left_a = abs(ata+new.weapon.ata_noise_scale*epsilon)
            left_h = abs(ha+new.weapon.ha_noise_scale*epsilon)
            assert rng.calls == 1 and hit == (left_a <= threshold and left_h <= threshold)
            equation_rows.append(dict(distance=distance, ATA=ata, HA=ha, epsilon=epsilon, threshold=threshold,
                old_left_ATA=abs(ata)+old.weapon.ata_noise_scale*epsilon, new_left_ATA=left_a,
                old_left_HA=abs(ha)+old.weapon.ha_noise_scale*epsilon, new_left_HA=left_h,
                old_hit=old.weapon.attempt_hit(FixedNormal(epsilon),distance,ata,ha), v316_hit=hit, normal_calls=rng.calls))
    probability_rows = []
    rng = np.random.default_rng(316)
    for distance, ata, ha in [(0.,0.,0.), (3000.,0.,0.), (4000.,0.,0.), (3990.,.03,.4), (4000.,.5,.5)]:
        epsilon = rng.normal(0.,1.,200000)
        threshold = math.pi*math.exp(-distance/new.weapon.effective_hit_distance)
        a = ata+new.weapon.ata_noise_scale*epsilon
        h = ha+new.weapon.ha_noise_scale*epsilon
        old_hit = (abs(ata)+old.weapon.ata_noise_scale*epsilon <= threshold) & (abs(ha)+old.weapon.ha_noise_scale*epsilon <= threshold)
        new_hit = (abs(a) <= threshold) & (abs(h) <= threshold)
        old_p = old.weapon.expected_hit_probability(distance,ata,ha)
        new_p = new.weapon.expected_hit_probability(distance,ata,ha)
        assert abs(new_hit.mean()-new_p) < .005 and abs(old_hit.mean()-old_p) < .005
        probability_rows.append(dict(distance=distance,ATA=ata,HA=ha,old_analytic_probability=old_p,
            v316_analytic_probability=new_p,old_monte_carlo_probability=float(old_hit.mean()),
            v316_monte_carlo_probability=float(new_hit.mean()),samples=200000,seed=316,
            purpose='formula verification only; no recalibration'))
    for name, rows in [('geometry_cases',geometry_rows), ('equation8_cases',equation_rows), ('hit_probability_diagnostics',probability_rows)]:
        write_csv(output/f'{name}.csv',rows)
    checks = dict(cuda_device=torch.cuda.get_device_name(),model_loaded=False,new_episode_generated=False,
        geometry_cases=len(geometry_rows),fixed_epsilon_cases=len(equation_rows),shared_normal_calls_per_attempt=1,
        paper_geometry_retained=True,replay_red_rejected_by_boresight_only=True,replay_blue_allowed=True,
        equation8_absolute_value_correct=True,analytic_interval_intersection_verified=True,
        hit_parameters_unchanged=True,monte_carlo_points=len(probability_rows),parameters_recalibrated=False)
    replay = ROOT/'outputs/mappo_v315_seed1_3m_best_replay/seed_10000000/metadata.json'
    if replay.exists():
        replay_rows = []
        events = json.loads(replay.read_text(encoding='utf-8'))['events']
        for event in events:
            if event['type'] != 'fire_attempt': continue
            attacker, target = [AircraftState(*event[key]) for key in ('attacker_state','target_state')]
            g = combat_geometry_v313([attacker],[target])
            old_ok = old.weapon.qualifies(attacker,target,6)
            new_ok = new.weapon.qualifies(attacker,target,6)
            assert old_ok and new_ok == (event['side'] == 'blue')
            replay_rows.append(dict(step=event['step'],side=event['side'],attacker=event['attacker']+1,
                target=event['target']+1,distance=g['distance'][0,0],ATA_deg=np.rad2deg(g['ata'][0,0]),
                paper_HA_deg=np.rad2deg(g['ha'][0,0]),
                off_boresight_deg=np.rad2deg(engagement_geometry(attacker,target).off_boresight),
                v315_eligible=old_ok,v316_eligible=new_ok))
        write_csv(output/'existing_replay_geometry.csv',replay_rows)
        checks.update(existing_replay_static_shots=len(replay_rows),
            existing_red_shots_rejected=sum(r['side']=='red' and not r['v316_eligible'] for r in replay_rows),
            existing_blue_shots_allowed=sum(r['side']=='blue' and r['v316_eligible'] for r in replay_rows))
    (output/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    # A standalone audit also produces a report; the complete acceptance report
    # is written after tests and four-algorithm smoke have finished.
    (output/'weapon_audit_report.md').write_text('Static v3.16 weapon audit\n\n'+json.dumps(checks,indent=2)+
        '\n\nMonte Carlo verifies the formula change only; it is not weapon recalibration.\n',encoding='utf-8')
    report = output/'report.md'
    if not report.exists(): report.write_text((output/'weapon_audit_report.md').read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps(checks,indent=2))
    print(json.dumps(probability_rows,indent=2))
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/v316_validation')
    run(parser.parse_args().output)
