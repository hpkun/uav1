"""Deterministic body-weapon and attacker-window audit; no model/training."""
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
    def __init__(self, epsilon=2.):
        self.epsilon, self.calls = epsilon, 0

    def normal(self, mean, std):
        assert (mean, std) == (0., 1.)
        self.calls += 1
        return self.epsilon


def body_pair(pitch=0., azimuth=0., elevation=0., distance=3000., heading=0.):
    """Place a fixture LOS in the known body basis; runtime uses engagement_geometry."""
    attacker = AircraftState(0., 0., -3000., 225., math.radians(pitch), math.radians(heading))
    forward = attacker.velocity_vector() / attacker.v
    right = np.array([-math.sin(attacker.psi), math.cos(attacker.psi), 0.])
    up = np.cross(right, forward)
    az, el = math.radians(azimuth), math.radians(elevation)
    los = math.cos(el)*math.cos(az)*forward + math.cos(el)*math.sin(az)*right + math.sin(el)*up
    point = attacker.as_array()[:3] + distance*los
    target = AircraftState(*point, 225., attacker.theta, attacker.psi)
    return attacker, target


def geometry_cases():
    cases = [('pitch45_aligned', *body_pair(pitch=45), True),
             ('horizontal_aligned', *body_pair(), True),
             ('body_az31', *body_pair(azimuth=31), False),
             ('body_el31', *body_pair(elevation=31), False),
             ('body_az25_el25_rectangle', *body_pair(azimuth=25,elevation=25), True),
             ('negative_body_az25_el25', *body_pair(azimuth=-25,elevation=-25), True),
             ('az_boundary', *body_pair(azimuth=30), True),
             ('el_boundary', *body_pair(elevation=-30), True),
             ('range_boundary', *body_pair(distance=4000), True),
             ('range_outside', *body_pair(distance=4000.000001), False)]
    attacker = AircraftState(0,0,-3000,225,math.pi/3,0)
    target = AircraftState(3990*math.cos(math.radians(22)),0,-3000+3990*math.sin(math.radians(22)),225,0,0)
    cases.append(('pitch60_LOS_minus22',attacker,target,False))
    return cases


def window_env(version='318'):
    env = MultiUAVCombatEnv(ROOT/f'configs/combat_environment_v{version}.yaml')
    env.reset(31)
    own, target = body_pair()
    env.red = [own] + [AircraftState(0,0,-3000,225,0,0,False) for _ in range(4)]
    env.blue = [target, AircraftState(3500,0,-3000,225,0,0)] + [AircraftState(0,0,-3000,225,0,0,False) for _ in range(3)]
    # Blue faces away from Red: only one Red attacker can attempt; epsilon=2 misses.
    env.rng = FixedNormal(2.)
    return env


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(output):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for repository audits; no CPU fallback')
    output.mkdir(parents=True,exist_ok=True)
    envs={v:MultiUAVCombatEnv(ROOT/f'configs/combat_environment_v{v}.yaml') for v in ('316','317','318')}
    geometry=[]
    for name,a,b,expected in geometry_cases():
        g=engagement_geometry(a,b);p=combat_geometry_v313([a],[b])
        old=envs['316'].weapon.qualifies(a,b,6);new=envs['317'].weapon.qualifies(a,b,6)
        assert new==expected and envs['318'].weapon.qualifies(a,b,6)==new
        if name=='pitch45_aligned':assert not old and new
        if name=='body_az25_el25_rectangle':assert g.off_boresight>math.pi/6 and new
        geometry.append(dict(case=name,distance=g.distance,paper_ATA_deg=np.rad2deg(p['ata'][0,0]),
            paper_HA_deg=np.rad2deg(p['ha'][0,0]),body_azimuth_deg=np.rad2deg(g.boresight_azimuth_error),
            body_elevation_deg=np.rad2deg(g.boresight_elevation_error),off_boresight_deg=np.rad2deg(g.off_boresight),
            v316_eligible=old,v317_eligible=new,v318_eligible=new))
    cadence=[]
    for version in ('317','318'):
        e=window_env(version)
        for _ in range(20):
            _,reward,term,trunc,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
            assert not term and not trunc and np.isfinite(reward).all()
            cadence.append(dict(version=version,step=e.steps,attempts=info['red_fire_attempts'],
                                ammo_used=6-int(e.red_ammo[0]),rng_calls=e.rng.calls,
                                armed=e.red_fire_states[0].armed if version=='318' else 'repeat_fire'))
        expected=1 if version=='318' else 6
        assert info['red_fire_attempts']==expected and e.rng.calls==expected and e.red_ammo[0]==6-expected
    e=window_env('318')
    fire=lambda:e._entry_attempts(e.red,e.blue,e.red_fire_states,'red')
    assert len(fire())==1
    e.blue[0].alive=False
    assert fire()==[] and e.red_ammo[0]==5 and e.rng.calls==1
    e.blue[1].x=5000
    assert fire()==[] and e.red_fire_states[0].armed
    e.blue[1].x=3500
    assert len(fire())==1 and e.red_ammo[0]==4 and e.rng.calls==2
    checks=dict(cuda_device=torch.cuda.get_device_name(),model_loaded=False,formal_training_started=False,
        pitch45_aligned_v316_rejected=True,pitch45_aligned_v317_allowed=True,pitch60_LOS_minus22_rejected=True,
        rectangular_window_not_conical=True,continuous_20_steps_v318_attempts=1,
        continuous_20_steps_v318_ammo_used=1,continuous_20_steps_v318_rng_calls=1,
        target_switch_does_not_rearm=True,empty_window_rearms=True,reentry_fires_again=True)
    write_csv(output/'geometry_cases.csv',geometry);write_csv(output/'cadence_20_steps.csv',cadence)
    (output/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))
    return checks


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/v317_v318_validation')
    run(p.parse_args().output)
