"""Canonical trace, neutral recorder, authoritative timelines and offline outputs."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import re
import shutil
import subprocess
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.geometry import engagement_geometry
from tools.combat_visualization import RecordingCombatEnv,states_array
from tools.combat_trace import read_trace,validate_trace
from tools import record_combat_episode as recorder
from tools.combat_replay_report import export_analysis,analyze_trace
from tools.render_combat_episode import render
from tools.render_combat_episode_interactive import render as render_html


def config(version='337'):
    return yaml.safe_load(Path(f'configs/combat_environment_v{version}.yaml').read_text(encoding='utf-8'))


def exact(a,b):
    if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:exact(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


@pytest.mark.parametrize('version',['24','35','330','333','334','335','336','337'])
def test_semantic_invariance_full_episode_all_info_rng_policy_and_state(version):
    a,b=MultiUAVCombatEnv(config(version)),RecordingCombatEnv(config(version))
    exact(a.reset(9),b.reset(9));rng=np.random.default_rng(5)
    while True:
        actions=rng.uniform(-1,1,(a.team_size,3)).astype(np.float32)
        x,y=a.step(actions),b.step(actions.copy());exact(x,y)
        exact(states_array(a.red+a.blue),states_array(b.red+b.blue))
        exact(a.rng.bit_generator.state,b.rng.bit_generator.state)
        for side in ('red','blue'):
            for suffix in ('_ammo','_fire_cooldown_steps'):
                if hasattr(a,side+suffix):exact(getattr(a,side+suffix),getattr(b,side+suffix))
        if hasattr(a.fixed_policy,'rng'):exact(a.fixed_policy.rng.bit_generator.state,b.fixed_policy.rng.bit_generator.state)
        if x[2] or x[3]:break


class ZeroActor:
    def act(self,obs,alive,deterministic=True):
        assert deterministic
        return np.zeros((*obs.shape[:-1],3),np.float32)


@pytest.fixture
def captured(tmp_path,monkeypatch):
    snapshots=[];infos=[]
    class ShortRecording(RecordingCombatEnv):
        def reset(self,seed=None):
            result=super().reset(seed);self.max_steps=5
            snapshots.append(states_array(self.red+self.blue));return result
        def step(self,actions):
            result=super().step(actions);snapshots.append(states_array(self.red+self.blue));infos.append(deepcopy(result[4]));return result
    monkeypatch.setattr(recorder,'RecordingCombatEnv',ShortRecording)
    out=tmp_path/'episode';meta=recorder.record(ZeroActor(),config(),1,out,{'algorithm':'test'})
    return out,meta,read_trace(out/'episode_trace.npz',meta),snapshots,infos


def test_frame_transition_blue_action_target_and_reward_alignment(captured):
    out,m,t,states,infos=captured
    assert t['red_kinematics'].shape==(6,4,6) and t['blue_actions'].shape==(5,4,3)
    assert t['reward_components'].shape==(5,4,4)
    for k in range(6):exact(np.concatenate((t['red_kinematics'][k],t['blue_kinematics'][k])),states[k])
    for k,info in enumerate(infos):
        exact(t['red_actions'][k],info['executed_red_actions'])
        exact(t['blue_actions'][k],info['executed_blue_actions'])
        exact(t['blue_pursuit_target_indices'][k],info['blue_pursuit_target_indices'])
        exact(t['local_rewards'][k],info['r1_rewards']+info['r2_rewards']+info['r3_rewards']+info['r4_rewards'])
        for i in range(4):exact(t['reward_components'][k,i],info[f'r{i+1}_rewards'])
        for side in ('red','blue'):exact(t[side+'_ammo'][k],info[side+'_ammo'])
    assert m['blue_target_source']=='info.blue_pursuit_target_indices'
    assert m['reward_component_names']==['R1','R2','R3','R4'] and m['reward_component_axis']==1 and m['agent_axis']==2


@pytest.mark.parametrize('field',['red_kinematics','blue_kinematics','red_alive','steps','time_s','red_actions','blue_actions',
    'local_rewards','reward_components','terminated','truncated','blue_pursuit_target_indices','red_ammo','blue_ammo','red_cooldown','blue_cooldown'])
def test_reject_misaligned_field(captured,field):
    t=deepcopy(captured[2]);t[field]=t[field][:-1]
    with pytest.raises(ValueError):validate_trace(t)


@pytest.mark.parametrize('field',['red_kinematics','blue_actions','local_rewards','reward_components'])
def test_reject_nonfinite_values(captured,field):
    t=deepcopy(captured[2]);t[field]=t[field].astype(float);t[field].flat[0]=np.nan
    with pytest.raises(ValueError,match='non-finite'):validate_trace(t)


@pytest.mark.parametrize('mutation',['missing','times','steps','terminal','target','pickle'])
def test_reject_corrupt_trace(captured,mutation,tmp_path):
    t=deepcopy(captured[2])
    if mutation=='missing':del t['blue_actions']
    elif mutation=='times':t['time_s'][2]+=.1
    elif mutation=='steps':t['steps'][2]=1
    elif mutation=='terminal':t['truncated'][-1]=False
    elif mutation=='target':t['blue_pursuit_target_indices'][0,0]=4
    else:t['red_actions']=np.asarray([object()],dtype=object)
    p=tmp_path/'bad.npz';np.savez(p,**t)
    with pytest.raises(ValueError):read_trace(p)


@pytest.mark.parametrize('side',['red','blue'])
@pytest.mark.parametrize('flags,cause',[
    ((False,False,True),'ground'),((False,True,False),'ceiling'),((True,False,False),'horizontal_boundary'),
    ((True,True,False),'ceiling'),((True,False,True),'ground')])
def test_actual_noncombat_loss_priority_and_invariance(side,flags,cause):
    a,b=MultiUAVCombatEnv(config()),RecordingCombatEnv(config());a.reset(1);b.reset(1)
    horizontal,ceiling,ground=flags
    for e in (a,b):
        s=getattr(e,side)[0];s.x=6000 if horizontal else 0;s.y=0;s.z=0 if ground else -6100 if ceiling else -3000
    exact(a._resolve_noncombat_losses(),b._resolve_noncombat_losses());exact(a.combat_counts,b.combat_counts)
    exact(states_array(a.red+a.blue),states_array(b.red+b.blue))
    event=next(e for e in b.events if e['type']=='noncombat_loss')
    assert event['cause']==cause and event['horizontal_exceeded']==horizontal and event['ceiling_exceeded']==ceiling and event['ground_exceeded']==ground


def test_historical_horizontal_has_priority_over_ground():
    e=RecordingCombatEnv('configs/combat_environment.yaml');e.reset(1);e.red[0].x=6000;e.red[0].z=0
    e._resolve_noncombat_losses();event=e.events[0]
    assert event['cause']=='horizontal_boundary' and event['ground_exceeded']


def test_real_fire_identity_complete_geometry_ammo_and_cooldown(monkeypatch):
    from env.combat_v336 import CombatEnvironmentV336
    original=CombatEnvironmentV336._entry_attempts;observed=[]
    def spy(self,attackers,targets,states,side):
        result=original(self,attackers,targets,states,side);observed.extend((side,*row) for row in result);return result
    monkeypatch.setattr(CombatEnvironmentV336,'_entry_attempts',spy)
    e=RecordingCombatEnv(config());e.reset(1)
    e.red=[AircraftState(0,20*i,-3000,225,0,0) for i in range(4)]
    e.blue=[AircraftState(600,20*i,-3000,225,0,np.pi) for i in range(4)]
    before=[e.red_ammo.copy(),e.blue_ammo.copy()]
    e.step(np.zeros((4,3)),np.zeros((4,3)))
    events=[x for x in e.events if x['type']=='fire_attempt']
    assert [(x['side'],x['attacker'],x['target'],x['hit']) for x in events]==observed
    for x in events:
        g=engagement_geometry(AircraftState(*x['attacker_state']),AircraftState(*x['target_state']))
        for key,value in asdict(g).items():assert x[key]==value
        assert x['range_pass'] and x['ata_pass'] and x['ha_pass']
    for side,start in zip(('red','blue'),before):
        shots=[x for x in events if x['side']==side]
        assert int(start.sum()-getattr(e,side+'_ammo').sum())==len(shots)
        assert all(getattr(e,side+'_fire_cooldown_steps')[x['attacker']]==10 for x in shots)


def test_report_and_csv_derive_only_from_same_trace(captured,monkeypatch):
    out,m,t,*_=captured
    monkeypatch.setattr(MultiUAVCombatEnv,'step',lambda *a,**k:pytest.fail('offline exporter must not roll out'))
    a=export_analysis(out/'episode_trace.npz',out/'metadata.json',out)
    text=(out/'detailed_report.md').read_text(encoding='utf-8')
    assert 'Reward-R1 total' in text and 'Red UAV R1' in text
    assert f"| red_survivors | {m['red_survivors']} |" in text and f"| episode_return | {m['episode_return']} |" in text
    rows=list(csv.DictReader((out/'all_uav_states.csv').open(encoding='utf-8-sig')))
    assert len(rows)==6*8
    assert all(float(r['altitude'])==-float(r['z_ned']) for r in rows)
    assert a['reward_sum_matches_local']
    assert a['combat']['red']['fire_attempts']==m['red_fire_attempts']
    assert '[All UAV states](all_uav_states.csv)' in text
    assert '[MP4](episode.mp4)' not in text
    assert '![Trajectory]' not in text


def test_reward_component_axis_with_eight_agents(tmp_path):
    c=config('24');c['simulation']['max_steps']=3;out=tmp_path/'eight'
    m=recorder.record(ZeroActor(),c,1,out);t=read_trace(out/'episode_trace.npz')
    assert t['reward_components'].shape==(3,4,8)
    bad=deepcopy(t);bad['reward_components']=bad['reward_components'].transpose(0,2,1)
    with pytest.raises(ValueError):validate_trace(bad)
    a=analyze_trace(t,m)
    for row in a['uavs']:
        if row['side']=='red':exact(np.asarray(row['reward_breakdown']),t['reward_components'][:,:,row['agent']].sum(axis=0,dtype=np.float64))


def test_html_regular_sampling_events_intervals_actual_schedule(captured,tmp_path,monkeypatch):
    import plotly.graph_objects as go
    out,m,t,*_=captured
    # Event at step 1 lies between sampled frames 0 and 4; no extra event frame.
    m['events']=[dict(type='fire_attempt',step=1,start=[0,0,-3000],end=[100,0,-3000],hit=True)]
    (out/'metadata.json').write_text(json.dumps(m),encoding='utf-8')
    figures=[];original=go.Figure.write_html
    def capture(fig,*a,**kw):figures.append(fig);return original(fig,*a,**kw)
    monkeypatch.setattr(go.Figure,'write_html',capture)
    p=render_html(out/'episode_trace.npz',out/'metadata.json',tmp_path/'preview.html',stride=4)
    f=figures[0];assert [x.name for x in f.frames]==['0','4','5']
    assert len(f.frames[1].data[-2].x)==3
    html=p.read_text(encoding='utf-8')
    timing=json.loads(re.search(r'const timing=(.*?); const names',html).group(1))
    assert timing==dict(frames=['0','4','5'],time_s=[0.,.4,.5])
    assert sum(np.diff(timing['time_s']))==.5
    assert all(x in html for x in ('Downsampled preview','performance.now()','setTimeout','Pause','Altitude (m)','R4'))


def test_mp4_realtime_duration_and_png(captured,tmp_path):
    out,m,*_=captured
    image=render(out/'episode_trace.npz',out/'metadata.json',tmp_path/'trajectory.png')
    assert image.stat().st_size>10000
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):pytest.skip('FFmpeg not available')
    movie=render(out/'episode_trace.npz',out/'metadata.json',tmp_path/'episode.mp4')
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(movie)],text=True))
    assert abs(float(data['format']['duration'])-m['episode_steps']*m['dt'])<=m['dt']+.01
    assert data['streams'][0]['r_frame_rate']=='10/1'


def test_legacy_analyzer_refuses_current_geometry():
    from tools.create_combat_v24_videos import analyze_episode
    with pytest.raises(ValueError,match='Legacy v2.4'):analyze_episode({}, {},config())


@pytest.mark.parametrize('name',['MAPPO','MAAC','MADSAC','RMAPPO','EA-MAPPO','STEA-MAPPO'])
def test_all_six_official_loader_dispatch_validator_factory_and_load(name,monkeypatch,tmp_path):
    import importlib
    import torch
    from algorithm.common.protocol import config_sha256
    calls=[];ec={'environment_version':'fixture'};ac={'fixture':'algorithm'}
    extra=dict(training_seed=17,training_smoke=False,environment_config_sha256=config_sha256(ec),
        algorithm_config_sha256=config_sha256(ac),network_architecture={'hidden_dim':16},observation_dim=45,action_dim=3,num_agents=4)
    if name=='MAAC':extra.pop('training_seed')
    state=dict(algorithm=name,training_seed=17,sampled_steps=100,extra=extra)
    checkpoint=tmp_path/'unit_dispatch.pt';checkpoint.write_bytes(b'unit-test mocked serialized input')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch,'load',lambda *a,**k:state)
    trainer=SimpleNamespace(load=lambda *a:calls.append('load'),load_for_evaluation=lambda *a:calls.append('load'))
    validator=lambda *a,**k:calls.append('validator')
    factory=lambda *a,**k:(calls.append('factory') or trainer)
    if name=='MAPPO':
        monkeypatch.setattr(recorder,'validate_checkpoint_for_evaluation',validator)
        monkeypatch.setattr(recorder,'build_mappo_trainer',factory)
        import algorithm.common.critic_protocol as protocol
        monkeypatch.setattr(protocol,'checkpoint_widths',lambda *a:(16,16))
    elif name=='MAAC':
        import algorithm.maac.protocol as protocol
        import algorithm.maac.trainer as trainers
        monkeypatch.setattr(protocol,'validate_checkpoint',validator);monkeypatch.setattr(trainers,'MAACTrainer',factory)
    elif name=='MADSAC':
        monkeypatch.setattr(recorder,'validate_madsac_checkpoint',validator);monkeypatch.setattr(recorder,'build_madsac_trainer',factory)
    else:
        slug=name.lower().replace('-','_');p=importlib.import_module('algorithm.'+slug+'.protocol');f=importlib.import_module('algorithm.'+slug+'.factory')
        monkeypatch.setattr(p,'validate_checkpoint',validator);monkeypatch.setattr(f,'build_'+slug+'_trainer',factory)
    loaded,_,metadata=recorder.load_recording_policy(checkpoint,ec,ac,'cuda')
    assert loaded is trainer and calls==['validator','factory','load']
    assert metadata['training_seed']==17 and metadata['algorithm']==name


def test_entry_one_rollout_and_automatic_training_config(captured,monkeypatch,tmp_path):
    from tools import replay_combat as entry
    out,m,t,*_=captured;run=tmp_path/'training';run.mkdir()
    (run/'algorithm_config.yaml').write_text('fixture: saved\n')
    envfile=run/'env.yaml';envfile.write_text('fixture: environment\n');checkpoint=run/'best.pt';checkpoint.write_bytes(b'test')
    calls=[];actor=object()
    def load(cp,ec,ac,device):
        assert cp==checkpoint and ac=={'fixture':'saved'} and device=='cuda'
        return actor,{},{}
    def record_once(*args):calls.append('rollout');return m
    monkeypatch.setattr(entry,'load_recording_policy',load);monkeypatch.setattr(entry,'record',record_once)
    monkeypatch.setattr(entry,'export_analysis',lambda *a:calls.append('offline'))
    entry.replay(checkpoint,envfile,1,out,formats=('trace',))
    assert calls==['rollout','offline']


@pytest.mark.parametrize('field',['red_survivors','episode_return','red_fire_attempts','episode_r3_total'])
def test_report_refuses_inconsistent_metadata(captured,field):
    _,m,t,*_=captured;m=deepcopy(m);m[field]+=1
    with pytest.raises(ValueError):analyze_trace(t,m)


def test_missing_legacy_target_is_marked_unavailable_not_reconstructed(tmp_path):
    c=config('24');c['simulation']['max_steps']=3
    m=recorder.record(ZeroActor(),c,1,tmp_path/'legacy');t=read_trace(tmp_path/'legacy/episode_trace.npz')
    assert not m['blue_target_available'] and (t['blue_pursuit_target_indices']==-1).all()
    assert analyze_trace(t,m)['blue_target_frequencies'] is None


@pytest.mark.parametrize('field,value',[('reward_component_names',['R2','R1','R3','R4']),('reward_component_axis',2),('agent_axis',1)])
def test_reject_ambiguous_component_metadata(captured,field,value):
    _,m,t,*_=captured;m=deepcopy(m);m[field]=value
    with pytest.raises(ValueError,match='axes/order'):validate_trace(t,m)


def test_schema_two_remains_renderable_without_inventing_timelines(captured,tmp_path):
    _,m,t,*_=captured;old={key:t[key] for key in ('red_kinematics','blue_kinematics','red_alive','blue_alive','steps','time_s','red_actions','local_rewards','reward_components','terminated','truncated')}
    old['trace_schema_version']=np.asarray(2);p=tmp_path/'old.npz';np.savez(p,**old)
    read=read_trace(p,m);assert 'blue_pursuit_target_indices' not in read
    with pytest.raises(ValueError,match='render-only'):export_analysis(p,captured[0]/'metadata.json',tmp_path)


def test_missing_ffmpeg_is_a_clear_error_not_a_pillow_mp4(tmp_path):
    from tools.combat_video import configure_ffmpeg
    with pytest.raises(RuntimeError,match='FFmpeg required'):configure_ffmpeg(tmp_path/'absent-ffmpeg')
