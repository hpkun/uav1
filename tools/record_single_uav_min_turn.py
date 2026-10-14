"""One-aircraft level-turn demonstration using the current unmodified controller."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.config import load_config,aircraft_spec
from env.models import AircraftState
from env.dynamics import PointMassDynamics
from env.integrator import RK4Integrator
from env.v329_control import action_to_control_v329


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-config',default='configs/combat_environment_v331.yaml')
    parser.add_argument('--output-dir',default='outputs/single_uav_min_turn')
    args=parser.parse_args()
    cfg=load_config(ROOT/args.env_config);spec=aircraft_spec(cfg)
    out=ROOT/args.output_dir;out.mkdir(parents=True,exist_ok=True)
    dynamics=PointMassDynamics();integrator=RK4Integrator(cfg['simulation']['dt'])
    speed=spec.v_min;limit=cfg['action']['controller']['normal_load_max'];g=dynamics.gravity
    horizontal_load=np.sqrt(limit**2-1.)
    radius=speed**2/(g*horizontal_load);period=2*np.pi*radius/speed
    # Maintain exactly one vertical G after the existing radial load projection.
    # This is a feasible high-level pitch action, not a bypassed control command.
    raw_lateral=speed*(cfg['action']['command']['heading_delta_max']/cfg['action']['controller']['heading_time_constant'])/g
    raw_vertical=raw_lateral/horizontal_load
    pitch_action=(raw_vertical-1)*g/speed*cfg['action']['controller']['pitch_time_constant']/cfg['action']['command']['pitch_delta_max']
    action=np.array([1.,pitch_action,0.]);assert np.all(np.abs(action)<=1)
    state=AircraftState(0.,-radius,-3000.,speed,0.,0.)
    states=[state.as_array()];commands=[];dt=integrator.dt
    for _ in range(int(np.ceil(3*period/dt))):
        control=action_to_control_v329(state,action,cfg['action'],spec,g)
        commands.append([control.nx,control.nz,control.phi])
        state=integrator.step(state,control,dynamics,spec);states.append(state.as_array())
    states=np.asarray(states);commands=np.asarray(commands);times=np.arange(len(states))*dt
    assert np.max(np.abs(states[:,2]+3000))<1e-6
    assert np.max(np.abs(states[:,3]-speed))<1e-8
    assert np.max(np.abs(states[:,4]))<1e-8
    assert commands[:,1].max()<=limit+1e-10
    x,y=states[:,:2].T
    cx,cy,c=np.linalg.lstsq(np.column_stack((2*x,2*y,np.ones_like(x))),x*x+y*y,rcond=None)[0]
    measured=float(np.sqrt(cx*cx+cy*cy+c))
    assert abs(measured-radius)<.01
    np.savez_compressed(out/'trajectory.npz',time_s=times,states=states,controls=commands,action=action)
    metadata=dict(environment_version=cfg['environment_version'],aircraft_count=1,dt=dt,
                  speed_m_s=speed,normal_load=limit,altitude_m=3000.,radius_theoretical_m=radius,
                  radius_measured_m=measured,diameter_m=2*radius,period_s=period,
                  duration_s=float(times[-1]),bank_angle_deg=float(np.rad2deg(commands[0,2])),
                  normalized_action=action.tolist(),altitude_max_error_m=float(np.max(np.abs(states[:,2]+3000))),
                  note='Minimum steady level turn at v_min. Scripted high-level action; no learned policy or combat. Environment mechanics unchanged.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))

    import matplotlib
    matplotlib.use('Agg')
    vendor=out/'vendor'
    if vendor.exists():sys.path.insert(0,str(vendor))
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon,Circle
    from matplotlib.animation import FFMpegWriter
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#e8edf5','axes.labelcolor':'#aab7c8','xtick.color':'#aab7c8','ytick.color':'#aab7c8'})
    fig,ax=plt.subplots(figsize=(9.6,8),dpi=100);fig.patch.set_facecolor('#101722');ax.set_facecolor('#141e2b')
    fig.subplots_adjust(left=.12,right=.95,bottom=.11,top=.79)
    bound=radius*1.3
    ax.set(xlim=(-bound,bound),ylim=(-bound,bound),aspect='equal',xlabel='X (m)',ylabel='Y (m)')
    ax.grid(color='#304055',alpha=.55)
    for spine in ax.spines.values():spine.set_color('#304055')
    ax.add_patch(Circle((0,0),radius,fill=False,edgecolor='#678099',linestyle='--',linewidth=1,alpha=.45))
    ax.plot([0,radius],[0,0],color='#678099',linewidth=1)
    ax.text(radius*.45,12,f'R = {radius:.1f} m',color='#aab7c8',ha='center',fontsize=10)
    ax.plot(0,0,'+',color='#678099',markersize=7)
    trail,=ax.plot([],[],color='#ed5267',linewidth=2,alpha=.65)
    # Stylized airframe marker, enlarged for visibility; no extra aircraft.
    shape=np.array([[24,0],[-9,7],[-3,3],[-3,14],[-8,14],[-12,3],[-18,4],[-18,-4],[-12,-3],[-8,-14],[-3,-14],[-3,-3],[-9,-7]])
    plane=Polygon(shape,closed=True,facecolor='#ff5068',edgecolor='#ffd0d8',linewidth=.7,zorder=5);ax.add_patch(plane)
    fig.text(.12,.94,'RED UAV  |  MINIMUM LEVEL TURN',fontsize=18,weight='bold')
    fig.text(.12,.895,f'{speed:.0f} m/s     {limit:.0f} G     altitude 3000 m     bank {metadata["bank_angle_deg"]:.2f} deg',fontsize=12,color='#aab7c8')
    fig.text(.12,.855,f'Diameter {2*radius:.1f} m  |  One circle {period:.2f} s  |  Real-time playback',fontsize=11,color='#aab7c8')
    status=fig.text(.12,.815,'',fontsize=11,color='#ed8796')
    heading=np.unwrap(states[:,5]);fps=24
    writer=FFMpegWriter(fps=fps,codec='libx264',bitrate=2400,extra_args=['-pix_fmt','yuv420p','-movflags','+faststart'])
    with writer.saving(fig,str(out/'minimum_turn.mp4'),dpi=100):
        for index,t in enumerate(np.arange(0,times[-1],1/fps)):
            px=np.interp(t,times,x);py=np.interp(t,times,y);psi=np.interp(t,times,heading)
            rotation=np.array([[np.cos(psi),-np.sin(psi)],[np.sin(psi),np.cos(psi)]])
            plane.set_xy(shape@rotation.T+[px,py])
            upto=np.searchsorted(times,t,side='right')
            trail.set_data(np.append(x[:upto],px),np.append(y[:upto],py))
            status.set_text(f't = {t:05.2f} s     circles = {t/period:.2f}')
            writer.grab_frame()
            if index==int(fps*period*.75):fig.savefig(out/'preview.png',dpi=100)
    plt.close(fig)
    print(json.dumps(metadata,indent=2))
    print(out/'minimum_turn.mp4')


if __name__=='__main__':main()
