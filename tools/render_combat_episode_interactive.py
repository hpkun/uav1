"""Internal trace-only HTML renderer; official CLI: replay_combat.py."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.combat_visualization import read_trace


def render(trace_path: str | Path, metadata_path: str | Path, output: str | Path,
           stride: int = 1, playback_speed: float = 1.) -> Path:
    import plotly.graph_objects as go
    if stride <= 0:
        raise ValueError('stride must be positive')
    if not np.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError('playback speed must be finite and positive')
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    trace = read_trace(trace_path,metadata)
    radius = float(metadata['arena_radius'])
    indices = sorted(set(range(0, len(trace['steps']), stride)) | {len(trace['steps'])-1})
    # Events never add frames: each sampled interval displays all its events.
    previous={frame:indices[max(0,k-1)] for k,frame in enumerate(indices)}

    def data_at(frame):
        data = []
        for side, color, label in (('red', '#df3348', 'R'), ('blue', '#2484d0', 'B')):
            alive = trace[f'{side}_alive']
            values = trace[f'{side}_kinematics']
            for agent in range(alive.shape[1]):
                deaths = np.flatnonzero(~alive[:frame+1, agent])
                last = int(deaths[0]) if len(deaths) else frame
                path = values[:last+1, agent]
                data.append(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=-path[:, 2],
                    mode='lines', line=dict(color=color, width=2), showlegend=False,
                    name=f'{label}{agent+1} trajectory', hoverinfo='skip'))
                state = values[last, agent]
                data.append(go.Scatter3d(x=[state[0]], y=[state[1]], z=[-state[2]],
                    mode='markers+text', text=[f'{label}{agent+1}'], textposition='top center',
                    name=f'{label}{agent+1}', marker=dict(color=color, size=5,
                    symbol='circle' if alive[frame, agent] else 'x'),
                    customdata=[[bool(alive[frame, agent]), *state[3:].tolist()]],
                    hovertemplate='Alive: %{customdata[0]}<br>Speed: %{customdata[1]:.1f} m/s<br>Pitch: %{customdata[2]:.3f}<br>Heading: %{customdata[3]:.3f}<extra>%{fullData.name}</extra>'))
        for hit, color in ((True, '#20c75a'), (False, '#ed9924')):
            coordinates = []
            for event in metadata.get('events', []):
                if event['type'] in ('fire_attempt','dwell_complete') and int(trace['steps'][previous[frame]]) < event['step'] <= int(trace['steps'][frame]) and event.get('hit',True) == hit:
                    coordinates.extend([event['start'], event['end'], [None, None, None]])
            data.append(go.Scatter3d(x=[p[0] for p in coordinates], y=[p[1] for p in coordinates],
                z=[None if p[2] is None else -p[2] for p in coordinates], mode='lines',
                line=dict(color=color, width=4), name=('Dwell completed' if metadata.get('event_semantics')=='continuous_dwell_neutralization' else 'Weapon hit') if hit else 'Missed attempt'))
        return data

    def title(frame):
        red = int(trace['red_alive'][frame].sum()); blue = int(trace['blue_alive'][frame].sum())
        current = [event['type'] for event in metadata.get('events', []) if int(trace['steps'][previous[frame]]) < event['step'] <= int(trace['steps'][frame])]
        ending = f" | {metadata['termination_reason']}" if frame == len(trace['steps'])-1 else ''
        return ("Downsampled preview | " if stride>1 else "Realtime replay | ")+f"Combat | step {trace['steps'][frame]} | {trace['time_s'][frame]:.1f}s | Red {red} / Blue {blue} | {', '.join(current)}{ending}"

    fig = go.Figure(data=data_at(0))
    fig.frames = [go.Frame(name=str(frame), data=data_at(frame), traces=list(range(len(fig.data))),
                           layout=go.Layout(title=title(frame))) for frame in indices]
    angle = np.linspace(0, 2*np.pi, 120)
    fig.add_trace(go.Scatter3d(x=radius*np.cos(angle), y=radius*np.sin(angle), z=np.zeros_like(angle),
                             mode='lines', line=dict(color='gray', width=2), name='Arena'))
    altitude = max(4000, max(float(-trace[f'{side}_kinematics'][:, :, 2].min()) for side in ('red', 'blue')))
    def animation(speed):
        return dict(frame=dict(duration=1000*metadata['dt']*stride/speed, redraw=True),
                    transition=dict(duration=0), fromcurrent=True)
    play_buttons = [dict(label=f'Play ({playback_speed:g}x)', method='skip', args=[None, animation(playback_speed)])]
    play_buttons.extend(dict(label=f'Play ({speed:g}x)', method='skip', args=[None, animation(speed)])
                        for speed in (.25, .5, 1., 2.) if speed != playback_speed)
    play_buttons.append(dict(label='Pause', method='skip',
        args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]))
    fig.update_layout(title=title(0), uirevision='combat-replay',
        scene=dict(xaxis=dict(title='X (m)', range=[-radius, radius]),
                   yaxis=dict(title='Y (m)', range=[-radius, radius]),
                   zaxis=dict(title='Altitude (m)', range=[0, altitude]), aspectmode='cube',
                   dragmode='orbit', uirevision='combat-camera'),
        updatemenus=[dict(type='buttons', buttons=play_buttons)],
        sliders=[dict(active=0, steps=[dict(label=f"{trace['time_s'][frame]:.1f}s", value=str(frame), method='animate',
            args=[[str(frame)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for frame in indices])])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A tiny scheduler uses actual simulation timestamps, including a partial
    # final preview interval. No extra event-frame holds and no frontend framework.
    timing={'frames':[str(i) for i in indices], 'time_s':[float(trace['time_s'][i]) for i in indices]}
    player = r"""
(function(){
 const gd=document.getElementById('{plot_id}');
 const timing=__TIMING__; const names=timing.frames, times=timing.time_s;
 let position=0,timer=null,generation=0,internal=false;
 function stop(){generation++;clearTimeout(timer);}
 async function show(k){
   position=k;
   await Plotly.animate(gd,[names[k]],{mode:'immediate',frame:{duration:0,redraw:true},transition:{duration:0}});
   internal=true;
   try{await Plotly.relayout(gd,{'sliders[0].active':k});}finally{internal=false;}
 }
 async function play(speed){
   stop(); const token=generation;
   if(position>=names.length-1)await show(0);
   if(token!==generation)return;
   const origin=performance.now()-times[position]*1000/speed;
   function next(){
     if(token!==generation||position>=names.length-1)return;
     const k=position+1,delay=origin+times[k]*1000/speed-performance.now();
     timer=setTimeout(async function(){
       if(token!==generation)return;
       await show(k);if(token===generation)next();
     },Math.max(0,delay));
   }
   next();
 }
 gd.on('plotly_buttonclicked',function(e){
   if(e.button.label==='Pause'){stop();return;}
   const match=e.button.label.match(/^Play \(([0-9.]+)x\)$/);
   if(match)play(Number(match[1]));
 });
 gd.on('plotly_sliderchange',function(e){
   if(!internal){stop();position=names.indexOf(String(e.step.value));}
 });
 gd.combatReplayTiming=timing;
})();
""".replace('__TIMING__',json.dumps(timing))
    fig.write_html(output, include_plotlyjs=True, full_html=True, auto_play=False,
                   post_script=player, config=dict(scrollZoom=True, displaylogo=False))
    return output



if __name__ == '__main__':
    raise SystemExit('Internal module. Use the official entry: python -u tools/replay_combat.py --help')
