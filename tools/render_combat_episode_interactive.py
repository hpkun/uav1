"""Generate a standalone, offline Plotly 3D combat replay."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.combat_visualization import (annotate_death_attack_sources, attack_totals,
    extract_death_frames, extract_events, infer_method_display_name, read_trace)
from env.geometry import engagement_geometry
from env.models import AircraftState

def _safe(v):
    if isinstance(v, np.ndarray): return [_safe(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)): return [_safe(x) for x in v]
    if isinstance(v, (float, np.floating)) and not math.isfinite(float(v)): return None
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.bool_): return bool(v)
    return v

def _ranges(trace, radius):
    alt=[]
    for key in ("red_kinematics", "blue_kinematics"):
        z=-np.asarray(trace[key])[...,2]; alt.extend(z[np.isfinite(z)].tolist())
    maximum=max(0.0, max(alt, default=0.0)); return [[-radius,radius],[-radius,radius],[0.0,math.ceil(maximum+max(200.0,maximum*.05))]]

def _derive_attack_links(trace, meta):
    """Reconstruct exact entry-triggered fire pairs from schema-v1 states."""
    required=("red_step_fire_attempts","blue_step_fire_attempts")
    if not all(key in trace for key in required): return []
    variant=str(meta.get("environment_variant", ""))
    configs={"persistent_wave_v2":"persistent_wave_v2_environment.yaml","persistent_wave_v1":"persistent_wave_environment.yaml"}
    if variant not in configs: return []
    config=yaml.safe_load((ROOT/"configs"/configs[variant]).read_text(encoding="utf-8"));weapon=config["weapon"];radius=float(config["arena"]["radius"])
    armed={"red":[True]*4,"blue":[True]*4};links=[]
    def state(row, alive):
        if not alive or not np.all(np.isfinite(row)): return None
        return AircraftState(*map(float,row),alive=True)
    def post_noncombat(row, alive):
        if not alive or not np.all(np.isfinite(row[:3])): return False
        return bool(np.hypot(row[0],row[1])<=radius and -row[2]>0.0)
    def select(attacker, targets):
        candidates=[]
        for index,target in enumerate(targets):
            if attacker is None or target is None: continue
            geometry=engagement_geometry(attacker,target)
            if float(weapon["range_min"])<=geometry.distance<=float(weapon["range_max"]) and geometry.off_boresight<=float(weapon["off_boresight_angle_max"]): candidates.append((geometry.distance,index))
        return min(candidates)[1] if candidates else None
    transitions=len(trace["steps"])-1
    for step in range(transitions):
        frame=step+1;wave=max(0,min(trace["blue_kinematics"].shape[1]-1,int(trace["active_wave"][step])-1))
        red_rows=trace["red_kinematics"][frame];blue_rows=trace["blue_kinematics"][frame,wave]
        red_states=[state(red_rows[a],post_noncombat(red_rows[a],bool(trace["red_alive"][step,a]))) for a in range(4)]
        blue_states=[state(blue_rows[a],post_noncombat(blue_rows[a],bool(trace["blue_alive"][step,wave,a]))) for a in range(4)]
        derived={"red":0,"blue":0}
        for side,attackers,targets in (("red",red_states,blue_states),("blue",blue_states,red_states)):
            for attacker_index,attacker in enumerate(attackers):
                target_index=select(attacker,targets)
                if target_index is None: armed[side][attacker_index]=True;continue
                if not armed[side][attacker_index]: continue
                armed[side][attacker_index]=False;derived[side]+=1
                attack_row=red_rows[attacker_index] if side=="red" else blue_rows[attacker_index]
                target_row=blue_rows[target_index] if side=="red" else red_rows[target_index]
                links.append({"frame":frame,"side":side,"attacker":attacker_index+1,"target":target_index+1,"wave":wave+1,"attacker_position":[float(attack_row[0]),float(attack_row[1]),float(attack_row[2])],"target_position":[float(target_row[0]),float(target_row[1]),float(target_row[2])]})
        for side in ("red","blue"):
            recorded=int(trace[f"{side}_step_fire_attempts"][step])
            if derived[side]!=recorded: raise ValueError(f"cannot reconstruct {side} attack pairs at transition {step}: derived={derived[side]} recorded={recorded}")
        spawned=trace.get("spawned_next_wave")
        if spawned is not None and bool(spawned[step]): armed={"red":[True]*4,"blue":[True]*4}
    return links

APP_JS = r"""
const P=JSON.parse(document.getElementById('payload').textContent),plot=document.getElementById('plot'),F=P.time.length;
const RED='#d62728',BLUE='#1f77b4',DEFAULT_CAMERA={eye:{x:1.5,y:-1.7,z:1.1},up:{x:0,y:0,z:1}},traces=[];
let playing=false,logicalFrame=0,renderedFrame=-1,playAnchorFrame=0,playAnchorWallTime=0,speed=1;
let cameraInteracting=false,pointerActive=false,wheelActive=false,renderBusy=false,renderPending=false,wheelEndTimer=null,renderEpoch=0;
const redTraj=[],blueTraj=[]; for(let a=0;a<4;a++){redTraj.push(traces.length);traces.push({type:'scatter3d',mode:'lines',name:`Red R${a+1}`,x:[],y:[],z:[],line:{color:RED,width:3}})}
for(let w=0;w<P.blue[0].length;w++)for(let a=0;a<4;a++){blueTraj.push(traces.length);traces.push({type:'scatter3d',mode:'lines',name:`Blue W${w+1} B${a+1}`,x:[],y:[],z:[],line:{color:BLUE,width:2,dash:['solid','dash','dot'][w%3]}})}
const redCurrent=traces.length;traces.push({type:'scatter3d',mode:'markers+text',name:'Current Red',x:[],y:[],z:[],text:[],customdata:[],marker:{size:6,color:RED},hovertemplate:'%{customdata[0]}<br>Side: Red<br>X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Altitude: %{z:.1f} m<br>Speed: %{customdata[1]:.1f} m/s<br>Heading: %{customdata[2]:.1f} deg<br>Pitch: %{customdata[3]:.1f} deg<br>Alive: Yes<extra></extra>'});
const blueCurrent=traces.length;traces.push({type:'scatter3d',mode:'markers+text',name:'Current Blue',x:[],y:[],z:[],text:[],customdata:[],marker:{size:6,color:BLUE},hovertemplate:'%{customdata[0]}<br>Side: Blue<br>Wave: %{customdata[4]}<br>X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Altitude: %{z:.1f} m<br>Speed: %{customdata[1]:.1f} m/s<br>Heading: %{customdata[2]:.1f} deg<br>Pitch: %{customdata[3]:.1f} deg<br>Alive: Yes<extra></extra>'});
const redHead=traces.length;traces.push({type:'scatter3d',mode:'lines',name:'Red headings',x:[],y:[],z:[],line:{color:RED,width:3},showlegend:false});const blueHead=traces.length;traces.push({type:'scatter3d',mode:'lines',name:'Blue headings',x:[],y:[],z:[],line:{color:BLUE,width:3},showlegend:false});
const attack=traces.length;traces.push({type:'scatter3d',mode:'lines',name:'Fire Attempt',x:[],y:[],z:[],line:{color:'#20c75a',width:6},hoverinfo:'skip',showlegend:true});
const death=traces.length;traces.push({type:'scatter3d',mode:'markers',name:'Deaths',x:[],y:[],z:[],customdata:[],marker:{symbol:'x',size:8,color:[]},hovertemplate:'%{customdata[0]}<br>%{customdata[1]}<br>Cause: %{customdata[3]}<br>Attack source evidence: %{customdata[4]}<br>Time: %{customdata[2]:.1f} s<br>X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Altitude: %{z:.1f} m<extra></extra>'});
const ang=Array.from({length:100},(_,i)=>2*Math.PI*i/99),arenaTrace=traces.length;traces.push({type:'scatter3d',mode:'lines',name:'Arena',x:ang.map(t=>P.metadata.radius*Math.cos(t)),y:ang.map(t=>P.metadata.radius*Math.sin(t)),z:ang.map(()=>0),line:{color:'#999',width:1}});
const valid=s=>s&&s[0]!==null, deg=x=>x*180/Math.PI, data=(s,label,w)=>[label,s[3],deg(s[5]),deg(s[4]),w||''];
const path=(d,a,b)=>{const q=d.slice(a,b+1).filter(valid);return{x:q.map(s=>s[0]),y:q.map(s=>s[1]),z:q.map(s=>-s[2])}};
const heads=ss=>{const x=[],y=[],z=[];ss.forEach(s=>{if(valid(s)){const L=250,c=Math.cos(s[4]);x.push(s[0],s[0]+L*c*Math.cos(s[5]),null);y.push(s[1],s[1]+L*c*Math.sin(s[5]),null);z.push(-s[2],-s[2]+L*Math.sin(s[4]),null)}});return{x,y,z}};
const eventText=e=>{const i=Math.max(0,Math.min(F-1,Number(e.trace_frame)||0)),p=e.type==='wave_cleared'||e.type==='wave_spawned'?`WAVE ${e.wave||''}`:String(e.type||'').replaceAll('_',' ').toUpperCase();return `${Number(P.time[i]).toFixed(1)} s  ${p}${e.count?` +${e.count}`:''}`};
const result=()=>{const t=String(P.metadata.termination).toLowerCase();if(P.metadata.success||P.waves_cleared[F-1]===P.metadata.total_waves)return'MISSION SUCCESS';if(t.includes('timeout'))return'TIMEOUT';if(t.includes('draw'))return'DRAW';return'RED FAILURE'};
function updatePanels(f,w,now,vd){document.getElementById('slider').value=f;document.getElementById('status').textContent=`${P.metadata.method} | Checkpoint ${P.metadata.checkpoint} | Training seed ${P.metadata.training_seed} | Episode seed ${P.metadata.episode_seed}\nSim time ${now.toFixed(1)} s | Step ${P.steps[f]} | Wave ${P.active_wave[f]} | Waves cleared ${P.waves_cleared[f]} | Red alive ${P.red_alive[f].filter(Boolean).length}/4 | Blue alive ${P.blue_alive[f][w].filter(Boolean).length}/4 | Current Total Blue Losses ${vd.filter(d=>d.side==='blue').length}`;document.getElementById('events').textContent='Recent Events\n'+P.events.filter(e=>e.trace_frame<=f).slice(-5).map(eventText).join('\n');const panel=document.getElementById('result');panel.hidden=f!==F-1;if(!panel.hidden)panel.textContent=`Result: ${result()}\nWaves Cleared ${P.waves_cleared[f]}/${P.metadata.total_waves}\nRed Survivors ${P.red_alive[f].filter(Boolean).length}/4\nTotal Blue Losses ${vd.filter(d=>d.side==='blue').length}\nEpisode Return ${Number(P.metadata.return).toFixed(2)}\nSimulation Time ${now.toFixed(1)} s`}
function updatePanelsForFrame(f){const w=Math.max(0,Math.min(P.blue[0].length-1,Number(P.active_wave[f])-1)),now=Number(P.time[f]),vd=P.deaths.filter(d=>d.frame<=f);updatePanels(f,w,now,vd)}
const renderAborted=epoch=>cameraInteracting||epoch!==renderEpoch;
async function renderFrame(f,epoch){if(renderAborted(epoch))return false;const w=Math.max(0,Math.min(P.blue[0].length-1,Number(P.active_wave[f])-1)),now=Number(P.time[f]),choice=document.getElementById('trail').value;let start=0;if(choice!=='full'){while(start<f&&Number(P.time[start])<now-Number(choice))start++}
  const redPaths=redTraj.map((_,a)=>path(P.red.map(r=>r[a]),start,f));
  const bluePaths=blueTraj.map((_,j)=>path(P.blue.map(r=>r[Math.floor(j/4)][j%4]),start,f));
  await Plotly.restyle(plot,{x:redPaths.map(q=>q.x),y:redPaths.map(q=>q.y),z:redPaths.map(q=>q.z),visible:redPaths.map(q=>q.x.length>0)},redTraj);if(renderAborted(epoch))return false;
  await Plotly.restyle(plot,{x:bluePaths.map(q=>q.x),y:bluePaths.map(q=>q.y),z:bluePaths.map(q=>q.z),visible:bluePaths.map(q=>q.x.length>0)},blueTraj);if(renderAborted(epoch))return false;
  const rs=[],bs=[],rx=[],ry=[],rz=[],rt=[],rc=[],bx=[],by=[],bz=[],bt=[],bc=[];for(let a=0;a<4;a++){const s=P.red[f][a];if(P.red_alive[f][a]&&valid(s)){rs.push(s);rx.push(s[0]);ry.push(s[1]);rz.push(-s[2]);rt.push(`R${a+1}`);rc.push(data(s,`R${a+1}`))}}for(let a=0;a<4;a++){const s=P.blue[f][w][a];if(P.blue_alive[f][w][a]&&valid(s)){bs.push(s);bx.push(s[0]);by.push(s[1]);bz.push(-s[2]);bt.push(`B${a+1}`);bc.push(data(s,`B${a+1}`,w+1))}}
  const labels=document.getElementById('labels').checked;await Plotly.restyle(plot,{x:[rx,bx],y:[ry,by],z:[rz,bz],text:[labels?rt:[],labels?bt:[]],customdata:[rc,bc],visible:[rx.length>0,bx.length>0]},[redCurrent,blueCurrent]);if(renderAborted(epoch))return false;
  const rh=heads(rs),bh=heads(bs),show=document.getElementById('heads').checked;await Plotly.restyle(plot,{x:[rh.x,bh.x],y:[rh.y,bh.y],z:[rh.z,bh.z],visible:[show&&rh.x.length>0,show&&bh.x.length>0]},[redHead,blueHead]);if(renderAborted(epoch))return false;
  const activeAttacks=P.attack_links.filter(a=>a.frame<=f&&now-Number(P.time[a.frame])<=0.6),ax=[],ay=[],az=[];activeAttacks.forEach(a=>{ax.push(a.attacker_position[0],a.target_position[0],null);ay.push(a.attacker_position[1],a.target_position[1],null);az.push(-a.attacker_position[2],-a.target_position[2],null)});await Plotly.restyle(plot,{x:[ax],y:[ay],z:[az],visible:activeAttacks.length>0},[attack]);if(renderAborted(epoch))return false;
  const causeText={attack_kill:'Weapon attack',ground_impact:'Ground impact',boundary_exit:'Boundary exit',unknown:'Unknown'},vd=P.deaths.filter(d=>d.frame<=f),dc=vd.map(d=>d.side==='red'?RED:BLUE),showDeaths=document.getElementById('markers').checked;await Plotly.restyle(plot,{x:[vd.map(d=>d.position[0])],y:[vd.map(d=>d.position[1])],z:[vd.map(d=>-d.position[2])],customdata:[vd.map(d=>[d.side==='red'?`R${d.agent} death`:`Wave ${d.wave} B${d.agent} death`,d.wave?`Wave ${d.wave}`:'Red',P.time[Math.min(F-1,d.frame)],causeText[d.cause]||d.cause,d.attack_source_evidence||'Unknown'])],'marker.color':[dc],visible:showDeaths&&vd.length>0},[death]);if(renderAborted(epoch))return false;updatePanels(f,w,now,vd);return true
}
async function drainRenderQueue(){if(renderBusy||cameraInteracting)return;renderBusy=true;try{while(renderPending&&!cameraInteracting){renderPending=false;const target=logicalFrame,epoch=renderEpoch,completed=await renderFrame(target,epoch);if(completed)renderedFrame=target;else{renderPending=true;break}}}catch(err){console.error('Combat frame render failed',err)}finally{renderBusy=false;if(renderPending&&!cameraInteracting)void drainRenderQueue()}}
function requestRender(){renderPending=true;if(cameraInteracting||renderBusy)return;void drainRenderQueue()}
function beginCameraInteraction(){if(cameraInteracting)return;cameraInteracting=true;renderEpoch+=1}
function endCameraInteraction(){if(pointerActive||wheelActive||!cameraInteracting)return;cameraInteracting=false;if(renderPending||logicalFrame!==renderedFrame)requestRender()}
function syncLogicalFrame(now=performance.now()){if(!playing)return logicalFrame;const elapsed=Math.max(0,(now-playAnchorWallTime)/1000),steps=Math.floor(elapsed*Number(P.metadata.fps)*speed);logicalFrame=Math.min(F-1,playAnchorFrame+steps*Number(P.metadata.stride));return logicalFrame}
function anchorPlayback(now=performance.now()){playAnchorFrame=logicalFrame;playAnchorWallTime=now}
function pausePlayback(){if(playing)syncLogicalFrame();playing=false;document.getElementById('play').textContent='Play';requestRender()}
function startPlayback(){if(logicalFrame>=F-1)logicalFrame=0;speed=Number(document.getElementById('speed').value);playing=true;anchorPlayback();document.getElementById('play').textContent='Pause';requestRender()}
function seek(frame){pausePlayback();logicalFrame=Math.max(0,Math.min(F-1,Number(frame)));requestRender()}
function playbackLoop(now){if(playing){const previous=logicalFrame;syncLogicalFrame(now);if(logicalFrame!==previous){if(cameraInteracting)updatePanelsForFrame(logicalFrame);requestRender()}if(logicalFrame>=F-1){playing=false;document.getElementById('play').textContent='Play';requestRender()}}requestAnimationFrame(playbackLoop)}
async function init(){try{if(typeof Plotly==='undefined')throw Error('Plotly failed to initialize.');await Plotly.newPlot(plot,traces,{title:P.metadata.title,uirevision:'combat-replay',scene:{aspectmode:'cube',dragmode:'orbit',xaxis:{title:'X / m',range:P.metadata.ranges[0],autorange:false},yaxis:{title:'Y / m',range:P.metadata.ranges[1],autorange:false},zaxis:{title:'Altitude / m',range:P.metadata.ranges[2],autorange:false}},margin:{l:0,r:0,t:40,b:0}},{responsive:true,displaylogo:false,scrollZoom:true});requestRender();document.getElementById('loading').hidden=true;
  document.getElementById('play').onclick=()=>playing?pausePlayback():startPlayback();document.getElementById('slider').oninput=e=>seek(e.target.value);document.getElementById('prev').onclick=()=>seek(logicalFrame-1);document.getElementById('next').onclick=()=>seek(logicalFrame+1);document.getElementById('restart').onclick=()=>seek(0);
  document.getElementById('speed').onchange=e=>{const now=performance.now();if(playing)syncLogicalFrame(now);speed=Number(e.target.value);if(playing)anchorPlayback(now)};['trail','heads','labels','markers'].forEach(id=>document.getElementById(id).onchange=requestRender);document.getElementById('camera').onclick=()=>Plotly.relayout(plot,{'scene.camera':DEFAULT_CAMERA});
  plot.addEventListener('pointerdown',()=>{pointerActive=true;beginCameraInteraction()},true);window.addEventListener('pointerup',()=>{pointerActive=false;endCameraInteraction()},true);window.addEventListener('pointercancel',()=>{pointerActive=false;endCameraInteraction()},true);plot.addEventListener('wheel',()=>{wheelActive=true;beginCameraInteraction();if(wheelEndTimer!==null)clearTimeout(wheelEndTimer);wheelEndTimer=setTimeout(()=>{wheelEndTimer=null;wheelActive=false;endCameraInteraction()},200)}, {capture:true,passive:true});requestAnimationFrame(playbackLoop)
}catch(err){console.error(err);document.getElementById('loading').hidden=true;const p=document.getElementById('error-panel');p.hidden=false;p.textContent=`Interactive replay initialization failed\n${err.message}\nCheck browser WebGL support.`}}
init();
"""

def render(trace_path, metadata_path, output, playback_fps=20, frame_stride=1, initial_trail="full", title="Combat Episode Replay"):
    trace=read_trace(Path(trace_path)); meta=json.loads(Path(metadata_path).read_text(encoding="utf-8"));
    if int(meta.get("trace_schema_version",1))!=1: raise ValueError("unsupported trace schema")
    radius=float(meta.get("arena_radius",5000)); trace_with_radius={**trace,"arena_radius":np.asarray(radius)}; attack_links=_derive_attack_links(trace,meta); deaths=annotate_death_attack_sources(extract_death_frames(trace_with_radius),attack_links); payload={"red":_safe(trace["red_kinematics"]),"red_alive":_safe(trace["red_alive"]),"blue":_safe(trace["blue_kinematics"]),"blue_alive":_safe(trace["blue_alive"]),"steps":_safe(trace["steps"]),"time":_safe(trace["time_s"]),"active_wave":_safe(trace["active_wave"]),"waves_cleared":_safe(trace["waves_cleared"]),"attack_links":attack_links,"attack_totals":attack_totals(trace),"deaths":deaths,"events":meta.get("events") or extract_events(trace),"metadata":{"method":infer_method_display_name(meta),"checkpoint":meta.get("checkpoint_sampled_steps","?"),"training_seed":meta.get("checkpoint_training_seed","?"),"episode_seed":meta.get("episode_seed","?"),"total_waves":meta.get("total_waves",3),"radius":radius,"ranges":_ranges(trace,radius),"success":bool(meta.get("red_success")),"termination":meta.get("termination_reason",""),"return":meta.get("episode_return",0),"title":title,"fps":float(playback_fps),"stride":max(1,int(frame_stride))}}
    try:
        from plotly.offline import get_plotlyjs; bundle=get_plotlyjs()
    except ImportError as exc: raise RuntimeError("plotly is required for interactive rendering; install plotly>=5.18") from exc
    template=r'''<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title><style>body{margin:0;font:14px Arial;color:#222}#bar{padding:8px;background:#f4f6f8;display:flex;gap:6px;align-items:center;flex-wrap:wrap}button,select{padding:5px}#plot-wrap{position:relative}#plot{height:calc(100vh - 185px)}#status,#events,#result,#error-panel,#loading{padding:6px 10px;white-space:pre-wrap}#result{background:#eaf5ea;font-weight:bold}#error-panel{background:#fde8e8;color:#a00}label{margin-left:8px}</style></head><body><div id="bar"><button id="play">Play</button><button id="prev">Previous Frame</button><button id="next">Next Frame</button><button id="restart">Restart</button><select id="speed"><option value=".25">0.25x</option><option value=".5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option></select><select id="trail"><option value="full">Full trail</option><option value="5">5s trail</option><option value="10">10s trail</option><option value="20">20s trail</option></select><button id="camera">Reset Camera</button><label><input id="heads" type="checkbox" checked> headings</label><label><input id="labels" type="checkbox" checked> labels</label><label><input id="markers" type="checkbox" checked> death markers</label><input id="slider" type="range" min="0" max="__MAX__" value="0" step="1" style="min-width:240px"></div><div id="loading">Loading interactive replay...</div><div id="error-panel" hidden></div><div id="plot-wrap"><div id="plot"></div></div><div id="status"></div><div id="events"></div><div id="result" hidden></div><script>__PLOTLY__</script><script type="application/json" id="payload">__PAYLOAD__</script><script id="application-script">// APP_JS_START
__APP_JS__
// APP_JS_END</script></body></html>'''
    html=template.replace("__TITLE__",str(title).replace("&","&amp;").replace('"','&quot;')).replace("__PLOTLY__",bundle).replace("__PAYLOAD__",json.dumps(payload,ensure_ascii=False,separators=(",",":"))).replace("__MAX__",str(len(trace["steps"])-1)).replace("__APP_JS__",APP_JS)
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(html,encoding="utf-8")

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir');p.add_argument('--trace');p.add_argument('--metadata');p.add_argument('--output');p.add_argument('--playback-fps',type=float,default=20);p.add_argument('--frame-stride',type=int,default=1);p.add_argument('--initial-trail',default='full');p.add_argument('--title',default='Combat Episode Replay');a=p.parse_args();d=Path(a.input_dir) if a.input_dir else None;render(Path(a.trace) if a.trace else d/'episode_trace.npz',Path(a.metadata) if a.metadata else d/'metadata.json',Path(a.output) if a.output else d/'episode_interactive.html',a.playback_fps,a.frame_stride,a.initial_trail,a.title)
if __name__=='__main__':main()
