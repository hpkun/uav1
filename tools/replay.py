"""Record a verified interception checkpoint and render an offline HTML replay."""
import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path

import numpy as np

from env.interception import InterceptionEnv
from experiment.interception import digest,load_run


def collect(config,policy,mode,seed):
    env=InterceptionEnv(replace(config,attack_mode=mode))
    obs,info=env.reset(seed=seed)
    frames=[]
    events=[]

    def frame(step,actions=None):
        return dict(step=step,time_s=step*config.dt,positions=env.positions.tolist(),
                    velocities=env.velocities.tolist(),attacker_alive=env.attacker_alive.tolist(),
                    defender_actions=None if actions is None else np.asarray(actions).tolist())

    frames.append(frame(0))
    while True:
        actions=policy.policy_actions(obs[None],np.ones((1,5),np.float32),deterministic=True)[0]
        obs,_,terminated,truncated,info=env.step(actions)
        frames.append(frame(env.step_count,actions))
        events.extend(info['events'])
        if terminated or truncated:
            break
    trace=dict(schema='InterceptionReplay/1',mode=mode,seed=seed,
               config=env.contract['config'],frames=frames,events=events,summary=info['episode_summary'])
    env.close()
    return trace


def validate_trace(trace):
    frames=trace['frames'];events=trace['events'];summary=trace['summary']
    if len(frames)!=summary['steps']+1 or [row['step'] for row in frames]!=list(range(len(frames))):
        raise ValueError('Replay frame count or order mismatch')
    if any(not 1<=event['step']<=summary['steps'] for event in events):
        raise ValueError('Replay event lies outside the episode')
    if sum(event['type']=='capture' for event in events)!=summary['captured_count']:
        raise ValueError('Replay capture events disagree with episode')
    if sum(event['type']=='penetration' for event in events)!=summary['penetrated_count']:
        raise ValueError('Replay penetration events disagree with episode')
    if summary['outcome']=='penetrated' and not summary['penetrated_count']:
        raise ValueError('Penetration outcome has no event')
    if summary['outcome']=='all_captured' and summary['captured_count']!=5:
        raise ValueError('All-capture outcome has incomplete captures')
    return trace


def html(trace,stride):
    keep=set(range(0,len(trace['frames']),stride))|{0,len(trace['frames'])-1}
    keep.update(event['step'] for event in trace['events'])
    display=[trace['frames'][index] for index in sorted(keep)]
    data=json.dumps(dict(frames=display,events=trace['events'],summary=trace['summary'],
                         config=trace['config'],mode=trace['mode'],seed=trace['seed']),
                    ensure_ascii=False,separators=(',',':'))
    data=data.replace('</','<\\/')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Interception replay</title>
<style>body{margin:0;background:#101722;color:#e9eef5;font:15px system-ui;display:grid;grid-template-columns:minmax(430px,1fr) 300px;min-height:100vh}main{padding:20px}aside{padding:20px;background:#1b2634}canvas{width:min(75vh,100%);aspect-ratio:1;background:#f5f7fb;border-radius:8px}button,input{accent-color:#67d1ee}button{background:#31475c;color:white;border:0;border-radius:5px;padding:7px 14px;cursor:pointer}li{margin:9px 0;cursor:pointer}small{color:#aebccc}</style>
<main><h1>Airbase interception</h1><p id="status"></p><canvas id="view" width="800" height="800"></canvas><p><button id="play">Play</button> <input id="slider" type="range" min="0" value="0" style="width:65%"></p><small>Blue: defenders · Red: attackers · Dashed ring: danger · Solid ring: breach</small></main>
<aside><h2>Episode</h2><p id="summary"></p><h2>Events</h2><ol id="events"></ol></aside>
<script id="trace" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('trace').textContent),f=d.frames,c=document.getElementById('view'),g=c.getContext('2d'),slider=document.getElementById('slider');let index=0,timer=null;
slider.max=f.length-1;const scale=x=>4*(x+100),sy=y=>4*(100-y),circle=(x,y,r,color,fill=false)=>{g.beginPath();g.arc(scale(x),sy(y),r*4,0,2*Math.PI);g.strokeStyle=color;g.fillStyle=color;g.lineWidth=2;fill?g.fill():g.stroke()};
function draw(){const row=f[index],cfg=d.config;g.clearRect(0,0,800,800);g.strokeStyle='#d9e0e8';g.lineWidth=1;for(let n=0;n<=800;n+=80){g.beginPath();g.moveTo(n,0);g.lineTo(n,800);g.moveTo(0,n);g.lineTo(800,n);g.stroke()}
g.setLineDash([7,6]);circle(cfg.base_x,cfg.base_y,cfg.danger_radius,'#e9a645');g.setLineDash([]);circle(cfg.base_x,cfg.base_y,cfg.breach_radius,'#e96864');circle(cfg.base_x,cfg.base_y,3,'#1c2631',true);
for(let side=0;side<2;side++)for(let i=0;i<5;i++){if(side===1&&!row.attacker_alive[i])continue;const p=row.positions[side][i];circle(p[0],p[1],2.4,side===0?'#2b84d6':'#e45850',true);g.fillStyle='#111';g.fillText((side===0?'D':'A')+i,scale(p[0])+10,sy(p[1])-9)}
document.getElementById('status').textContent=`${d.mode} · seed ${d.seed} · step ${row.step} · ${row.time_s.toFixed(1)} s`;slider.value=index}
slider.oninput=()=>{index=+slider.value;draw()};document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;return}timer=setInterval(()=>{index=Math.min(index+1,f.length-1);draw();if(index===f.length-1){clearInterval(timer);timer=null}},70)};
document.getElementById('summary').textContent=`${d.summary.outcome}; captured ${d.summary.captured_count}/5; ${d.summary.duration_s.toFixed(1)} s`;
const list=document.getElementById('events');for(const event of d.events){const li=document.createElement('li');li.textContent=`${event.time_s.toFixed(1)} s · ${event.type}`;li.onclick=()=>{index=f.findIndex(row=>row.step===event.step);draw()};list.append(li)}draw();
</script></html>'''
    return page.replace('__DATA__',data)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--label',choices=('Best','Last'),default='Best')
    p.add_argument('--attack-mode',choices=('direct_rush','split_attack'),required=True)
    p.add_argument('--seed',type=int,default=900001)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--frame-stride',type=int,default=5)
    p.add_argument('--html',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.frame_stride<1 or args.output.exists():raise ValueError('Invalid stride or existing output')
    config,policy,name,model,_=load_run(args.checkpoint,args.label,args.device)
    trace=validate_trace(collect(config,policy,args.attack_mode,args.seed))
    trace['algorithm']=name
    trace['checkpoint_sha256']=digest(model)
    args.output.mkdir(parents=True)
    with gzip.open(args.output/'trace.json.gz','wt',encoding='utf-8') as file:
        json.dump(trace,file,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    (args.output/'replay.html').write_text(html(trace,args.frame_stride),encoding='utf-8')
    (args.output/'summary.json').write_text(json.dumps(trace['summary'],indent=2)+'\n',encoding='utf-8')
    print(json.dumps(trace['summary'],indent=2))
    return trace


if __name__=='__main__':main()
