"""Headless incremental Matplotlib renderer for recorded combat traces."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, writers
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.combat_visualization import (blue_losses_at_frame, extract_death_frames, extract_events,
    infer_method_display_name, read_trace, recent_events, trace_frame_to_render_index,
    trajectory_slice, heading_endpoint)

def set_line(line, xyz):
    line.set_data(xyz[:, 0], xyz[:, 1]); line.set_3d_properties(xyz[:, 2])

def render(trace_path: Path, metadata_path: Path, preview_path: Path, mp4_path: Path | None,
           fps=30, stride=4, dpi=100, elev=25, azim=-55, trail_length=0, heading=True,
           preview_frame=None, event_hold_seconds=1.0, final_hold_seconds=2.0):
    if not trace_path.is_file(): raise FileNotFoundError(f"trace not found: {trace_path}; run recorder first")
    if not metadata_path.is_file(): raise FileNotFoundError(f"metadata not found: {metadata_path}; run recorder first")
    trace = read_trace(trace_path); metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("trace_schema_version", -1)) != 1: raise ValueError("unsupported trace schema")
    n = len(trace["steps"]); stride = max(1, int(stride)); fps = max(1, int(fps))
    rendered = list(range(0, n, stride)); rendered += [] if rendered[-1] == n - 1 else [n - 1]
    events = metadata.get("events") or extract_events(trace); deaths = extract_death_frames(trace)
    event_render_frames = [trace_frame_to_render_index(rendered, int(e.get("trace_frame", 0))) for e in events]
    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi); ax = fig.add_subplot(111, projection="3d")
    radius=float(metadata.get("arena_radius",5000)); ax.set_xlim(-radius,radius); ax.set_ylim(-radius,radius)
    h=np.concatenate([(-trace["red_kinematics"][...,2]).ravel(),(-trace["blue_kinematics"][...,2]).ravel()]); h=h[np.isfinite(h)]
    ax.set_zlim(0,max(500.,float(h.max()+100) if h.size else 500.)); ax.set_xlabel("X / m"); ax.set_ylabel("Y / m"); ax.set_zlabel("Altitude / m"); ax.view_init(elev=elev,azim=azim)
    c=np.linspace(0,2*np.pi,128); ax.plot(radius*np.cos(c),radius*np.sin(c),np.zeros_like(c),color="lightgray",lw=.7)
    red_lines=[ax.plot([],[],[],color="crimson",lw=1.2)[0] for _ in range(4)]
    styles=["-","--",":"]; blue_lines=[[ax.plot([],[],[],color="royalblue",ls=styles[w],lw=1.)[0] for _ in range(4)] for w in range(trace["blue_kinematics"].shape[1])]
    red_markers=[ax.plot([],[],[],marker="o",color="crimson",ls="None",ms=5)[0] for _ in range(4)]; blue_markers=[ax.plot([],[],[],marker="o",color="royalblue",ls="None",ms=5)[0] for _ in range(4)]
    red_labels=[ax.text(0,0,0,f"R{i+1}",color="crimson",fontsize=8) for i in range(4)]; blue_labels=[ax.text(0,0,0,f"B{i+1}",color="royalblue",fontsize=8) for i in range(4)]
    red_heads=[ax.plot([],[],[],color="crimson",lw=.8)[0] for _ in range(4)] if heading else []; blue_heads=[ax.plot([],[],[],color="royalblue",lw=.8)[0] for _ in range(4)] if heading else []
    death_artists=[]
    for death in deaths:
        x,y,z=death["position"]; color="crimson" if death["side"]=="red" else "royalblue"; death_artists.append((death,ax.plot([x],[y],[-z],marker="x",color=color,ms=7,ls="None")[0]))
    overlay=fig.text(.02,.95,"",va="top",family="monospace"); event_text=fig.text(.98,.08,"",ha="right",va="bottom",family="monospace",color="dimgray")
    fig.legend(handles=[Line2D([],[],color="crimson",label="Red UAV"),Line2D([],[],color="royalblue",label="Blue UAV"),Line2D([],[],color="royalblue",ls="-",label="Blue Wave 1"),Line2D([],[],color="royalblue",ls="--",label="Blue Wave 2"),Line2D([],[],color="royalblue",ls=":",label="Blue Wave 3")],loc="upper right")
    hold=max(0,round(fps*float(final_hold_seconds))); animation_frames=rendered+[rendered[-1]]*hold
    def update(index):
        frame=animation_frames[index]; start=max(0,frame-int(trail_length)) if trail_length else 0
        for a,line in enumerate(red_lines):
            data=trajectory_slice(trace["red_kinematics"][:,a],frame,trail_length); valid=np.isfinite(data[:,0]); set_line(line,data[valid][:,[0,1,2]]*np.array([1,1,-1])) if valid.any() else set_line(line,np.empty((0,3)))
            alive=bool(trace["red_alive"][frame,a]); state=trace["red_kinematics"][frame,a]; red_markers[a].set_visible(alive); red_labels[a].set_visible(alive)
            if alive: set_line(red_markers[a],state[[0,1,2]][None]*np.array([1,1,-1])); red_labels[a].set_position((state[0],state[1])); red_labels[a].set_3d_properties(-state[2])
            if heading: red_heads[a].set_visible(alive); set_line(red_heads[a],np.array([[state[0],state[1],-state[2]],heading_endpoint(state)*np.array([1,1,-1])])) if alive else None
        active=max(0,min(len(blue_lines)-1,int(trace["active_wave"][frame])-1))
        for w in range(len(blue_lines)):
            for a,line in enumerate(blue_lines[w]):
                data=trajectory_slice(trace["blue_kinematics"][:,w,a],frame,trail_length); valid=np.isfinite(data[:,0]); set_line(line,data[valid][:,[0,1,2]]*np.array([1,1,-1])) if valid.any() else set_line(line,np.empty((0,3)))
        for a,marker in enumerate(blue_markers):
            alive=bool(trace["blue_alive"][frame,active,a]); state=trace["blue_kinematics"][frame,active,a]; marker.set_visible(alive); blue_labels[a].set_visible(alive)
            if alive: set_line(marker,state[[0,1,2]][None]*np.array([1,1,-1])); blue_labels[a].set_position((state[0],state[1])); blue_labels[a].set_3d_properties(-state[2])
            if heading: blue_heads[a].set_visible(alive); set_line(blue_heads[a],np.array([[state[0],state[1],-state[2]],heading_endpoint(state)*np.array([1,1,-1])])) if alive else None
        for death,artist in death_artists: artist.set_visible(frame>=death["frame"])
        final=index>=len(rendered); total=int(metadata.get("total_waves",3)); overlay.set_text(f"{infer_method_display_name(metadata)}\nCheckpoint: {metadata.get('checkpoint_sampled_steps','?')}\nEpisode seed: {metadata.get('episode_seed','?')}\n\nSim Time: {trace['time_s'][frame]:.1f} s\nWave: {int(trace['active_wave'][frame])} / {total}\nWaves Cleared: {int(trace['waves_cleared'][frame])}\nRed Alive: {int(trace['red_alive'][frame].sum())} / 4\nBlue Alive: {int(trace['blue_alive'][frame,active].sum())} / 4\nTotal Blue Losses: {blue_losses_at_frame(deaths, frame)}")
        if final: event_text.set_text(f"{'MISSION SUCCESS' if metadata.get('red_success') or metadata.get('waves_cleared',0)==total else 'TIMEOUT' if metadata.get('termination_reason')=='red_failure_timeout' else 'DRAW' if 'draw' in str(metadata.get('termination_reason','')) else 'RED FAILURE'}\nWaves Cleared: {metadata.get('waves_cleared',0)} / {total}\nRed Survivors: {metadata.get('final_red_survivors',0)} / 4\nTotal Blue Losses: {metadata.get('total_blue_losses',0)}\nEpisode Return: {metadata.get('episode_return',0):.2f}\nSimulation Time: {metadata.get('simulation_time_s',0):.1f} s")
        else:
            active_events=[e for e, rendered_frame in zip(events, event_render_frames) if rendered_frame<=index and index-rendered_frame<max(1,round(fps*event_hold_seconds))]
            labels=[f"WAVE {e['wave']} CLEARED" if e['type']=='wave_cleared' else f"WAVE {e['wave']} SPAWNED" if e['type']=='wave_spawned' else e['type'].upper().replace('_',' ')+f" +{e.get('count',1)}" for e in active_events[-5:]]
            event_text.set_text("\n".join(labels))
        if index==0 or index==len(animation_frames)-1 or index%25==0: print(f"[RENDER] frame {index+1}/{len(animation_frames)} step={int(frame)}",flush=True)
        return []
    target=n//2 if preview_frame is None else int(preview_frame); target_index=min(range(len(rendered)),key=lambda i:abs(rendered[i]-target)); update(target_index); fig.savefig(preview_path,dpi=dpi)
    if mp4_path is not None:
        if not writers.is_available("ffmpeg"): raise RuntimeError("ffmpeg is required for MP4 rendering")
        FuncAnimation(fig,update,frames=len(animation_frames),interval=1000/fps,blit=False).save(mp4_path,writer=FFMpegWriter(fps=fps),dpi=dpi)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input-dir"); p.add_argument("--trace"); p.add_argument("--metadata"); p.add_argument("--output"); p.add_argument("--preview"); p.add_argument("--fps",type=int,default=30); p.add_argument("--frame-stride",type=int,default=4); p.add_argument("--dpi",type=int,default=100); p.add_argument("--elev",type=float,default=25); p.add_argument("--azim",type=float,default=-55); p.add_argument("--trail-length",type=int,default=0); p.add_argument("--no-heading",action="store_true"); p.add_argument("--preview-frame",type=int); p.add_argument("--event-hold-seconds",type=float,default=1.0); p.add_argument("--final-hold-seconds",type=float,default=2.0); a=p.parse_args(); d=Path(a.input_dir) if a.input_dir else None; trace=Path(a.trace) if a.trace else d/"episode_trace.npz"; metadata=Path(a.metadata) if a.metadata else d/"metadata.json"; output=Path(a.output) if a.output else (None if a.preview else d/"episode.mp4"); preview=Path(a.preview) if a.preview else (output.with_name("preview.png") if output else Path("preview.png")); render(trace,metadata,preview,output,a.fps,a.frame_stride,a.dpi,a.elev,a.azim,a.trail_length,not a.no_heading,a.preview_frame,a.event_hold_seconds,a.final_hold_seconds)
if __name__ == "__main__": main()
