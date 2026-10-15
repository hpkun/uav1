"""Official single-episode deterministic combat replay entry."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.record_combat_episode import load_recording_policy,record,resolved
from tools.combat_replay_report import export_analysis

FORMATS=('mp4','png','html','csv','report')


def replay(checkpoint,env_config,episode_seed,output_dir,device='cuda',
           algorithm_config=None,playback_speed=1.,formats=FORMATS,ffmpeg=None):
    if not np.isfinite(playback_speed) or playback_speed<=0:
        raise ValueError('playback speed must be finite and positive')
    selected=set(formats)
    if selected-{'trace',*FORMATS}:raise ValueError(f'unknown formats: {selected-set(FORMATS)-{"trace"}}')
    checkpoint=resolved(checkpoint);output=resolved(output_dir)
    config_path=resolved(algorithm_config) if algorithm_config else checkpoint.parent/'algorithm_config.yaml'
    if not config_path.is_file():
        raise FileNotFoundError('No training-directory algorithm_config.yaml; supply --algorithm-config explicitly')
    if 'mp4' in selected:
        from tools.combat_video import configure_ffmpeg
        configure_ffmpeg(ffmpeg)
    ec=yaml.safe_load(resolved(env_config).read_text(encoding='utf-8'))
    ac=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    trainer,state,provenance=load_recording_policy(checkpoint,ec,ac,device)
    provenance.update(environment_config_path=str(resolved(env_config)),algorithm_config_path=str(config_path))
    # Exactly one rollout. All consumers below accept only these saved artifacts.
    metadata=record(trainer,ec,episode_seed,output,provenance)
    trace=output/'episode_trace.npz';meta=output/'metadata.json'
    if 'png' in selected:
        from tools.render_combat_episode import render
        render(trace,meta,output/'trajectory.png')
    if 'mp4' in selected:
        from tools.render_combat_episode import render
        render(trace,meta,output/'episode.mp4',stride=1,fps=playback_speed/metadata['dt'])
    if 'html' in selected:
        from tools.render_combat_episode_interactive import render
        render(trace,meta,output/'interactive.html',stride=1,playback_speed=playback_speed)
    export_analysis(trace,meta,output,selected)
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--env-config',required=True)
    p.add_argument('--algorithm-config')
    p.add_argument('--episode-seed',required=True,type=int)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cuda',choices=('cuda',))
    p.add_argument('--playback-speed',default=1.,type=float)
    p.add_argument('--formats',nargs='+',default=list(FORMATS),help='mp4 png html csv report; comma-separated also accepted; trace always saved')
    p.add_argument('--ffmpeg',help='Optional FFmpeg binary path; otherwise FFMPEG_BINARY/PATH')
    args=p.parse_args()
    formats=[value for group in args.formats for value in group.split(',')]
    result=replay(args.checkpoint,args.env_config,args.episode_seed,args.output_dir,args.device,
        args.algorithm_config,args.playback_speed,formats,args.ffmpeg)
    print(json.dumps({key:result[key] for key in ('algorithm','checkpoint_sampled_steps','episode_steps',
        'simulation_time_s','termination_reason','red_survivors','blue_survivors','episode_return')},indent=2))


if __name__=='__main__':main()
