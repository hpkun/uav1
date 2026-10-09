"""Render existing representative traces at simulation speed without reevaluation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.render_combat_episode import render
from tools.render_combat_episode_interactive import render as render_interactive
from tools.create_combat_v24_videos import probe_video, write_gallery
from tools.combat_visualization import dump_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'outputs/combat_v24_8v8_videos')
    args = parser.parse_args()
    results = {}
    for directory in sorted(args.directory.glob('*/*')):
        trace, metadata = directory/'episode_trace.npz', directory/'metadata.json'
        if not trace.exists() or directory.parent.name == 'diagnostics':
            continue
        data = json.loads(metadata.read_text())
        fps = round(1/float(data['dt']))
        if abs(fps*float(data['dt'])-1) > 1e-9:
            raise ValueError('simulation dt cannot be represented by integral realtime fps')
        output = directory/'video_realtime.mp4'
        print(f'[REALTIME] {directory.parent.name}/{directory.name} stride=1 fps={fps}', flush=True)
        render(trace, metadata, output, stride=1, fps=fps)
        results[str(directory.relative_to(args.directory))] = probe_video(output)
        dump_metadata(args.directory/'realtime_video_validation.json', results)
        print(results[str(directory.relative_to(args.directory))], flush=True)
    directory = args.directory/'stea_mappo/hard'
    output = render_interactive(directory/'episode_trace.npz', directory/'metadata.json',
        directory/'interactive.html', stride=1, playback_speed=1.)
    print(f'[INTERACTIVE] {output} bytes={output.stat().st_size}', flush=True)
    write_gallery(args.directory, json.loads((args.directory/'summary.json').read_text()),
                  json.loads((args.directory/'selected_seeds.json').read_text()))


if __name__ == '__main__':
    main()
