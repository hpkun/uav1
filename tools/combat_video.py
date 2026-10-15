"""Explicit FFmpeg discovery; never silently write a Pillow file as MP4."""
import os
from pathlib import Path
import shutil
import subprocess


def configure_ffmpeg(binary=None):
    import matplotlib
    candidate=binary or os.environ.get('FFMPEG_BINARY') or shutil.which('ffmpeg')
    if candidate:candidate=shutil.which(str(candidate)) or candidate
    if not candidate:
        configured=matplotlib.rcParams['animation.ffmpeg_path']
        candidate=configured if Path(configured).is_file() else None
    if not candidate or not Path(candidate).is_file():
        raise RuntimeError('FFmpeg required for MP4. Set --ffmpeg / FFMPEG_BINARY / PATH, or select --formats png,html,csv,report.')
    try:
        subprocess.run([str(candidate),'-version'],check=True,capture_output=True,timeout=10)
    except (OSError,subprocess.SubprocessError) as error:
        raise RuntimeError(f'FFmpeg cannot run: {candidate}') from error
    matplotlib.rcParams['animation.ffmpeg_path']=str(Path(candidate).resolve())
    return Path(candidate).resolve()
