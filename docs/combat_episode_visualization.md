# Combat Episode Visualization

This is qualitative visualization only. Recorded episodes never enter formal quantitative metrics, aggregation, checkpoint selection, or holdout evidence. The recommended visualization-only seed is `40000000`; monitoring seeds `29000000..29000019` and formal holdout seeds `30000000..30000199` are rejected.

From the repository root on Ubuntu:

```bash
conda activate uav
python -u tools/record_combat_episode.py \
  --checkpoint outputs/<chosen_run>/latest.pt \
  --env-config configs/persistent_wave_v2_environment.yaml \
  --episode-seed 40000000 --device cuda \
  --output-dir outputs/visualization/ea_wb_seed40000000

python -u tools/render_combat_episode.py \
  --input-dir outputs/visualization/ea_wb_seed40000000 \
  --output outputs/visualization/ea_wb_seed40000000/episode.mp4 \
  --fps 30 --frame-stride 4
```

The recorder loads `state["extra"]["algorithm_config"]`, validates the existing modular checkpoint protocol, reconstructs the trainer through the modular factory, and runs deterministic CUDA inference with the same actor/critic hidden-state lifecycle as the evaluator. It reads truth directly from `env.red` and `env.blue`; NED `z` is rendered as `altitude = -z`.

Output is `episode_trace.npz` plus `metadata.json` with `trace_schema_version=1`. Red uses `[F,4,6]` kinematics; Blue uses `[F,total_waves,4,6]`, with inactive wave slots as NaN and false. Blue trajectories are separated by wave and a visualization-only observer captures the old wave's post-combat, pre-respawn terminal states, preventing a spawn “teleport” line. Schema v1 does not directly store per-attempt hit flags or credited killers.

The renderer is headless Matplotlib Agg, fixed-camera 1280x720 at 30 fps and stride 4 by default. Preview PNG generation does not require ffmpeg. MP4 generation explicitly reports `ffmpeg is required for MP4 rendering` when the system writer is unavailable; it never installs packages or runs an environment.

For an offline, mouse-driven replay, generate a standalone Plotly HTML from the same trace (no checkpoint, environment, or CUDA is loaded):

```bash
python -u tools/render_combat_episode_interactive.py \
  --input-dir outputs/visualization/ea_wb_seed40000000 \
  --output outputs/visualization/ea_wb_seed40000000/episode_interactive.html
```

The HTML embeds Plotly JS and supports rotate, zoom, pan, Play/Pause, Previous/Next/Restart, a time slider, 0.25x-4x speed, trail windows, heading/label/death-marker toggles, Recent Events, a final result panel, and Reset Camera. It is intended to be opened directly from disk and does not use a CDN. MP4 remains fixed-camera and cannot be mouse-rotated.

At each entry-triggered **Fire Attempt**, the interactive renderer displays a thick green attacker-to-target line for 0.6 simulation seconds. Green means weapon fire attempt only; it does not mean hit, kill, or successful attack. For schema-v1 traces, these pairs are reconstructed deterministically from the recorded post-motion states, alive masks, weapon envelope, nearest-in-window targeting rule, wave-reset fire state, and per-step fire-attempt counts; generation fails rather than inventing a pair if reconstructed and recorded counts disagree. Aggregate totals are computed independently: misses are attempts minus weapon hits, never attempts minus kills. Aggregate hit counts are never assigned to individual lines. Death-marker hover text reports `Weapon attack`, `Ground impact`, `Boundary exit`, or `Unknown`, derived from the recorded alive transition and per-step cause counters. A death can show `Single reconstructed attacker` only when exactly one reconstructed attempt targets that aircraft in that frame; this is fire-attempt evidence, not unique kill credit. Multiple same-frame sources are listed with `unique killer unavailable`. This adds no fields to the trace and does not change `trace_schema_version=1`.

Interactive playback uses a wall-clock logical playback clock and a separate `requestAnimationFrame` render scheduler. Slow WebGL updates may skip obsolete visual frames, but they do not slow simulation time: the renderer always catches up directly to the latest logical frame and never queues every missed frame. Plotly mutations are serialized, and trajectory/current-marker/heading/fire-attempt/death updates are batched into six data calls per rendered frame instead of one call per aircraft trace.

Camera interaction is playback-aware. Pointer drag or wheel input increments a render epoch and stops the current frame after its in-flight Plotly batch finishes. The already visible combat snapshot stays on screen unchanged: trajectories, current aircraft, headings, Fire Attempt lines, death markers, arena, axes, and grid are neither hidden nor reloaded, and no camera-adjustment overlay is shown. No new combat WebGL mutation runs during the rest of the interaction, while logical playback and the text panels may continue advancing. Releasing both pointer and wheel renders only the latest logical frame when playback actually advanced or a render was interrupted; a paused, otherwise unchanged replay does not redraw merely because the camera moved. Skipped frames are never replayed, expired 0.6-second Fire Attempt lines do not reappear, and `uirevision='combat-replay'` preserves the user's camera. The axes explicitly disable autorange, `aspectmode='cube'` gives the X/Y/Z axes equal on-screen lengths without changing their numeric ranges, the scene uses orbit dragging, and scroll zoom is enabled. Pause and manual seek retain their normal semantics, and changing speed re-anchors the clock without a time jump.

The renderer infers the method name from enabled modules (with optional metadata override), keeps persistent trajectory/marker/heading artists and updates them incrementally, and displays Red/Blue labels, real-trace death markers, structured hit/kill/loss/wave events, and a final result card. `--event-hold-seconds` defaults to 1.0, `--final-hold-seconds` to 2.0, and `--preview-frame` defaults deterministically to the midpoint. `--trail-length 0` means full trace history; a positive value is a trace-frame window. The recorder's `--wall-timeout-s` defaults to 0 (disabled), so environment termination remains authoritative.

For a faster quick video use `--fps 30 --frame-stride 4 --no-heading`. For a detailed reference-like render use `--fps 20 --frame-stride 1 --event-hold-seconds 1.0 --final-hold-seconds 2.0`. No attacker-target kill identity is fabricated because the environment does not expose an unambiguous mapping.
