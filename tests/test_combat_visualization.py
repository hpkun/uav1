from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import pytest

from env.models import AircraftState
from env.persistent_env import PersistentWaveCombatEnv
from tools.combat_visualization import (TRACE_SCHEMA_VERSION, RecordingPersistentWaveCombatEnv,
    annotate_death_attack_sources, append_frame, assert_episode_seed_allowed,
    attack_totals, blue_losses_at_frame, ensure_fresh_output, extract_death_frames,
    heading_endpoint, read_trace, recent_events, states_array, trace_frame_to_render_index,
    write_trace)
from tools.render_combat_episode_interactive import _derive_attack_links, render as render_interactive

ROOT = Path(__file__).resolve().parents[1]


def same_state(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_state(left[key], right[key]) for key in left)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    return left == right


def test_aircraft_state_and_trace_shapes(tmp_path):
    state = AircraftState(1, 2, -3, 4, 5, 6, True)
    result = states_array([state])
    assert np.array_equal(result[0], [1, 2, -3, 4, 5, 6])
    assert np.all(np.isnan(result[1:]))
    env = RecordingPersistentWaveCombatEnv(ROOT / "configs/persistent_wave_v2_environment.yaml")
    env.reset(40000000)
    frames = {key: [] for key in ("red_kinematics", "red_alive", "blue_kinematics", "blue_alive", "steps", "time_s", "active_wave", "waves_cleared")}
    append_frame(frames, env, 1, 0, 0.0)
    trace_path = tmp_path / "trace.npz"
    arrays = write_trace(trace_path, frames, {})
    trace = read_trace(trace_path)
    assert trace["red_kinematics"].shape == (1, 4, 6)
    assert trace["blue_kinematics"].shape == (1, 3, 4, 6)
    assert np.all(np.isnan(trace["blue_kinematics"][0, 1:]))
    assert np.all(~trace["blue_alive"][0, 1:])


def test_reserved_seed_guard_and_fresh_output(tmp_path):
    for seed in (29000000, 30000199):
        with pytest.raises(ValueError): assert_episode_seed_allowed(seed)
    assert_episode_seed_allowed(40000000)
    ensure_fresh_output(tmp_path / "new")
    (tmp_path / "new" / "x").write_text("x")
    with pytest.raises(FileExistsError): ensure_fresh_output(tmp_path / "new")


def test_visualization_subclass_captures_pre_spawn_terminal_state():
    env = RecordingPersistentWaveCombatEnv(ROOT / "configs/persistent_wave_v2_environment.yaml")
    env.reset(40000000)
    old = [state.copy() for state in env.blue]
    env._spawn_next_wave()
    snapshot = env.visualization_spawn_snapshot
    assert snapshot["step"] == env.steps and snapshot["old_wave_index"] == 1
    assert np.array_equal(states_array(snapshot["old_blue"]), states_array(old))


def test_visualization_observer_is_spawn_semantics_neutral():
    base = PersistentWaveCombatEnv(ROOT / "configs/persistent_wave_v2_environment.yaml")
    observed = RecordingPersistentWaveCombatEnv(ROOT / "configs/persistent_wave_v2_environment.yaml")
    base.reset(40000000); observed.reset(40000000)
    assert np.array_equal(states_array(base.red), states_array(observed.red))
    assert np.array_equal(states_array(base.blue), states_array(observed.blue))
    base_angle = base._spawn_next_wave(); observed_angle = observed._spawn_next_wave()
    assert base_angle == observed_angle
    assert base.last_spawn_candidate_index == observed.last_spawn_candidate_index
    assert base.last_minimum_spawn_distance == observed.last_minimum_spawn_distance
    assert np.array_equal(states_array(base.red), states_array(observed.red))
    assert np.array_equal(states_array(base.blue), states_array(observed.blue))
    assert same_state(base.rng.bit_generator.state, observed.rng.bit_generator.state)


def test_altitude_rule_is_ned_negation():
    state = AircraftState(0, 0, -123, 1, 0, 0)
    assert -state.z == 123


def test_heading_endpoint_pitch_uses_ned_sign():
    climbing = np.array([0, 0, -100, 100, 0.2, 0], dtype=float)
    descending = np.array([0, 0, -100, 100, -0.2, 0], dtype=float)
    assert -heading_endpoint(climbing)[2] > -climbing[2]
    assert -heading_endpoint(descending)[2] < -descending[2]


def test_blue_loss_timeline_has_no_future_leakage():
    deaths = [
        {"side": "blue", "frame": 3},
        {"side": "red", "frame": 4},
        {"side": "blue", "frame": 8},
    ]
    assert blue_losses_at_frame(deaths, 0) == 0
    assert blue_losses_at_frame(deaths, 3) == 1
    assert blue_losses_at_frame(deaths, 99) == 2


def test_recent_events_excludes_future_events():
    events = [{"trace_frame": frame, "type": str(frame)} for frame in (1, 3, 8)]
    assert [event["trace_frame"] for event in recent_events(events, 3)] == [1, 3]
    assert recent_events(events, 0) == []


def test_schema_v1_attack_pair_reconstruction_and_death_cause():
    red = np.full((2, 4, 6), np.nan, dtype=np.float32)
    blue = np.full((2, 1, 4, 6), np.nan, dtype=np.float32)
    red[:, 0] = [0, 0, -1000, 200, 0, 0]
    blue[:, 0, 0] = [1000, 0, -1000, 200, 0, np.pi]
    trace = {
        "red_kinematics": red, "blue_kinematics": blue,
        "red_alive": np.asarray([[True, False, False, False], [True, False, False, False]]),
        "blue_alive": np.asarray([[[True, False, False, False]], [[False, False, False, False]]]),
        "steps": np.asarray([0, 1]), "time_s": np.asarray([0.0, 0.1]),
        "active_wave": np.asarray([1, 1]), "waves_cleared": np.asarray([0, 1]),
        "red_step_fire_attempts": np.asarray([1]), "blue_step_fire_attempts": np.asarray([1]),
        "red_step_attack_kills": np.asarray([1]), "blue_step_attack_kills": np.asarray([0]),
        "red_boundary_exit_delta": np.asarray([0]), "blue_boundary_exit_delta": np.asarray([0]),
        "red_ground_loss_delta": np.asarray([0]), "blue_ground_loss_delta": np.asarray([0]),
        "spawned_next_wave": np.asarray([False]), "arena_radius": np.asarray(5000.0),
    }
    links = _derive_attack_links(trace, {"environment_variant": "persistent_wave_v2"})
    assert {(link["side"], link["attacker"], link["target"]) for link in links} == {
        ("red", 1, 1), ("blue", 1, 1),
    }
    deaths = extract_death_frames(trace)
    assert [(death["side"], death["agent"], death["cause"]) for death in deaths] == [
        ("blue", 1, "attack_kill"),
    ]
    assert all("hit" not in link and "kill" not in link for link in links)


def test_attack_totals_use_hits_not_kills_for_misses():
    totals = attack_totals({
        "red_step_fire_attempts": np.asarray([2, 1]),
        "blue_step_fire_attempts": np.asarray([2]),
        "red_step_weapon_hits": np.asarray([2]),
        "blue_step_weapon_hits": np.asarray([1]),
        "red_step_attack_kills": np.asarray([1]),
        "blue_step_attack_kills": np.asarray([1]),
    })
    assert totals == {"attempts": 5, "hits": 3, "misses": 2, "kills": 2}


def test_attack_reconstruction_resets_fire_state_after_wave_spawn():
    red = np.full((3, 4, 6), np.nan, dtype=np.float32)
    blue = np.full((3, 2, 4, 6), np.nan, dtype=np.float32)
    red[:, 0] = [0, 0, -1000, 200, 0, 0]
    blue[0:2, 0, 0] = [1000, 0, -1000, 200, 0, np.pi]
    blue[1:3, 1, 0] = [1000, 0, -1000, 200, 0, np.pi]
    red_alive = np.zeros((3, 4), dtype=bool); red_alive[:, 0] = True
    blue_alive = np.zeros((3, 2, 4), dtype=bool)
    blue_alive[0, 0, 0] = True; blue_alive[1:, 1, 0] = True
    trace = {
        "red_kinematics": red, "blue_kinematics": blue,
        "red_alive": red_alive, "blue_alive": blue_alive,
        "steps": np.arange(3), "time_s": np.arange(3) * 0.1,
        "active_wave": np.asarray([1, 2, 2]), "waves_cleared": np.asarray([0, 1, 1]),
        "red_step_fire_attempts": np.asarray([1, 1]),
        "blue_step_fire_attempts": np.asarray([1, 1]),
        "spawned_next_wave": np.asarray([True, False]),
    }
    links = _derive_attack_links(trace, {"environment_variant": "persistent_wave_v2"})
    assert [(link["frame"], link["side"], link["wave"]) for link in links] == [
        (1, "red", 1), (1, "blue", 1), (2, "red", 2), (2, "blue", 2),
    ]


def test_schema_v1_attack_source_evidence_is_conservative():
    death = {"side": "blue", "wave": 2, "agent": 3, "frame": 10,
             "cause": "attack_kill"}
    single = annotate_death_attack_sources([death], [
        {"side": "red", "wave": 2, "attacker": 4, "target": 3, "frame": 10},
    ])[0]
    assert single["attack_source_evidence"] == "Single reconstructed attacker: R4"
    multiple = annotate_death_attack_sources([death], [
        {"side": "red", "wave": 2, "attacker": 2, "target": 3, "frame": 10},
        {"side": "red", "wave": 2, "attacker": 4, "target": 3, "frame": 10},
    ])[0]
    assert multiple["attack_sources"] == ["R2", "R4"]
    assert "unique killer unavailable" in multiple["attack_source_evidence"]


def test_event_hold_mapping_uses_rendered_frame_units_with_stride_four():
    rendered = [0, 4, 8, 12, 16]
    assert trace_frame_to_render_index(rendered, 1) == 1
    assert trace_frame_to_render_index(rendered, 4) == 1
    assert trace_frame_to_render_index(rendered, 9) == 3
    # At 30 fps an event beginning at rendered index 1 remains for 30 output frames,
    # independent of the four-trace-frame stride.
    hold_frames = round(30 * 1.0)
    assert sum(1 for index in range(100) if 1 <= index and index - 1 < hold_frames) == 30


def test_interactive_html_is_standalone_and_accepts_old_metadata(tmp_path):
    frames = 3
    red = np.zeros((frames, 4, 6), dtype=np.float32)
    blue = np.full((frames, 2, 4, 6), np.nan, dtype=np.float32)
    blue[:, 0] = 0
    trace_path = tmp_path / "episode_trace.npz"
    np.savez_compressed(trace_path, red_kinematics=red,
        red_alive=np.ones((frames, 4), dtype=bool), blue_kinematics=blue,
        blue_alive=np.concatenate((np.ones((frames, 1, 4), dtype=bool),
                                   np.zeros((frames, 1, 4), dtype=bool)), axis=1),
        steps=np.arange(frames), time_s=np.arange(frames) * 0.1,
        active_wave=np.ones(frames, dtype=int), waves_cleared=np.zeros(frames, dtype=int))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"trace_schema_version": 1, "total_waves": 2}), encoding="utf-8")
    output = tmp_path / "episode_interactive.html"
    render_interactive(trace_path, metadata_path, output)
    html = output.read_text(encoding="utf-8")
    assert output.stat().st_size > 1_000_000
    assert "<script src=" not in html.lower()
    assert all(token in html for token in ("Play", "Reset Camera", "slider", "uirevision", "scatter3d"))
    payload = html.split('<script type="application/json" id="payload">', 1)[1].split("</script>", 1)[0]
    json.loads(payload)
    assert "NaN" not in payload
    application = html.split("// APP_JS_START", 1)[1].split("// APP_JS_END", 1)[0]
    assert "\nFINAL" not in application
    assert all(token in application for token in (
        "cameraInteracting", "beginCameraInteraction", "endCameraInteraction",
        "renderBusy", "renderPending", "renderEpoch", "requestAnimationFrame(playbackLoop)",
        "pointerdown", "pointerup", "pointercancel", "wheelEndTimer",
        "setTimeout", "arenaTrace", "scrollZoom:true",
        "aspectmode:'cube'", "dragmode:'orbit'", "autorange:false",
    ))
    assert "setInterval(" not in application
    assert "clearInterval(" not in application
    assert "aspectmode:'data'" not in application
    assert "if(cameraInteracting||renderBusy)return" in application
    assert "dynamicTraceIndices" not in application
    assert "staticTraceIndices" not in application
    assert "hideDynamicCombatTraces" not in application
    assert "scheduleHideDynamic" not in application
    assert "interactionHidePending" not in application
    render_frame = application.split("async function renderFrame", 1)[1].split("async function drainRenderQueue", 1)[0]
    assert "Plotly.relayout" not in render_frame
    assert "scene.camera" not in render_frame
    assert render_frame.count("Plotly.restyle") == 6
    assert render_frame.count("if(renderAborted(epoch))return false") >= 7
    assert "return true" in render_frame
    assert "P.attack_links.filter" in render_frame
    assert "color:'#20c75a'" in application
    assert "name:'Fire Attempt'" in application
    assert "Cause: %{customdata[3]}" in application
    begin_interaction = application.split("function beginCameraInteraction", 1)[1].split("function endCameraInteraction", 1)[0]
    assert "Plotly." not in begin_interaction
    assert "renderEpoch+=1" in begin_interaction
    assert "requestRender" not in begin_interaction
    assert "renderPending" not in begin_interaction
    end_interaction = application.split("function endCameraInteraction", 1)[1].split("function syncLogicalFrame", 1)[0]
    assert "if(renderPending||logicalFrame!==renderedFrame)requestRender()" in end_interaction
    scheduler = application.split("async function drainRenderQueue", 1)[1].split("function requestRender", 1)[0]
    assert "if(completed)renderedFrame=target" in scheduler
    assert "else{renderPending=true;break}" in scheduler
    assert "Adjusting camera" not in html
    assert "camera-interaction" not in html
    assert "combat traces are hidden" not in html
    assert "uirevision:'combat-replay'" in application
    assert "'scene.camera':DEFAULT_CAMERA" in application
    node = shutil.which("node")
    if node:
        script = tmp_path / "application.js"
        script.write_text(application, encoding="utf-8")
        checked = subprocess.run([node, "--check", str(script)], capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr
