from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
import trimesh
import yaml

import server as server_module
from model_frame_registration import load_model_frame_offset, save_model_frame_offset
from server import InitInteraction, ObjectState, SyncFrameCache, load_object_specs
from test_model_frame_registration import _pose
from test_server_pose_validity import _Estimator, _setup


def _editor_server(tmp_path):
    server, state, frame = _setup()
    state.spec.model_frame_offset_path = str(tmp_path / "T_fp_object.pkl")
    state.mesh = trimesh.creation.box(extents=(0.06, 0.04, 0.08))
    state.last_accepted_frame = frame
    server.latest_frame = frame
    server._load_model_frame_offsets()
    return server, state, frame


def test_config_default_and_relative_override(tmp_path):
    path = tmp_path / "objects.yml"
    config = {"objects": [{"object_id": "Pipette #1", "mesh_path": "/unused/model.obj"}]}
    path.write_text(yaml.safe_dump(config))
    spec = load_object_specs(str(path))[0]
    assert spec.model_frame_offset_path == (
        "/home/pjlab/dex_manip/data/obj_platform_reg/Pipette #1/T_fp_object.pkl"
    )
    config["objects"][0]["model_frame_offset_path"] = "registration/custom.pkl"
    path.write_text(yaml.safe_dump(config))
    spec = load_object_specs(str(path))[0]
    assert Path(spec.model_frame_offset_path) == tmp_path / "registration" / "custom.pkl"


def test_load_saved_offset_and_fail_on_corrupt_file(tmp_path):
    server, state, _ = _editor_server(tmp_path)
    offset = _pose((0.1, 0.2, 0.0), (0.02, 0.01, 0.0))
    save_model_frame_offset(state.spec.model_frame_offset_path, offset)
    server._load_model_frame_offsets()
    np.testing.assert_allclose(state.T_fp_object, offset)
    Path(state.spec.model_frame_offset_path).write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="Cannot load model-frame offset for 'mug'"):
        server._load_model_frame_offsets()


def test_editor_uses_accepted_frame_and_saved_correction(tmp_path):
    server, state, frame = _editor_server(tmp_path)
    offset = _pose((0.0, np.pi, 0.0), (0.02, 0.0, 0.0))
    save_model_frame_offset(state.spec.model_frame_offset_path, offset)
    server._load_model_frame_offsets()
    server.selected_view_camera_by_object[0] = "camera_top"
    server.latest_frame = replace(frame, sync_timestamp=13.0)
    server._handle_key(ord("m"))
    interaction = server.registration_interaction
    assert interaction is not None
    assert interaction.frame is frame
    assert interaction.camera_name == "camera_top"
    np.testing.assert_allclose(interaction.T_view_object, interaction.T_view_fp @ offset)
    assert not np.shares_memory(interaction.initial_T_fp_object, state.T_fp_object)
    assert not np.shares_memory(interaction.T_view_fp, state.last_T_camera_object)


@pytest.mark.parametrize("camera_name", ["camera_lowfield_1", "camera_top"])
@pytest.mark.parametrize("key,translation", [
    ("a", (-0.001, 0.0, 0.0)), ("d", (0.001, 0.0, 0.0)),
    ("w", (0.0, 0.001, 0.0)), ("s", (0.0, -0.001, 0.0)),
    ("f", (0.0, 0.0, -0.001)), ("r", (0.0, 0.0, 0.001)),
])
def test_translation_keys_follow_rotated_model_in_either_view(tmp_path, camera_name, key, translation):
    server, state, _ = _editor_server(tmp_path)
    server.T_world_camera_map["camera_top"] = _pose((0.3, -0.2, 0.5), (0.5, -0.2, 0.7))
    state.last_T_camera_object = _pose((0.2, 0.4, -0.3), (0.0, 0.0, 1.0))
    state.last_T_world_object = (
        server.T_world_camera_map[state.source_camera] @ state.last_T_camera_object
    )
    offset = _pose((0.0, 0.0, np.pi / 2), (0.02, 0.01, 0.0))
    state.T_fp_object = offset.copy()
    server.selected_view_camera_by_object[0] = camera_name
    server._handle_key(ord("m"))
    interaction = server.registration_interaction
    interaction.rotation_step_rad = np.pi / 2
    server._handle_key(ord("l"))
    server._handle_key(ord(key))

    expected_offset = offset @ _pose((np.pi / 2, 0.0, 0.0)) @ _pose(translation=translation)
    np.testing.assert_allclose(
        interaction.T_view_object, interaction.T_view_fp @ expected_offset, atol=1e-12,
    )
    server._handle_key(13)
    saved, loaded = load_model_frame_offset(state.spec.model_frame_offset_path)
    assert loaded
    np.testing.assert_allclose(saved, expected_offset, atol=1e-12)


def test_edit_commit_after_live_motion_applies_frozen_anchor_and_survives_restart(tmp_path):
    server, state, frame = _editor_server(tmp_path)
    server._handle_key(ord("m"))
    interaction = server.registration_interaction
    raw_snapshot = interaction.T_view_fp.copy()
    old_offset = state.T_fp_object.copy()
    server._handle_key(ord("l"))
    server._handle_key(ord("d"))
    desired_pose = interaction.T_view_object.copy()
    np.testing.assert_array_equal(state.T_fp_object, old_offset)
    assert not Path(state.spec.model_frame_offset_path).exists()

    # Exercise the same request path used by the running server during editing.
    next_frame = replace(frame, sync_timestamp=13.0)
    state.estimator.result[0, 3] = 0.01
    server.frame_cache = SyncFrameCache(3)
    server.frame_cache.add(next_frame)
    server.timestamp_tolerance = 0.005
    server.last_tracking_timestamp = frame.sync_timestamp
    published = []
    server._publish_packet = published.append
    reply = server._handle_pose_request(msgpack.packb({"synced_ts": 13.0}))
    assert reply["ok"] and published
    assert state.last_accepted_frame is next_frame
    assert interaction.frame is frame
    np.testing.assert_array_equal(interaction.T_view_fp, raw_snapshot)
    np.testing.assert_array_equal(reply["object_poses"][0]["meta"]["T_fp_object"], old_offset)
    seed = state.estimator.pose_last
    raw_live_pose = state.last_T_camera_object.copy()

    server._handle_key(13)

    assert server.registration_interaction is None
    assert state.estimator.pose_last is seed
    np.testing.assert_array_equal(state.last_T_camera_object, raw_live_pose)
    np.testing.assert_allclose(raw_snapshot @ state.T_fp_object, desired_pose, atol=1e-12)
    assert server.pose_cache.match(13.0, 0.005) is None
    saved, loaded = load_model_frame_offset(state.spec.model_frame_offset_path)
    assert loaded
    np.testing.assert_allclose(saved, state.T_fp_object)
    refreshed = server._handle_pose_request(msgpack.packb({"synced_ts": 13.0}))
    assert refreshed["request_processing"] == "isolated"
    np.testing.assert_allclose(refreshed["object_poses"][0]["meta"]["T_fp_object"], saved)
    np.testing.assert_allclose(refreshed["object_poses"][0]["T_world_object"], state.last_T_world_object @ saved)

    restarted, reloaded_state, _ = _editor_server(tmp_path)
    np.testing.assert_allclose(reloaded_state.T_fp_object, saved)
    restarted._handle_key(ord("m"))
    np.testing.assert_allclose(restarted.registration_interaction.initial_T_fp_object, saved)
    restarted._handle_key(ord("d"))
    second_desired = restarted.registration_interaction.T_view_object.copy()
    second_raw = restarted.registration_interaction.T_view_fp.copy()
    restarted._handle_key(13)
    second_saved, _ = load_model_frame_offset(reloaded_state.spec.model_frame_offset_path)
    np.testing.assert_allclose(second_raw @ second_saved, second_desired, atol=1e-12)


@pytest.mark.parametrize("cancel_key", [27, ord("q")])
def test_restore_and_cancel_do_not_modify_saved_offset_or_tracker(tmp_path, cancel_key):
    server, state, _ = _editor_server(tmp_path)
    original = _pose((0.1, 0.2, 0.3), (0.02, 0.0, 0.01))
    save_model_frame_offset(state.spec.model_frame_offset_path, original)
    server._load_model_frame_offsets()
    saved_bytes = Path(state.spec.model_frame_offset_path).read_bytes()
    seed = state.estimator.pose_last
    server._handle_key(ord("m"))
    initial_pose = server.registration_interaction.T_view_object.copy()
    server._handle_key(ord("d"))
    server._handle_key(ord("l"))
    server._handle_key(ord("0"))
    np.testing.assert_allclose(server.registration_interaction.T_view_object, initial_pose)
    server._handle_key(ord("d"))
    server._handle_key(cancel_key)
    assert server.registration_interaction is None
    assert not server.shutdown
    assert state.estimator.pose_last is seed
    np.testing.assert_array_equal(state.T_fp_object, original)
    assert Path(state.spec.model_frame_offset_path).read_bytes() == saved_bytes


def test_failed_save_retains_draft_active_offset_and_cached_packets(tmp_path, monkeypatch):
    server, state, _ = _editor_server(tmp_path)
    save_model_frame_offset(state.spec.model_frame_offset_path, state.T_fp_object)
    before = Path(state.spec.model_frame_offset_path).read_bytes()
    packet = server._make_pose_packet(12.0, 0.0)
    server.pose_cache.add(packet)
    server._handle_key(ord("m"))
    server._handle_key(ord("d"))
    interaction = server.registration_interaction

    def fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(server_module, "save_model_frame_offset", fail)
    server._handle_key(13)
    assert server.registration_interaction is interaction
    assert "disk full" in server.ui_message
    assert server.pose_cache.match(12.0, 0.005) is packet
    assert Path(state.spec.model_frame_offset_path).read_bytes() == before
    np.testing.assert_array_equal(state.T_fp_object, np.eye(4))


def test_editor_key_routing_locks_selection_and_initialization(tmp_path):
    server, state, _ = _editor_server(tmp_path)
    second = ObjectState(spec=replace(state.spec, object_id="second"))
    server.object_states.append(second)
    server._handle_key(ord("m"))
    interaction = server.registration_interaction
    seed = state.estimator.pose_last
    camera = interaction.camera_name
    server._handle_key(ord("]"))
    assert interaction.translation_step_m == pytest.approx(0.01)
    assert interaction.rotation_step_rad == pytest.approx(np.deg2rad(5))
    server._handle_key(ord("["))
    assert interaction.translation_step_m == pytest.approx(0.001)
    for key in (ord(","), ord("."), ord("r"), ord("a"), ord("i")):
        server._handle_key(key)
    server._start_init_interaction()
    assert server.current_object_idx == 0
    assert interaction.camera_name == camera
    assert server.init_interaction is None
    assert state.estimator.pose_last is seed
    assert state.valid
    assert second.state == "uninitialized"
    np.testing.assert_array_equal(second.T_fp_object, np.eye(4))
    server._handle_key(13)
    np.testing.assert_array_equal(second.T_fp_object, np.eye(4))


@pytest.mark.parametrize("missing", ["validity", "frame", "mesh"])
def test_editor_requires_a_valid_pose_and_matching_frame(tmp_path, missing):
    server, state, _ = _editor_server(tmp_path)
    if missing == "validity":
        state.pose_validity = {"valid": False}
    elif missing == "frame":
        state.last_accepted_frame = None
    else:
        state.mesh = None
    server._handle_key(ord("m"))
    assert server.registration_interaction is None
    assert "needs a valid" in server.ui_message


def test_registration_survives_reset_and_successful_reinitialization(tmp_path, monkeypatch):
    server, state, frame = _editor_server(tmp_path)
    state.T_fp_object = _pose((0.0, np.pi, 0.0), (0.02, 0.0, 0.0))
    offset = state.T_fp_object.copy()
    server._handle_key(ord("r"))
    np.testing.assert_array_equal(state.T_fp_object, offset)
    assert state.last_accepted_frame is None
    estimator = _Estimator(_pose(translation=(0.0, 0.0, 1.0)))
    estimator.register = lambda **_kwargs: estimator.result
    server.factory = SimpleNamespace(create_estimator=lambda *_args: estimator)
    server.est_refine_iter = 5
    server._draw_status = lambda _frame: None
    monkeypatch.setattr(server_module.cv2, "waitKey", lambda _delay: None)
    server.init_interaction = InitInteraction(
        object_idx=0, camera_name="camera_lowfield_1", frame=frame,
        mode="mask", mask=np.ones((2, 2), dtype=np.uint8),
        bbox_xywh=(0, 0, 2, 2), mask_source="sam_hq",
    )
    assert server._complete_initialization()
    assert state.last_accepted_frame is frame
    np.testing.assert_array_equal(state.T_fp_object, offset)
    np.testing.assert_array_equal(state.last_T_camera_object, estimator.result)
    np.testing.assert_allclose(server._state_object_payload(state)["T_world_object"], state.last_T_world_object @ offset)


def test_editor_render_uses_frozen_frame_and_draft_pose(tmp_path, monkeypatch):
    server, state, frame = _editor_server(tmp_path)
    frame = replace(frame, color_by_camera={name: np.full((160, 240, 3), 40, dtype=np.uint8)
                                           for name in frame.color_by_camera})
    state.last_accepted_frame = frame
    server._handle_key(ord("m"))
    server._handle_key(ord("d"))
    draft = server.registration_interaction.T_view_object.copy()
    next_frame = replace(frame, sync_timestamp=13.0, color_by_camera={
        name: np.full((160, 240, 3), 200, dtype=np.uint8) for name in frame.color_by_camera
    })
    server.latest_frame = next_frame
    server._track_object(state, next_frame)
    displayed = []
    axes = []
    monkeypatch.setattr(server_module.cv2, "imshow", lambda _name, image: displayed.append(image.copy()))
    monkeypatch.setattr(server_module, "draw_pose_frame_bgr", lambda image, pose, *_args, **_kwargs: axes.append(pose.copy()))
    server._redraw_ui()
    np.testing.assert_array_equal(displayed[0][-1, -1], [40, 40, 40])
    np.testing.assert_array_equal(axes[0], draft)


def test_rejected_frame_preserves_editor_snapshot_source(tmp_path):
    server, state, frame = _editor_server(tmp_path)
    state.estimator.result[0, 3] = 0.2
    server._track_object(state, replace(frame, sync_timestamp=13.0))
    assert not state.valid
    assert state.last_accepted_frame is frame
