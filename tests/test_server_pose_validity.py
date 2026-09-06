from types import SimpleNamespace
import sys
import types

import msgpack
import numpy as np
import pytest

from server import InitInteraction, ObjectPoseServer, SyncFrameCache
from test_server_camera_frames import (
    _server_with_camera_frames, _tracking_state, _translation, _two_camera_frame,
)


class _Seed:
    def __init__(self, pose):
        self.pose = pose.copy()

    def detach(self):
        return self

    def clone(self):
        return _Seed(self.pose)


class _Estimator:
    def __init__(self, pose):
        self.result = pose.copy()
        self.pose_last = _Seed(pose)

    def track_one(self, **_kwargs):
        self.pose_last = _Seed(self.result)
        return self.result.copy()

    def track_one_w_spec_last_pose(self, **kwargs):
        return self.track_one(**kwargs)


def _setup():
    pose = _translation(0.0, 0.0, 1.0)
    state = _tracking_state(pose)
    state.estimator = _Estimator(pose)
    server = _server_with_camera_frames(state)
    server.activate_2d_tracker = False
    server.activate_kalman_filter = False
    return server, state, _two_camera_frame()


def test_transform_presence_alone_is_not_valid():
    _, state, _ = _setup()
    state.pose_validity = {}
    assert not state.valid
    state.pose_validity = {"valid": True}
    state.last_T_world_object[0, 0] = np.nan
    assert not state.valid


def test_rejection_preserves_accepted_pose_and_seed_then_recovers():
    server, state, frame = _setup()
    accepted_pose = state.last_T_camera_object.copy()
    accepted_world = state.last_T_world_object.copy()
    state.estimator.result[0, 3] = 0.2

    server._track_object(state, frame)

    assert not state.valid
    assert state.can_track
    payload = server._state_object_payload(state)
    assert payload["state"] == "rejected"
    assert payload["meta"]["pose_validity"]["reason"] == "translation_jump"
    np.testing.assert_array_equal(state.last_T_camera_object, accepted_pose)
    np.testing.assert_array_equal(state.last_T_world_object, accepted_world)
    np.testing.assert_array_equal(state.estimator.pose_last.pose, accepted_pose)

    state.estimator.result[0, 3] = 0.01
    server._track_object(state, frame)
    assert state.valid
    assert state.error_msg is None
    assert state.last_T_camera_object[0, 3] == 0.01


def test_rejection_restores_kalman_arrays_even_after_in_place_changes():
    server, state, frame = _setup()
    state.kf_mean = np.zeros(12)
    state.kf_covariance = np.eye(12)
    original_track = state.estimator.track_one

    def changed_filter(**kwargs):
        state.kf_mean[:] = 100.0
        state.kf_covariance[:] = 100.0
        return original_track(**kwargs)

    state.estimator.track_one = changed_filter
    state.estimator.result[0, 3] = 0.2
    server._track_object(state, frame)
    np.testing.assert_array_equal(state.kf_mean, np.zeros(12))
    np.testing.assert_array_equal(state.kf_covariance, np.eye(12))


@pytest.mark.parametrize("mask_timestamp", [None, 11.0])
def test_tracking_rejects_missing_or_stale_masks(mask_timestamp):
    server, state, frame = _setup()
    server.activate_2d_tracker = True
    state.last_mask = np.ones((2, 2), dtype=np.uint8)
    state.last_mask_synced_ts = mask_timestamp
    # Missing tracker must not let the previous mask validate this frame.
    server._track_object(state, frame)
    assert not state.valid
    assert state.pose_validity["reason"] == "mask_missing"


@pytest.mark.parametrize("failure", ["jump", "nan", "depth"])
def test_isolated_requests_validate_candidates_and_preserve_continuous_state(failure):
    server, state, frame = _setup()
    original_seed = state.estimator.pose_last
    original_validity = state.pose_validity.copy()
    if failure == "jump":
        state.estimator.result[0, 3] = 0.2
    elif failure == "nan":
        state.estimator.result[0, 0] = np.nan
    else:
        server._render_validation_depth = lambda *_args: np.full((2, 2), 1.1)

    payload = server._process_isolated_request(frame)["object_poses"][0]

    assert not payload["valid"]
    assert payload["state"] == "rejected"
    assert payload["T_camera_object"] is None
    assert payload["T_world_object"] is None
    assert state.estimator.pose_last is original_seed
    assert state.pose_validity == original_validity
    assert state.valid


def test_isolated_request_does_not_reuse_mask_from_a_different_frame():
    server, state, frame = _setup()
    state.last_mask = np.zeros((2, 2), dtype=np.uint8)
    state.last_mask_synced_ts = 13.0
    payload = server._process_isolated_request(frame)["object_poses"][0]
    assert payload["valid"]
    assert payload["meta"]["pose_validity"]["mask_state"] == "unavailable"
    assert payload["tracking_mask"] is None


def test_all_rejected_objects_still_serve_normal_packets_for_fallback_and_recovery():
    server, state, frame = _setup()
    state.pose_validity = {"valid": False, "reason": "translation_jump"}
    state.estimator.result[0, 3] = 0.2
    server.frame_cache = SyncFrameCache(3)
    server.frame_cache.add(frame)
    server.timestamp_tolerance = 0.005
    server.last_tracking_timestamp = 11.0
    server._publish_packet = lambda _packet: None
    message = msgpack.packb({"synced_ts": frame.sync_timestamp})

    reply = server._handle_pose_request(message)

    assert reply["ok"]
    assert not reply["object_poses"][0]["valid"]
    assert reply["object_poses"][0]["state"] == "rejected"


def test_validation_uses_observation_camera_before_expression_transform():
    server, state, frame = _setup()
    # Top-camera depth intentionally disagrees; lowfield is the observation.
    frame.depth_by_camera["camera_top"][:] = 5.0
    server._track_object(state, frame)
    assert state.valid
    assert server._state_object_payload(state)["pose_camera"] == "camera_top"


def test_renderer_failure_rejects_pose(monkeypatch):
    server, state, frame = _setup()

    def broken_render(*_args):
        raise RuntimeError("render failed")

    monkeypatch.setattr(server, "_render_validation_depth", broken_render)
    server._track_object(state, frame)
    assert not state.valid
    assert state.pose_validity["reason"] == "validation_render_failed"


def test_initialization_rejects_unsupported_pose(monkeypatch):
    server, state, frame = _setup()
    state.reset_runtime()
    estimator = _Estimator(_translation(0.0, 0.0, 1.0))
    estimator.register = lambda **_kwargs: estimator.result
    server.factory = SimpleNamespace(create_estimator=lambda *_args: estimator)
    state.mesh = object()
    server.est_refine_iter = 5
    server._draw_status = lambda _frame: None
    server._render_validation_depth = lambda *_args: np.full((2, 2), 2.0)
    monkeypatch.setattr("server.cv2.waitKey", lambda _delay: None)
    server.init_interaction = InitInteraction(
        object_idx=0, camera_name="camera_lowfield_1", frame=frame,
        mode="mask", mask=np.ones((2, 2), dtype=np.uint8),
        bbox_xywh=(0, 0, 2, 2), mask_source="sam_hq",
    )

    assert not server._complete_initialization()
    assert not state.valid
    assert "insufficient_depth_agreement" in state.error_msg
    assert server.init_interaction.mode == "mask"


def test_validation_render_converts_original_mesh_pose_to_centered_mesh(monkeypatch):
    server, state, frame = _setup()
    pose = state.last_T_camera_object.copy()
    pose[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]

    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

        def reshape(self, *shape):
            return Tensor(self.value.reshape(*shape))

        def __getitem__(self, key):
            return Tensor(self.value[key])

    from contextlib import nullcontext
    import server as server_module
    monkeypatch.setattr(server_module.torch, "as_tensor", lambda value, **_kwargs: Tensor(value), raising=False)
    monkeypatch.setattr(server_module.torch, "float32", object(), raising=False)
    monkeypatch.setattr(server_module.torch, "no_grad", nullcontext, raising=False)
    to_centered = _translation(-0.1, -0.2, -0.3)
    state.estimator.get_tf_to_centered_mesh = lambda: Tensor(to_centered)
    state.estimator.mesh_tensors = {"pos": SimpleNamespace(device="cuda:0")}
    state.estimator.glctx = object()
    calls = []

    def render(**kwargs):
        calls.append(kwargs)
        return None, Tensor(np.ones((1, 2, 2))), None

    estimator_module = types.ModuleType("FoundationPose.estimater")
    estimator_module.nvdiffrast_render = render
    monkeypatch.setitem(sys.modules, "FoundationPose.estimater", estimator_module)
    rendered = ObjectPoseServer._render_validation_depth(server, state, pose, frame)

    expected_centered = pose @ _translation(0.1, 0.2, 0.3)
    np.testing.assert_allclose(calls[0]["ob_in_cams"].numpy()[0], expected_centered)
    assert calls[0]["K"] is frame.cam_intr_by_camera["camera_lowfield_1"]
    assert calls[0]["mesh_tensors"] is state.estimator.mesh_tensors
    assert rendered.shape == (2, 2)
