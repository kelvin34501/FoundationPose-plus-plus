import numpy as np

import server as server_module
from server import ObjectPoseServer, ObjectSpec, ObjectState, PosePacketCache, SyncFrame
from world_calibration import express_world_pose_in_camera
from pose_validity import PoseValidityConfig


def _translation(x, y, z):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = (x, y, z)
    return transform


def _server_with_camera_frames(state):
    server = ObjectPoseServer.__new__(ObjectPoseServer)
    server.observation_camera = None
    server.expression_camera = "camera_top"
    server.T_world_camera_map = {
        "camera_top": _translation(0.5, -0.2, 0.7),
        "camera_lowfield_1": _translation(-0.1, 0.3, 0.2),
    }
    server.object_states = [state]
    server.current_object_idx = 0
    server.latest_frame = None
    server.selected_view_camera_by_object = {}
    server.ui_message = None
    server.track_refine_iter = 5
    server.pose_cache = PosePacketCache(3)
    server.pose_validity_config = PoseValidityConfig(min_pixels=1)
    server._render_validation_depth = lambda state, _pose, frame: frame.depth_by_camera[state.source_camera]
    return server


def _tracking_state(T_lowfield_object):
    T_world_lowfield = _translation(-0.1, 0.3, 0.2)
    return ObjectState(
        spec=ObjectSpec("mug", "/unused/mug.obj"),
        state="tracking",
        source_camera="camera_lowfield_1",
        last_T_camera_object=T_lowfield_object,
        last_T_world_object=T_world_lowfield @ T_lowfield_object,
        pose_validity={"valid": True, "reason": "ok"},
    )


def _two_camera_frame():
    return SyncFrame(
        sync_timestamp=12.0,
        color_by_camera={
            "camera_top": np.zeros((2, 2, 3), dtype=np.uint8),
            "camera_lowfield_1": np.zeros((2, 2, 3), dtype=np.uint8),
        },
        depth_by_camera={
            "camera_top": np.ones((2, 2), dtype=np.float32),
            "camera_lowfield_1": np.ones((2, 2), dtype=np.float32),
        },
        cam_intr_by_camera={
            "camera_top": np.eye(3, dtype=np.float64),
            "camera_lowfield_1": np.eye(3, dtype=np.float64),
        },
    )


def test_normal_packet_observes_lowfield_and_expresses_top():
    T_lowfield_object = _translation(0.05, -0.02, 0.8)
    state = _tracking_state(T_lowfield_object)
    server = _server_with_camera_frames(state)

    payload = server._state_object_payload(state)

    expected = express_world_pose_in_camera(
        server.T_world_camera_map["camera_top"],
        state.last_T_world_object,
    )
    assert payload["source_camera"] == "camera_lowfield_1"
    assert payload["pose_camera"] == "camera_top"
    np.testing.assert_allclose(payload["T_camera_object"], expected)


def test_top_observation_and_expression_preserve_the_estimator_pose():
    T_top_object = _translation(0.2, -0.1, 0.65)
    state = ObjectState(
        spec=ObjectSpec("mug", "/unused/mug.obj"),
        state="tracking",
        source_camera="camera_top",
        last_T_camera_object=T_top_object,
        last_T_world_object=_translation(0.5, -0.2, 0.7) @ T_top_object,
        pose_validity={"valid": True, "reason": "ok"},
    )
    server = _server_with_camera_frames(state)

    payload = server._state_object_payload(state)

    assert payload["source_camera"] == "camera_top"
    assert payload["pose_camera"] == "camera_top"
    np.testing.assert_array_equal(payload["T_camera_object"], T_top_object)


def test_observation_camera_sets_initial_view_and_initialization_camera():
    state = ObjectState(
        spec=ObjectSpec("mug", "/unused/mug.obj", camera_name="camera_top"),
    )
    server = _server_with_camera_frames(state)
    server.observation_camera = "camera_lowfield_1"
    frame = _two_camera_frame()

    assert server._preview_camera(frame) == "camera_lowfield_1"
    assert server._initialization_camera(frame) == "camera_lowfield_1"


def test_view_can_switch_after_initialization_without_changing_tracking_or_output():
    T_lowfield_object = _translation(0.05, -0.02, 0.8)
    state = _tracking_state(T_lowfield_object)
    server = _server_with_camera_frames(state)
    server.observation_camera = "camera_lowfield_1"
    server.latest_frame = _two_camera_frame()

    assert server._preview_camera(server.latest_frame) == "camera_lowfield_1"
    payload_before = server._state_object_payload(state)

    server._cycle_display_camera(1)

    assert server._preview_camera(server.latest_frame) == "camera_top"
    assert state.source_camera == "camera_lowfield_1"
    assert state.valid
    assert server.expression_camera == "camera_top"
    assert "tracking remains on camera_lowfield_1" in server.ui_message
    payload_after = server._state_object_payload(state)
    assert payload_after["source_camera"] == payload_before["source_camera"]
    assert payload_after["pose_camera"] == payload_before["pose_camera"]
    np.testing.assert_allclose(
        payload_after["T_camera_object"],
        payload_before["T_camera_object"],
    )


def test_pose_can_be_expressed_in_selected_display_camera():
    T_lowfield_object = _translation(0.05, -0.02, 0.8)
    state = _tracking_state(T_lowfield_object)
    server = _server_with_camera_frames(state)

    expected_top = express_world_pose_in_camera(
        server.T_world_camera_map["camera_top"],
        state.last_T_world_object,
    )
    np.testing.assert_allclose(
        server._pose_for_camera(state, "camera_top"),
        expected_top,
    )
    np.testing.assert_array_equal(
        server._pose_for_camera(state, "camera_lowfield_1"),
        state.last_T_camera_object,
    )


def test_status_draws_transformed_pose_in_non_source_view():
    state = _tracking_state(_translation(0.05, -0.02, 0.8))
    server = _server_with_camera_frames(state)
    server.init_interaction = None
    server.display_scale = 1.0
    server.selected_view_camera_by_object[0] = "camera_top"
    frame = _two_camera_frame()
    drawn_poses = []

    original_draw = server_module.draw_pose_frame_bgr
    original_imshow = server_module.cv2.imshow

    def record_pose(image, T_camera_object, _K, *, axis_scale, thickness):
        drawn_poses.append((T_camera_object, axis_scale, thickness))
        return image

    try:
        server_module.draw_pose_frame_bgr = record_pose
        server_module.cv2.imshow = lambda *_args, **_kwargs: None
        server._draw_status(frame)
    finally:
        server_module.draw_pose_frame_bgr = original_draw
        server_module.cv2.imshow = original_imshow

    expected_top = express_world_pose_in_camera(
        server.T_world_camera_map["camera_top"],
        state.last_T_world_object,
    )
    assert len(drawn_poses) == 1
    np.testing.assert_allclose(drawn_poses[0][0], expected_top)


def test_pinned_observation_camera_does_not_pin_the_display_view():
    state = ObjectState(spec=ObjectSpec("mug", "/unused/mug.obj"))
    server = _server_with_camera_frames(state)
    server.observation_camera = "camera_lowfield_1"
    server.latest_frame = _two_camera_frame()

    server._cycle_display_camera(1)

    assert server._preview_camera(server.latest_frame) == "camera_top"
    assert server._initialization_camera(server.latest_frame) == "camera_lowfield_1"
    assert "initialization uses camera_lowfield_1" in server.ui_message


def test_missing_selected_view_falls_back_without_changing_the_preference():
    state = _tracking_state(_translation(0.05, -0.02, 0.8))
    server = _server_with_camera_frames(state)
    server.selected_view_camera_by_object[0] = "camera_top"
    lowfield_only = SyncFrame(
        sync_timestamp=13.0,
        color_by_camera={"camera_lowfield_1": np.zeros((2, 2, 3), dtype=np.uint8)},
        depth_by_camera={"camera_lowfield_1": np.ones((2, 2), dtype=np.float32)},
        cam_intr_by_camera={"camera_lowfield_1": np.eye(3, dtype=np.float64)},
    )

    assert server._preview_camera(lowfield_only) == "camera_lowfield_1"
    assert server.selected_view_camera_by_object[0] == "camera_top"
    assert state.source_camera == "camera_lowfield_1"


class _PoseSeed:

    def detach(self):
        return self

    def clone(self):
        return self


class _Estimator:

    def __init__(self, result):
        self.pose_last = _PoseSeed()
        self.result = result

    def track_one_w_spec_last_pose(self, **_kwargs):
        return self.result


def test_isolated_packet_uses_the_same_expression_camera():
    T_lowfield_object = _translation(0.08, 0.01, 0.75)
    state = _tracking_state(T_lowfield_object)
    state.estimator = _Estimator(T_lowfield_object)
    server = _server_with_camera_frames(state)
    frame = SyncFrame(
        sync_timestamp=12.0,
        color_by_camera={"camera_lowfield_1": np.zeros((2, 2, 3), dtype=np.uint8)},
        depth_by_camera={"camera_lowfield_1": np.ones((2, 2), dtype=np.float32)},
        cam_intr_by_camera={"camera_lowfield_1": np.eye(3, dtype=np.float64)},
    )

    packet = server._process_isolated_request(frame)
    payload = packet["object_poses"][0]

    expected_world = server.T_world_camera_map["camera_lowfield_1"] @ T_lowfield_object
    expected_top = express_world_pose_in_camera(
        server.T_world_camera_map["camera_top"],
        expected_world,
    )
    assert payload["source_camera"] == "camera_lowfield_1"
    assert payload["pose_camera"] == "camera_top"
    np.testing.assert_allclose(payload["T_world_object"], expected_world)
    np.testing.assert_allclose(payload["T_camera_object"], expected_top)
