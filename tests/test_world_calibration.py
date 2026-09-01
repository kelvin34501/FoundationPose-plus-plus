import numpy as np

from world_calibration import express_world_pose_in_camera


def _transform(rotation, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def test_express_lowfield_pose_in_top_camera():
    T_world_top = _transform(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [0.4, -0.2, 0.7],
    )
    T_world_lowfield = _transform(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [-0.1, 0.3, 0.2],
    )
    T_lowfield_object = _transform(np.eye(3), [0.05, -0.02, 0.8])
    T_world_object = T_world_lowfield @ T_lowfield_object

    actual = express_world_pose_in_camera(T_world_top, T_world_object)
    expected = np.linalg.inv(T_world_top) @ T_world_lowfield @ T_lowfield_object

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_express_pose_in_its_observation_camera_is_identity_conversion():
    T_world_camera = _transform(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        [0.6, 0.1, -0.3],
    )
    T_camera_object = _transform(np.eye(3), [0.2, -0.4, 1.1])

    actual = express_world_pose_in_camera(
        T_world_camera,
        T_world_camera @ T_camera_object,
    )

    np.testing.assert_allclose(actual, T_camera_object, atol=1e-12)
