import base64
import pickle

import numpy as np
import pytest

import model_frame_registration as registration
from dev_fn.transform.rotation_np import rotvec_to_rotmat_np
from dev_fn.transform.transform_np import inv_transf_np


def _pose(rotvec=(0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0)):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotvec_to_rotmat_np(np.asarray(rotvec, dtype=np.float64))
    pose[:3, 3] = translation
    return pose


def test_offset_recovers_manual_anchor_and_preserves_rigid_motion():
    manual = _pose((0.6, -0.2, 0.3), (0.1, -0.2, 0.8))
    gauge = _pose((0.0, np.pi, 0.0), (0.03, 0.01, -0.02))
    motion = _pose((0.2, 0.1, -0.4), (-0.1, 0.2, 0.0))
    raw = manual @ gauge
    raw_before = raw.copy()
    offset = registration.derive_model_frame_offset(raw, manual)

    np.testing.assert_allclose(offset, inv_transf_np(gauge), atol=1e-12)
    np.testing.assert_allclose(registration.apply_model_frame_offset(raw, offset), manual, atol=1e-12)
    np.testing.assert_allclose(
        registration.apply_model_frame_offset(motion @ raw, offset), motion @ manual, atol=1e-12,
    )
    np.testing.assert_array_equal(raw, raw_before)


def test_offset_origin_rotates_with_raw_pose():
    offset = _pose(translation=(0.1, 0.0, 0.0))
    raw = _pose((0.0, 0.0, np.pi / 2), (0.0, 0.0, 1.0))
    np.testing.assert_allclose(
        registration.apply_model_frame_offset(raw, offset)[:3, 3], [0.0, 0.1, 1.0], atol=1e-12,
    )
    np.testing.assert_array_equal(registration.apply_model_frame_offset(raw, np.eye(4)), raw)


def test_nudges_use_local_translation_and_rotation_without_mutation():
    pose = _pose((0.0, 0.0, np.pi / 2), (0.1, 0.2, 0.8))
    original = pose.copy()
    translated = registration.translate_pose_in_object(pose, 0, 0.01)
    rotated = registration.rotate_pose_in_object(pose, 2, np.pi / 2)
    np.testing.assert_allclose(translated[:3, 3], [0.1, 0.21, 0.8])
    np.testing.assert_array_equal(translated[:3, :3], pose[:3, :3])
    np.testing.assert_allclose(rotated, pose @ _pose((0.0, 0.0, np.pi / 2)), atol=1e-12)
    np.testing.assert_array_equal(pose, original)


def test_missing_file_and_repeated_atomic_saves(tmp_path):
    path = tmp_path / "object" / "T_fp_object.pkl"
    offset, loaded = registration.load_model_frame_offset(path)
    assert not loaded
    assert not path.parent.exists()
    np.testing.assert_array_equal(offset, np.eye(4))
    for transform in (_pose((0.1, 0.2, 0.0)), _pose((0.0, 0.0, 1.0), (0.02, -0.03, 0.01))):
        registration.save_model_frame_offset(path, transform)
        offset, loaded = registration.load_model_frame_offset(path)
        assert loaded
        assert offset.dtype == np.float64
        np.testing.assert_allclose(offset, transform)
        with path.open("rb") as stream:
            assert isinstance(pickle.load(stream), np.ndarray)
    assert list(path.parent.iterdir()) == [path]


def test_numpy_2_pickle_is_supported(tmp_path):
    # Fixed NumPy 2.2.6/protocol-5 fixture also exercises the loader on NumPy 1.
    data = base64.b64decode(
        "gAWV9gAAAAAAAACME251bXB5Ll9jb3JlLm51bWVyaWOUjAtfZnJvbWJ1ZmZlcpSTlCiWgAAAAAAAAAAAAAAAAADw"
        "PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8D8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAPA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADwP5SMBW51bXB5lIwF"
        "ZHR5cGWUk5SMAmY4lImIh5RSlChLA4wBPJROTk5K/////0r/////SwB0lGJLBEsEhpSMAUOUdJRSlC4="
    )
    path = tmp_path / "numpy2.pkl"
    path.write_bytes(data)
    offset, loaded = registration.load_model_frame_offset(path)
    assert loaded
    np.testing.assert_array_equal(offset, np.eye(4))


@pytest.mark.parametrize("bad", [
    np.eye(3), np.full((4, 4), np.nan), np.full((4, 4), np.inf),
    np.diag([-1.0, 1.0, 1.0, 1.0]), np.diag([2.0, 1.0, 1.0, 1.0]),
    np.diag([1.0, 1.0, 1.0, 0.0]),
])
def test_invalid_registration_is_rejected_on_load_and_save(tmp_path, bad):
    path = tmp_path / "offset.pkl"
    with path.open("wb") as stream:
        pickle.dump(bad, stream)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="rigid 4x4"):
        registration.load_model_frame_offset(path)
    with pytest.raises(ValueError, match="rigid 4x4"):
        registration.save_model_frame_offset(path, bad)
    assert path.read_bytes() == original


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_save_failure_preserves_existing_file_and_removes_temporary_file(tmp_path, monkeypatch, failure):
    path = tmp_path / "offset.pkl"
    registration.save_model_frame_offset(path, np.eye(4))
    original = path.read_bytes()

    def fail(*_args, **_kwargs):
        raise OSError("simulated save failure")

    if failure == "write":
        monkeypatch.setattr(registration.pickle, "dump", fail)
    else:
        monkeypatch.setattr(registration.os, "replace", fail)
    with pytest.raises(OSError, match="simulated"):
        registration.save_model_frame_offset(path, _pose(translation=(0.1, 0.0, 0.0)))
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_unreadable_file_does_not_fall_back_to_identity(tmp_path, monkeypatch):
    def denied(_path):
        raise PermissionError("cannot read registration")

    monkeypatch.setattr(registration, "load_pkl_numpy2_compat", denied)
    with pytest.raises(PermissionError):
        registration.load_model_frame_offset(tmp_path / "offset.pkl")


@pytest.mark.parametrize("axis,amount", [(3, 0.1), (-1, 0.1), (1, np.inf), (0, np.nan)])
def test_invalid_nudges_are_rejected(axis, amount):
    for nudge in (registration.translate_pose_in_object, registration.rotate_pose_in_object):
        with pytest.raises(ValueError):
            nudge(np.eye(4), axis, amount)
