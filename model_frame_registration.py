"""Persistent correction of FoundationPose's returned original-mesh frame.

T_fp_object maps corrected object coordinates into FoundationPose mesh
coordinates. Output poses are T_camera_fp @ T_fp_object; estimator seeds stay
in FoundationPose's own (internally centered) frame.
"""
from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from dev_fn.compat.pickle_numpy_compat import load_pkl_numpy2_compat
from dev_fn.transform.rotation_np import rotvec_to_rotmat_np
from dev_fn.transform.transform_np import inv_transf_np
from pose_validity import is_rigid_transform

PathLike = Union[str, Path]
REGISTRATION_ROOT = Path("/home/pjlab/dex_manip/data/obj_platform_reg")


def default_model_frame_offset_path(object_id: str) -> Path:
    # Object IDs become a directory name; spaces and '#' are valid.
    if not object_id or object_id in (".", "..") or "/" in object_id or "\\" in object_id:
        raise ValueError("An object ID containing a path requires model_frame_offset_path")
    return REGISTRATION_ROOT / object_id / "T_fp_object.pkl"


def _validated_transform(transform: np.ndarray) -> np.ndarray:
    if not is_rigid_transform(transform):
        raise ValueError("Model-frame registration must be a finite rigid 4x4 transform")
    return np.array(transform, dtype=np.float64, copy=True)


def load_model_frame_offset(path: PathLike) -> Tuple[np.ndarray, bool]:
    """Load one transform, or identity if no registration has been saved yet."""
    path = Path(path).expanduser()
    try:
        transform = load_pkl_numpy2_compat(str(path))
    except FileNotFoundError:
        return np.eye(4, dtype=np.float64), False
    return _validated_transform(transform), True


def save_model_frame_offset(path: PathLike, transform: np.ndarray) -> None:
    """Atomically replace the saved transform; callers activate it on success."""
    transform = _validated_transform(transform)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            pickle.dump(transform, stream, protocol=4)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def apply_model_frame_offset(pose: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=np.float64) @ np.asarray(offset, dtype=np.float64)


def derive_model_frame_offset(raw_pose: np.ndarray, adjusted_pose: np.ndarray) -> np.ndarray:
    raw_pose = _validated_transform(raw_pose)
    adjusted_pose = _validated_transform(adjusted_pose)
    return _validated_transform(inv_transf_np(raw_pose) @ adjusted_pose)


def _validate_nudge(axis: int, amount: float) -> None:
    if axis not in (0, 1, 2) or not np.isfinite(amount):
        raise ValueError("Pose adjustment requires axis 0, 1, or 2 and a finite step")


def translate_pose_in_object(pose: np.ndarray, axis: int, distance_m: float) -> np.ndarray:
    """Move the origin along an axis of the current corrected model frame."""
    _validate_nudge(axis, distance_m)
    result = _validated_transform(pose)
    result[:3, 3] += result[:3, axis] * distance_m
    return _validated_transform(result)


def rotate_pose_in_object(pose: np.ndarray, axis: int, angle_rad: float) -> np.ndarray:
    _validate_nudge(axis, angle_rad)
    result = _validated_transform(pose)
    delta = np.zeros(3, dtype=np.float64)
    delta[axis] = angle_rad
    result[:3, :3] = result[:3, :3] @ rotvec_to_rotmat_np(delta)
    return _validated_transform(result)
