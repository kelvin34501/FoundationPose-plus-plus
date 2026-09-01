"""Discover fixed-camera world calibration for FoundationPose++."""

from __future__ import annotations

import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


_LOGGER = logging.getLogger("world_calibration")
DEX_MANIP_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_ROOT = DEX_MANIP_ROOT / "common" / "calib_camera" / "checkerboard_calib"
EASYROBOT_SRC = DEX_MANIP_ROOT / "easyrobot_custom" / "src"
if str(EASYROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(EASYROBOT_SRC))

from dev_fn.transform.transform_np import inv_transf_np  # noqa: E402


@dataclass(frozen=True)
class FixedCameraCalibration:
    T_world_camera_by_name: Dict[str, np.ndarray]
    automatic: bool
    session_dir: Path
    world_dir: Path
    camera_pickles: Dict[str, Path]


def express_world_pose_in_camera(
    T_world_camera: np.ndarray,
    T_world_object: np.ndarray,
) -> np.ndarray:
    """Express a world-frame object pose in a fixed camera frame."""
    return inv_transf_np(np.asarray(T_world_camera)) @ np.asarray(T_world_object)


def _select_session(world_calib: Optional[str]):
    if world_calib is None:
        sessions = sorted(path for path in CALIBRATION_ROOT.glob("calib__*") if path.is_dir())
        if not sessions:
            raise FileNotFoundError(f"No calibration sessions found under {CALIBRATION_ROOT}")
        automatic = True
        session_dir = sessions[-1].resolve()
        world_dir = session_dir / "world"
    else:
        automatic = False
        selected = Path(world_calib).expanduser()
        if not selected.is_absolute():
            selected = DEX_MANIP_ROOT / selected
        selected = selected.resolve()
        if selected.name == "world":
            session_dir = selected.parent
            world_dir = selected
        else:
            session_dir = selected
            world_dir = selected / "world" if (selected / "world").is_dir() else selected

    if not session_dir.is_dir():
        raise FileNotFoundError(f"Calibration session directory not found: {session_dir}")
    if not world_dir.is_dir():
        raise FileNotFoundError(f"World calibration directory not found: {world_dir}")
    return automatic, session_dir.resolve(), world_dir.resolve()


def _load_transform(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required calibration transform not found: {path}")
    with path.open("rb") as stream:
        transform = np.asarray(pickle.load(stream), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} in {path} must be a finite 4x4 matrix, got {transform.shape}")
    return transform


def load_fixed_camera_calibrations(
    world_calib: Optional[str],
    camera_names: List[str],
) -> FixedCameraCalibration:
    """Load explicit world-camera pickles for every configured fixed camera."""
    automatic, session_dir, world_dir = _select_session(world_calib)
    transforms: Dict[str, np.ndarray] = {}
    paths: Dict[str, Path] = {}
    for camera_name in camera_names:
        path = (world_dir / f"{camera_name}_T_world_cam.pkl").resolve()
        transforms[camera_name] = _load_transform(path, f"T_world_{camera_name}")
        paths[camera_name] = path

    _LOGGER.info("World calibration selection: %s", "automatic" if automatic else "explicit")
    _LOGGER.info("World calibration session: %s", session_dir)
    for camera_name in camera_names:
        _LOGGER.info("World calibration camera pickle [%s]: %s", camera_name, paths[camera_name])

    return FixedCameraCalibration(
        T_world_camera_by_name=transforms,
        automatic=automatic,
        session_dir=session_dir,
        world_dir=world_dir,
        camera_pickles=paths,
    )
