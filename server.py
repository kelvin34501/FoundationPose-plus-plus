#!/usr/bin/env python3
"""Interactive Object Pose Estimation Server.

This server is intentionally started manually.  It subscribes to a SyncUnit
RGB-D stream, lets the operator initialize one or more objects from the latest
frame, tracks them with FoundationPose++, publishes world-frame object poses,
and serves higher-priority pose estimates for requested SyncUnit timestamps.
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import pickle
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

import cv2
import msgpack
import msgpack_numpy
import numpy as np
import torch
import trimesh
import yaml
import zmq
from scipy.spatial.transform import Rotation

# Keep heavy native libraries from oversubscribing CPU threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
cv2.setNumThreads(0)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
FOUNDATIONPOSE_DIR = os.path.join(THIS_DIR, "FoundationPose")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if FOUNDATIONPOSE_DIR not in sys.path:
    sys.path.append(FOUNDATIONPOSE_DIR)

from VOT import Cutie, Tracker_2D  # noqa: E402
from utils.kalman_filter_6d import KalmanFilter6D  # noqa: E402
from world_calibration import load_fixed_camera_calibrations  # noqa: E402

_logger = logging.getLogger("object_pose_server")
WINDOW_NAME = "object_pose_server"

# Preserve SAM request frames and masks for debugging. By default, exchange
# both images in memory so object initialization does not create files.
SAVE_SAM_INTERMEDIATE_FILES = False


class ThrottleFilter(logging.Filter):
    """同一 (logger, level, msg) 在 *throttle_seconds* 内只放行一次。

    WARNING 及以上级别永远放行，不节流。
    """

    def __init__(self, throttle_seconds: float = 1.0):
        super().__init__()
        self._throttle = throttle_seconds
        self._last: Dict[tuple, float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        key = (record.name, record.levelno, record.getMessage())
        now = time.time()
        if now - self._last.get(key, 0) < self._throttle:
            return False
        self._last[key] = now
        return True


def channel_name_to_endpoint(channel_name: str, ipc_prefix: str = "/dev/shm/hcc_demo") -> str:
    endpoint = channel_name
    if not endpoint.startswith("ipc://") and not endpoint.startswith("tcp://"):
        if os.path.splitext(os.path.basename(endpoint))[-1] != ".ipc":
            endpoint += ".ipc"
        endpoint = "ipc://" + os.path.join(ipc_prefix, endpoint)
    return endpoint


def ensure_ipc_dir(endpoint: str) -> None:
    if endpoint.startswith("ipc://"):
        os.makedirs(os.path.dirname(endpoint[6:]), exist_ok=True)


def parse_video_shape(shape: str) -> Tuple[int, int]:
    try:
        width_s, height_s = shape.lower().split("x", 1)
        return int(width_s), int(height_s)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Expected WIDTHxHEIGHT, got {shape!r}") from exc


def parse_camera_info(value: str) -> Dict[str, str]:
    """Parse ``serial=name,serial=name`` into a dict."""
    result: Dict[str, str] = {}
    if not value:
        return result
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError("camera_info must be comma-separated serial=name entries")
        serial, name = item.split("=", 1)
        result[serial.strip()] = name.strip()
    return result


def get_pose_xy_from_image_point(
    ob_in_cam: torch.Tensor,
    K: np.ndarray,
    x: float,
    y: float,
) -> Tuple[float, float]:
    if x < 0 or y < 0:
        return x, y
    pose = ob_in_cam[0].detach().cpu().numpy() if ob_in_cam.ndim == 3 else ob_in_cam.detach().cpu().numpy()
    z = pose[:3, 3][2]
    tx = (x - K[0, 2]) * z / K[0, 0]
    ty = (y - K[1, 2]) * z / K[1, 1]
    return float(tx), float(ty)


def adjust_pose_to_image_point(
    ob_in_cam: torch.Tensor,
    K: np.ndarray,
    x: float,
    y: float,
) -> torch.Tensor:
    is_batched = ob_in_cam.ndim == 3
    pose_batch = ob_in_cam if is_batched else ob_in_cam.unsqueeze(0)
    out = torch.eye(4, device=pose_batch.device, dtype=pose_batch.dtype).repeat(pose_batch.shape[0], 1, 1)
    for i in range(pose_batch.shape[0]):
        tx, ty = get_pose_xy_from_image_point(pose_batch[i], K, x, y)
        out[i, :3, :3] = pose_batch[i, :3, :3]
        out[i, :3, 3] = torch.tensor([tx, ty, pose_batch[i, 2, 3]], device=pose_batch.device, dtype=pose_batch.dtype)
    return out if is_batched else out[0]


def get_6d_pose_arr_from_mat(pose: Any) -> np.ndarray:
    pose_np = pose[0].detach().cpu().numpy() if torch.is_tensor(pose) and pose.ndim == 3 else (
        pose.detach().cpu().numpy() if torch.is_tensor(pose) else np.asarray(pose))
    xyz = pose_np[:3, 3]
    euler = Rotation.from_matrix(pose_np[:3, :3]).as_euler("xyz", degrees=False)
    return np.r_[xyz, euler]


def get_mat_from_6d_pose_arr(pose_arr: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("xyz", pose_arr[3:6], degrees=False).as_matrix()
    T[:3, 3] = pose_arr[:3]
    return T


def run_with_cpu_default_tensor_type(fn):
    """Run code that loads CPU checkpoints after FoundationPose sets CUDA defaults."""
    try:
        was_cuda_default = torch.Tensor().device.type == "cuda"
    except Exception:
        was_cuda_default = False

    torch.set_default_tensor_type(torch.FloatTensor)
    try:
        return fn()
    finally:
        if was_cuda_default:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)


def clear_hydra_if_initialized() -> None:
    try:
        from hydra.core.global_hydra import GlobalHydra

        hydra = GlobalHydra.instance()
        if hydra.is_initialized():
            hydra.clear()
    except ImportError:
        return


def draw_pose_frame_bgr(
    image_bgr: np.ndarray,
    T_camera_object: np.ndarray,
    K: np.ndarray,
    axis_scale: float,
    thickness: int = 3,
) -> np.ndarray:
    points_obj = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [axis_scale, 0.0, 0.0, 1.0],
            [0.0, axis_scale, 0.0, 1.0],
            [0.0, 0.0, axis_scale, 1.0],
        ],
        dtype=np.float64,
    )
    points_cam = (np.asarray(T_camera_object, dtype=np.float64) @ points_obj.T).T[:, :3]
    if np.any(points_cam[:, 2] <= 1e-6):
        return image_bgr
    uv = (np.asarray(K, dtype=np.float64) @ points_cam.T).T
    uv = np.round(uv[:, :2] / uv[:, 2:3]).astype(int)
    origin = tuple(uv[0].tolist())
    cv2.arrowedLine(image_bgr, origin, tuple(uv[1].tolist()), (0, 0, 255), thickness, cv2.LINE_AA, tipLength=0.08)
    cv2.arrowedLine(image_bgr, origin, tuple(uv[2].tolist()), (0, 255, 0), thickness, cv2.LINE_AA, tipLength=0.08)
    cv2.arrowedLine(image_bgr, origin, tuple(uv[3].tolist()), (255, 0, 0), thickness, cv2.LINE_AA, tipLength=0.08)
    return image_bgr


@dataclass
class SyncFrame:
    sync_timestamp: float
    color_by_camera: Dict[str, np.ndarray]
    depth_by_camera: Dict[str, np.ndarray]
    cam_intr_by_camera: Dict[str, np.ndarray]


class SyncFrameCache:
    """Thread-safe bounded cache keyed by the SyncUnit timestamp."""

    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError(f"cache_size must be positive, got {max_size}")
        self._frames: Deque[SyncFrame] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def add(self, frame: SyncFrame) -> None:
        with self._lock:
            self._frames.append(frame)

    def latest(self) -> Optional[SyncFrame]:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def match(self, timestamp: float, tolerance_seconds: float) -> Optional[SyncFrame]:
        with self._lock:
            candidates = list(self._frames)
        if not candidates:
            return None
        frame = min(
            reversed(candidates),
            key=lambda candidate: abs(candidate.sync_timestamp - timestamp),
        )
        if abs(frame.sync_timestamp - timestamp) > tolerance_seconds:
            return None
        return frame

    def nearest_debug(self, timestamp: float, limit: int = 2) -> List[Dict[str, float]]:
        with self._lock:
            candidates = list(self._frames)
        nearest = sorted(
            reversed(candidates),
            key=lambda candidate: abs(candidate.sync_timestamp - timestamp),
        )[:max(0, limit)]
        return [{
            "sync_timestamp": frame.sync_timestamp,
            "delta_ms": (frame.sync_timestamp - timestamp) * 1000.0,
        } for frame in nearest]


class PosePacketCache:
    """Bounded cache of completed pose packets, owned by the GPU scheduler."""

    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError(f"cache_size must be positive, got {max_size}")
        self._packets: Deque[Dict[str, Any]] = deque(maxlen=max_size)

    def add(self, packet: Dict[str, Any]) -> None:
        self._packets.append(packet)

    def clear(self) -> None:
        self._packets.clear()

    def match(self, timestamp: float, tolerance_seconds: float) -> Optional[Dict[str, Any]]:
        if not self._packets:
            return None
        packet = min(
            reversed(self._packets),
            key=lambda candidate: abs(float(candidate["sync_timestamp"]) - timestamp),
        )
        if abs(float(packet["sync_timestamp"]) - timestamp) > tolerance_seconds:
            return None
        return packet


def resize_frame_and_intrinsics(
    color: np.ndarray,
    depth: Optional[np.ndarray],
    cam_K: np.ndarray,
    target_height: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    if target_height == -1:
        return color, depth, np.asarray(cam_K, dtype=np.float64)
    if target_height <= 0:
        raise ValueError(f"target_height must be -1 or a positive integer, got {target_height}")

    height, width = color.shape[:2]
    if height <= target_height:
        return color, depth, np.asarray(cam_K, dtype=np.float64)

    scale = float(target_height) / float(height)
    target_width = max(1, int(round(width * scale)))
    resized_color = cv2.resize(color, (target_width, target_height), interpolation=cv2.INTER_AREA)
    resized_depth = None
    if depth is not None:
        resized_depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    resized_K = np.asarray(cam_K, dtype=np.float64).copy()
    resized_K[0, 0] *= scale
    resized_K[1, 1] *= scale
    resized_K[0, 2] *= scale
    resized_K[1, 2] *= scale
    return resized_color, resized_depth, resized_K


@dataclass
class ObjectSpec:
    object_id: str
    mesh_path: str
    camera_name: Optional[str] = None
    apply_scale: float = 1.0
    force_apply_color: bool = False
    apply_color: Tuple[int, int, int] = (0, 159, 237)
    est_refine_iter: Optional[int] = None
    track_refine_iter: Optional[int] = None
    mask_path: Optional[str] = None


@dataclass
class ObjectState:
    spec: ObjectSpec
    state: str = "uninitialized"
    source_camera: Optional[str] = None
    mesh: Optional[trimesh.Trimesh] = None
    estimator: Any = None
    tracker_2d: Any = None
    kalman_filter: Optional[KalmanFilter6D] = None
    kf_mean: Optional[np.ndarray] = None
    kf_covariance: Optional[np.ndarray] = None
    init_mask: Optional[np.ndarray] = None
    last_mask: Optional[np.ndarray] = None
    last_T_camera_object: Optional[np.ndarray] = None
    last_T_world_object: Optional[np.ndarray] = None
    last_score: Optional[float] = None
    error_msg: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return (self.state == "tracking" and self.last_T_camera_object is not None
                and self.last_T_world_object is not None)

    def reset_runtime(self) -> None:
        self.state = "uninitialized"
        self.source_camera = None
        self.estimator = None
        self.tracker_2d = None
        self.kalman_filter = None
        self.kf_mean = None
        self.kf_covariance = None
        self.init_mask = None
        self.last_mask = None
        self.last_T_camera_object = None
        self.last_T_world_object = None
        self.last_score = None
        self.error_msg = None
        self.meta = {}


@dataclass
class InitInteraction:
    """Frozen RGB-D input and operator choices for one initialization attempt."""

    object_idx: int
    camera_name: str
    frame: SyncFrame
    mode: str = "roi"
    drag_start: Optional[Tuple[int, int]] = None
    drag_current: Optional[Tuple[int, int]] = None
    bbox_xywh: Optional[Tuple[int, int, int, int]] = None
    mask: Optional[np.ndarray] = None
    mask_source: Optional[str] = None


class FoundationPoseFactory:

    def __init__(self, debug: int = 0, debug_dir: str = "./debug/object_pose_server"):
        from FoundationPose.estimater import PoseRefinePredictor, ScorePredictor, dr

        self._debug = debug
        self._debug_dir = debug_dir
        self._scorer = ScorePredictor()
        self._refiner = PoseRefinePredictor()
        self._glctx = dr.RasterizeCudaContext()

    def load_mesh(self, spec: ObjectSpec) -> trimesh.Trimesh:
        from FoundationPose.estimater import trimesh_add_pure_colored_texture

        if not os.path.exists(spec.mesh_path):
            raise FileNotFoundError(f"Mesh file not found for {spec.object_id}: {spec.mesh_path}")
        mesh = trimesh.load(spec.mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if spec.apply_scale != 1.0:
            mesh.apply_scale(spec.apply_scale)
        if spec.force_apply_color:
            mesh = trimesh_add_pure_colored_texture(
                mesh,
                color=np.asarray(spec.apply_color, dtype=np.uint8),
                resolution=10,
            )
        return mesh

    def create_estimator(self, spec: ObjectSpec, mesh: trimesh.Trimesh) -> Any:
        from FoundationPose.estimater import FoundationPose

        debug_dir = os.path.join(self._debug_dir, spec.object_id)
        return FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=self._scorer,
            refiner=self._refiner,
            glctx=self._glctx,
            debug=self._debug,
            debug_dir=debug_dir,
        )


class ObjectPoseServer:

    def __init__(
        self,
        *,
        video_shape: Tuple[int, int],
        sync_channel: str,
        pub_channel: str,
        request_channel: str,
        camera_info: Dict[str, str],
        calib_filedir: str,
        world_calib: Optional[str],
        object_specs: List[ObjectSpec],
        est_refine_iter: int,
        track_refine_iter: int,
        internal_height: int,
        activate_2d_tracker: bool,
        activate_kalman_filter: bool,
        kf_measurement_noise_scale: float,
        sam_api_endpoint: Optional[str],
        sam_api_autostart: bool,
        sam_api_script: Optional[str],
        sam_api_checkpoint_path: Optional[str],
        sam_api_model_type: str,
        sam_api_startup_timeout: float,
        display_scale: float,
        debug: int,
        background_fps: float,
        cache_size: int,
        timestamp_tolerance_ms: float,
    ):
        self.video_shape = video_shape
        self.sync_channel = sync_channel
        self.pub_channel = pub_channel
        self.request_channel = request_channel
        self.camera_info = camera_info
        self.camera_name_list = list(camera_info.values())
        self.calib_filedir = calib_filedir
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self.internal_height = internal_height
        self.activate_2d_tracker = activate_2d_tracker
        self.activate_kalman_filter = activate_kalman_filter
        self.kf_measurement_noise_scale = kf_measurement_noise_scale
        self.sam_api_endpoint = sam_api_endpoint
        self.sam_api_autostart = sam_api_autostart
        self.sam_api_script = sam_api_script
        self.sam_api_checkpoint_path = sam_api_checkpoint_path
        self.sam_api_model_type = sam_api_model_type
        self.sam_api_startup_timeout = sam_api_startup_timeout
        self.display_scale = display_scale
        self.debug = debug
        if not np.isfinite(display_scale) or display_scale <= 0.0:
            raise ValueError(f"display_scale must be positive and finite, got {display_scale}")
        if not np.isfinite(background_fps) or background_fps <= 0.0:
            raise ValueError(f"background_fps must be positive and finite, got {background_fps}")
        if not np.isfinite(timestamp_tolerance_ms) or timestamp_tolerance_ms < 0.0:
            raise ValueError(
                f"timestamp_tolerance_ms must be non-negative and finite, got {timestamp_tolerance_ms}")
        self.background_fps = float(background_fps)
        self.background_interval = 1.0 / self.background_fps
        self.timestamp_tolerance = float(timestamp_tolerance_ms) / 1000.0
        self.cache_size = int(cache_size)
        self.frame_cache = SyncFrameCache(cache_size)
        self.pose_cache = PosePacketCache(cache_size)

        self.object_states = [ObjectState(spec=spec) for spec in object_specs]
        self.current_object_idx = 0
        self.latest_frame: Optional[SyncFrame] = None
        self.shutdown = False
        self.factory: Optional[FoundationPoseFactory] = None
        self.sam_api_process: Optional[subprocess.Popen] = None
        self.cache_thread: Optional[threading.Thread] = None
        self.last_tracking_timestamp: Optional[float] = None
        self.init_interaction: Optional[InitInteraction] = None
        self.selected_camera_by_object: Dict[int, str] = {}
        self.ui_message: Optional[str] = None
        self._ui_dirty = True
        self._window_created = False
        self._closed = False

        self.cam_intr_map = self._load_intrinsics()
        fixed_camera_calibration = load_fixed_camera_calibrations(world_calib, self.camera_name_list)
        self.T_world_camera_map = fixed_camera_calibration.T_world_camera_by_name

        self.ctx = zmq.Context()

        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.SNDHWM, 1)
        self.pub_socket.setsockopt(zmq.CONFLATE, 1)
        pub_endpoint = channel_name_to_endpoint(pub_channel)
        ensure_ipc_dir(pub_endpoint)
        self.pub_socket.bind(pub_endpoint)

        self.request_socket = self.ctx.socket(zmq.REP)
        self.request_socket.setsockopt(zmq.LINGER, 0)
        request_endpoint = channel_name_to_endpoint(request_channel)
        ensure_ipc_dir(request_endpoint)
        self.request_socket.bind(request_endpoint)

        self.poller = zmq.Poller()
        self.poller.register(self.request_socket, zmq.POLLIN)

    def _load_intrinsics(self) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        for camera_name in self.camera_name_list:
            path = os.path.join(self.calib_filedir, "cam_intr", f"{camera_name}.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing cam_intr for camera {camera_name}: {path}")
            with open(path, "rb") as f:
                result[camera_name] = np.asarray(pickle.load(f), dtype=np.float64)
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.shutdown = True
        if self.cache_thread is not None and self.cache_thread.is_alive():
            self.cache_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        self.pub_socket.close(linger=0)
        self.request_socket.close(linger=0)
        self.ctx.term()
        self._stop_sam_api()

    def _parse_sam_api_endpoint(self) -> Optional[Tuple[str, int]]:
        if not self.sam_api_endpoint:
            return None
        parsed = urllib.parse.urlparse(self.sam_api_endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            _logger.warning("SAM-HQ autostart only supports http(s) endpoints, got %s", self.sam_api_endpoint)
            return None
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return parsed.hostname, port

    def _sam_api_port_open(self, host: str, port: int, timeout: float = 0.2) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _maybe_start_sam_api(self) -> None:
        if not self.sam_api_autostart:
            return

        endpoint = self._parse_sam_api_endpoint()
        if endpoint is None:
            return
        host, port = endpoint
        if host not in ("127.0.0.1", "localhost"):
            _logger.warning("SAM-HQ autostart only starts local endpoints, got host=%s", host)
            return
        if self._sam_api_port_open(host, port):
            _logger.info("SAM-HQ API already reachable at %s", self.sam_api_endpoint)
            return

        script = os.path.abspath(os.path.expanduser(self.sam_api_script or ""))
        checkpoint_path = os.path.abspath(os.path.expanduser(self.sam_api_checkpoint_path or ""))
        if not os.path.exists(script):
            raise FileNotFoundError(f"SAM-HQ API script not found: {script}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM-HQ checkpoint not found: {checkpoint_path}")

        cmd = [
            sys.executable,
            script,
            "--checkpoint_path",
            checkpoint_path,
            "--model_type",
            self.sam_api_model_type,
            "--port",
            str(port),
        ]
        _logger.info("Starting SAM-HQ API: %s", " ".join(cmd))
        self.sam_api_process = subprocess.Popen(
            cmd,
            cwd=THIS_DIR,
            start_new_session=True,
        )

        deadline = time.time() + self.sam_api_startup_timeout
        while time.time() < deadline:
            if self.sam_api_process.poll() is not None:
                raise RuntimeError(f"SAM-HQ API exited early with code {self.sam_api_process.returncode}")
            if self._sam_api_port_open(host, port):
                _logger.info("SAM-HQ API is ready at %s", self.sam_api_endpoint)
                return
            time.sleep(0.25)

        self._stop_sam_api()
        raise TimeoutError(f"Timed out waiting for SAM-HQ API at {self.sam_api_endpoint}")

    def _stop_sam_api(self) -> None:
        process = self.sam_api_process
        self.sam_api_process = None
        if process is None or process.poll() is not None:
            return
        _logger.info("Stopping SAM-HQ API process pid=%s", process.pid)
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            _logger.warning("SAM-HQ API did not exit after terminate; killing pid=%s", process.pid)
            process.kill()
            process.wait(timeout=5.0)

    def _decode_sync_message(self, msg: bytes) -> Optional[SyncFrame]:
        data = msgpack.unpackb(msg, object_hook=msgpack_numpy.decode, raw=False)
        synced_data = data.get("synced_data")
        synced_ts = data.get("synced_ts")
        if synced_data is None or synced_ts is None:
            return None

        width, height = self.video_shape
        colors: Dict[str, np.ndarray] = {}
        depths: Dict[str, np.ndarray] = {}
        cam_intr_by_camera: Dict[str, np.ndarray] = {}
        camera_idx = 0
        for payload in synced_data:
            if not isinstance(payload, dict) or "color" not in payload:
                continue
            if camera_idx >= len(self.camera_name_list):
                break
            camera_name = self.camera_name_list[camera_idx]
            color = payload["color"].reshape(height, width, 3)
            depth: Optional[np.ndarray] = None
            if "depth" in payload:
                depth = payload["depth"].reshape(height, width)
                if depth.dtype == np.uint16:
                    depth = depth.astype(np.float32) / float(payload.get("depth_scale", 1000))
                else:
                    depth = depth.astype(np.float32)
                depth[(depth < 0.001) | ~np.isfinite(depth)] = 0.0
            resized_color, resized_depth, resized_K = resize_frame_and_intrinsics(
                color=color,
                depth=depth,
                cam_K=self.cam_intr_map[camera_name],
                target_height=self.internal_height,
            )
            colors[camera_name] = resized_color
            cam_intr_by_camera[camera_name] = resized_K
            if resized_depth is not None:
                depths[camera_name] = resized_depth
            camera_idx += 1

        if not colors:
            return None
        return SyncFrame(
            sync_timestamp=float(synced_ts),
            color_by_camera=colors,
            depth_by_camera=depths,
            cam_intr_by_camera=cam_intr_by_camera,
        )

    def _cache_sync_frames(self) -> None:
        """Continuously decode SyncUnit frames without blocking GPU inference."""
        socket = self.ctx.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.setsockopt(zmq.RCVHWM, self.cache_size)
        endpoint = channel_name_to_endpoint(self.sync_channel)
        ensure_ipc_dir(endpoint)
        socket.connect(endpoint)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        try:
            while not self.shutdown:
                if socket not in dict(poller.poll(timeout=250)):
                    continue
                try:
                    frame = self._decode_sync_message(socket.recv())
                    if frame is not None:
                        self.frame_cache.add(frame)
                except Exception as exc:
                    _logger.warning("Skipping malformed SyncUnit frame: %s", exc)
        finally:
            poller.unregister(socket)
            socket.close(linger=0)

    def _start_cache_thread(self) -> None:
        self.cache_thread = threading.Thread(
            target=self._cache_sync_frames,
            name="foundationpose-sync-cache",
            daemon=True,
        )
        self.cache_thread.start()

    @staticmethod
    def _available_init_cameras(frame: SyncFrame) -> List[str]:
        return [
            camera_name
            for camera_name in frame.color_by_camera
            if camera_name in frame.depth_by_camera and camera_name in frame.cam_intr_by_camera
        ]

    def _preview_camera(self, frame: SyncFrame) -> Optional[str]:
        state = self.object_states[self.current_object_idx]
        if state.source_camera in frame.color_by_camera:
            return state.source_camera
        if state.spec.camera_name:
            return state.spec.camera_name if state.spec.camera_name in frame.color_by_camera else None

        available = self._available_init_cameras(frame)
        if not available:
            return next(iter(frame.color_by_camera), None)
        selected = self.selected_camera_by_object.get(self.current_object_idx)
        if selected not in available:
            selected = available[0]
            self.selected_camera_by_object[self.current_object_idx] = selected
        return selected

    def _cycle_init_camera(self, delta: int) -> None:
        if self.latest_frame is None:
            self.ui_message = "No synchronized RGB-D frame is available"
            return
        state = self.object_states[self.current_object_idx]
        if state.spec.camera_name:
            self.ui_message = f"Camera is pinned to {state.spec.camera_name} by object config"
            return
        if state.source_camera:
            self.ui_message = "Reset the tracked object before changing its initialization camera"
            return
        available = self._available_init_cameras(self.latest_frame)
        if not available:
            self.ui_message = "No camera currently has synchronized RGB-D data"
            return
        current = self.selected_camera_by_object.get(self.current_object_idx, available[0])
        current_idx = available.index(current) if current in available else 0
        selected = available[(current_idx + delta) % len(available)]
        self.selected_camera_by_object[self.current_object_idx] = selected
        self.ui_message = f"Selected initialization camera: {selected}"

    def _start_init_interaction(self) -> None:
        if self.latest_frame is None:
            self.ui_message = "No synchronized frame is available for initialization"
            return
        state = self.object_states[self.current_object_idx]
        if state.state != "uninitialized":
            self.ui_message = f"Reset {state.spec.object_id} before initializing it again"
            return
        camera_name = self._preview_camera(self.latest_frame)
        if (camera_name is None or camera_name not in self.latest_frame.color_by_camera or
                camera_name not in self.latest_frame.depth_by_camera or
                camera_name not in self.latest_frame.cam_intr_by_camera):
            self.ui_message = "Selected camera does not have synchronized RGB-D data"
            return

        state.meta["init_attempted"] = True
        state.error_msg = None
        self.init_interaction = InitInteraction(
            object_idx=self.current_object_idx,
            camera_name=camera_name,
            frame=self.latest_frame,
        )
        self.ui_message = "Drag a bounding box, then press Enter/Space"
        _logger.info("Selecting initialization ROI for %s from %s", state.spec.object_id, camera_name)

    def _cancel_init_interaction(self) -> None:
        interaction = self.init_interaction
        if interaction is None:
            return
        state = self.object_states[interaction.object_idx]
        state.reset_runtime()
        state.meta["init_cancelled"] = True
        state.error_msg = "Initialization cancelled"
        self.init_interaction = None
        self.ui_message = f"Initialization cancelled for {state.spec.object_id}"
        _logger.info("Initialization cancelled for %s", state.spec.object_id)

    @staticmethod
    def _bbox_from_points(
        start: Tuple[int, int],
        end: Tuple[int, int],
        shape: Tuple[int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        height, width = shape
        x1, x2 = sorted((max(0, min(width - 1, start[0])), max(0, min(width - 1, end[0]))))
        y1, y2 = sorted((max(0, min(height - 1, start[1])), max(0, min(height - 1, end[1]))))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2 - x1, y2 - y1

    def _display_to_image_point(self, x: int, y: int, shape: Tuple[int, int]) -> Tuple[int, int]:
        height, width = shape
        image_x = int(round(x / self.display_scale))
        image_y = int(round(y / self.display_scale))
        return max(0, min(width - 1, image_x)), max(0, min(height - 1, image_y))

    def _handle_mouse(self, event: int, x: int, y: int, _flags: int, _param: Any = None) -> None:
        interaction = self.init_interaction
        if interaction is None or interaction.mode != "roi":
            return
        shape = interaction.frame.color_by_camera[interaction.camera_name].shape[:2]
        point = self._display_to_image_point(x, y, shape)
        if event == cv2.EVENT_LBUTTONDOWN:
            interaction.drag_start = point
            interaction.drag_current = point
            interaction.bbox_xywh = None
        elif event == cv2.EVENT_MOUSEMOVE and interaction.drag_start is not None:
            interaction.drag_current = point
        elif event == cv2.EVENT_LBUTTONUP and interaction.drag_start is not None:
            interaction.drag_current = point
            interaction.bbox_xywh = self._bbox_from_points(interaction.drag_start, point, shape)
            interaction.drag_start = None
            if interaction.bbox_xywh is None:
                self.ui_message = "ROI must have non-zero width and height"
            else:
                self.ui_message = "Press Enter/Space to segment this ROI"
        self._ui_dirty = True

    def _read_mask_override(self, state: ObjectState, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        mask_path = state.spec.mask_path
        if not mask_path:
            return None
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read configured mask_path for {state.spec.object_id}: {mask_path}")
        if mask.shape != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8)

    def _request_sam_mask(self, frame_rgb: np.ndarray, bbox_xywh: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        if not self.sam_api_endpoint:
            _logger.warning("SAM-HQ API endpoint is not configured; cannot initialize from bbox prompt")
            return None

        mask_path: Optional[str] = None
        if SAVE_SAM_INTERMEDIATE_FILES:
            tmp_dir = os.path.join(THIS_DIR, "tmp", "object_pose_server")
            os.makedirs(tmp_dir, exist_ok=True)
            token = f"{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
            frame_path = os.path.join(tmp_dir, f"sam_frame_{token}.png")
            mask_path = os.path.join(tmp_dir, f"sam_mask_{token}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            request_data = {
                "frame_path": frame_path,
                "bbox_xywh": list(map(int, bbox_xywh)),
                "output_mask_path": mask_path,
            }
            endpoint = self.sam_api_endpoint
            payload = json.dumps(request_data).encode("utf-8")
            content_type = "application/json"
        else:
            encoded, frame_png = cv2.imencode(".png", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            if not encoded:
                _logger.warning("Could not encode SAM input frame")
                return None
            x, y, width, height = map(int, bbox_xywh)
            parsed_endpoint = urllib.parse.urlsplit(self.sam_api_endpoint)
            endpoint = urllib.parse.urlunsplit((
                parsed_endpoint.scheme,
                parsed_endpoint.netloc,
                parsed_endpoint.path.rstrip("/") + "/binary",
                urllib.parse.urlencode({"x": x, "y": y, "w": width, "h": height}),
                parsed_endpoint.fragment,
            ))
            payload = frame_png.tobytes()
            content_type = "image/png"

        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status >= 300:
                    _logger.warning("SAM API returned HTTP %s", resp.status)
                    return None
                response_data = resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            _logger.warning("SAM API request failed: %s", exc)
            return None

        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = cv2.imdecode(np.frombuffer(response_data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            _logger.warning("SAM API did not return a usable mask")
            return None
        return (mask > 0).astype(np.uint8)

    def _validate_init_mask(
        self,
        state: ObjectState,
        mask: np.ndarray,
        shape: Tuple[int, int],
    ) -> Optional[str]:
        if mask.ndim != 2:
            return f"Init mask must be 2D, got shape {mask.shape}"
        if mask.shape != shape:
            return f"Init mask shape {mask.shape} does not match RGB frame shape {shape}"
        area = int((mask > 0).sum())
        image_area = int(shape[0] * shape[1])
        if area < 16:
            return "Init mask has fewer than 16 foreground pixels"
        if area > int(image_area * 0.95):
            return "Init mask covers more than 95% of the frame"
        state.meta["init_mask_area"] = area
        return None

    def _prepare_init_mask(self) -> None:
        interaction = self.init_interaction
        if interaction is None or interaction.mode != "roi":
            return
        if interaction.bbox_xywh is None:
            self.ui_message = "Draw a valid ROI before submitting"
            return

        state = self.object_states[interaction.object_idx]
        color = interaction.frame.color_by_camera[interaction.camera_name]
        interaction.mode = "processing_mask"
        self.ui_message = "Generating initialization mask..."
        self._draw_status(interaction.frame)
        cv2.waitKey(1)
        try:
            mask = self._read_mask_override(state, color.shape[:2])
            mask_source = "mask_path"
            if mask is None:
                mask = self._request_sam_mask(color, interaction.bbox_xywh)
                mask_source = "sam_hq"
            if mask is None:
                raise RuntimeError("SAM-HQ did not return a usable init mask")
            mask_error = self._validate_init_mask(state, mask, color.shape[:2])
            if mask_error is not None:
                raise ValueError(mask_error)
        except Exception as exc:
            interaction.mode = "roi"
            state.error_msg = str(exc)
            self.ui_message = f"Mask failed: {exc}"
            _logger.error("%s: %s", state.spec.object_id, state.error_msg)
            return

        interaction.mask = mask
        interaction.mask_source = mask_source
        interaction.mode = "mask"
        state.error_msg = None
        self.ui_message = "Enter/Space: accept mask; r: redraw; c/Esc: cancel"

    def _complete_initialization(self) -> bool:
        interaction = self.init_interaction
        if (interaction is None or interaction.mode != "mask" or interaction.mask is None or
                interaction.bbox_xywh is None or interaction.mask_source is None):
            return False

        state = self.object_states[interaction.object_idx]
        frame = interaction.frame
        camera_name = interaction.camera_name
        color = frame.color_by_camera[camera_name]
        depth = frame.depth_by_camera[camera_name]
        mask = interaction.mask
        interaction.mode = "initializing"
        self.ui_message = "Loading models and registering pose..."
        self._draw_status(frame)
        cv2.waitKey(1)
        try:
            if self.factory is None:
                _logger.info("Loading FoundationPose models...")
                self.factory = FoundationPoseFactory(debug=self.debug)
            if state.mesh is None:
                state.mesh = self.factory.load_mesh(state.spec)
            state.estimator = self.factory.create_estimator(state.spec, state.mesh)
            cam_K = frame.cam_intr_by_camera[camera_name]
            refine_iter = state.spec.est_refine_iter or self.est_refine_iter
            pose_cam = state.estimator.register(
                K=cam_K,
                rgb=color,
                depth=depth,
                ob_mask=(mask.astype(np.uint8) * 255),
                iteration=refine_iter,
            )
            state.source_camera = camera_name
            state.init_mask = mask
            if self.activate_2d_tracker:

                def _init_cutie_tracker():
                    clear_hydra_if_initialized()
                    tracker = Cutie()
                    tracker.initialize(color, init_info={"mask": mask})
                    return tracker

                state.tracker_2d = run_with_cpu_default_tensor_type(_init_cutie_tracker)
            else:
                state.tracker_2d = Tracker_2D()
            # Capture the init mask from Cutie for visualization overlay
            if (hasattr(state.tracker_2d, 'last_mask') and state.tracker_2d.last_mask is not None):
                state.last_mask = state.tracker_2d.last_mask.copy()
            if self.activate_kalman_filter:
                state.kalman_filter = KalmanFilter6D(self.kf_measurement_noise_scale)
                state.kf_mean, state.kf_covariance = state.kalman_filter.initiate(get_6d_pose_arr_from_mat(pose_cam))
            state.last_T_camera_object = np.asarray(pose_cam, dtype=np.float64)
            state.last_T_world_object = self.T_world_camera_map[camera_name] @ state.last_T_camera_object
            state.state = "tracking"
            state.error_msg = None
            state.meta = {
                "init_bbox_xywh": interaction.bbox_xywh,
                "init_mask_area": int((mask > 0).sum()),
                "init_mask_source": interaction.mask_source,
                "init_input_shape": [int(color.shape[1]), int(color.shape[0])],
                "init_sync_timestamp": frame.sync_timestamp,
            }
            self.last_tracking_timestamp = max(
                self.last_tracking_timestamp if self.last_tracking_timestamp is not None else -np.inf,
                frame.sync_timestamp,
            )
            self.pose_cache.clear()
            self.init_interaction = None
            self.ui_message = f"Initialized {state.spec.object_id} from {camera_name}"
            _logger.info("Initialized %s from %s", state.spec.object_id, camera_name)
            return True
        except Exception as exc:
            state.reset_runtime()
            state.error_msg = str(exc)
            interaction.mode = "mask"
            self.ui_message = f"Initialization failed: {exc}"
            _logger.exception("Failed to initialize %s", state.spec.object_id)
            return False

    def _track_object(self, state: ObjectState, frame: SyncFrame) -> None:
        if state.state != "tracking" or state.estimator is None or state.source_camera is None:
            return
        camera_name = state.source_camera
        color = frame.color_by_camera.get(camera_name)
        depth = frame.depth_by_camera.get(camera_name)
        if color is None or depth is None:
            state.state = "lost"
            state.error_msg = f"Missing RGB-D frame for {camera_name}"
            return

        try:
            cam_K = frame.cam_intr_by_camera[camera_name]
            if self.activate_2d_tracker and state.tracker_2d is not None:
                bbox = state.tracker_2d.track(color)
                # Collect the tracking mask for visualization
                if hasattr(state.tracker_2d, 'last_mask') and state.tracker_2d.last_mask is not None:
                    state.last_mask = state.tracker_2d.last_mask.copy()
                if bbox[0] >= 0 and bbox[1] >= 0 and state.estimator.pose_last is not None:
                    cx = bbox[0] + bbox[2] / 2.0
                    cy = bbox[1] + bbox[3] / 2.0
                    if self.activate_kalman_filter and state.kalman_filter is not None:
                        state.kf_mean, state.kf_covariance = state.kalman_filter.update(
                            state.kf_mean,
                            state.kf_covariance,
                            get_6d_pose_arr_from_mat(state.estimator.pose_last),
                        )
                        measurement_xy = np.asarray(
                            get_pose_xy_from_image_point(state.estimator.pose_last, cam_K, cx, cy))
                        state.kf_mean, state.kf_covariance = state.kalman_filter.update_from_xy(
                            state.kf_mean,
                            state.kf_covariance,
                            measurement_xy,
                        )
                        pose_guess = get_mat_from_6d_pose_arr(state.kf_mean[:6])
                        state.estimator.pose_last = torch.from_numpy(pose_guess).float().unsqueeze(0).cuda()
                    else:
                        state.estimator.pose_last = adjust_pose_to_image_point(
                            state.estimator.pose_last,
                            cam_K,
                            cx,
                            cy,
                        )

            pose_cam = state.estimator.track_one(
                rgb=color,
                depth=depth,
                K=cam_K,
                iteration=state.spec.track_refine_iter or self.track_refine_iter,
            )
            if self.activate_2d_tracker and self.activate_kalman_filter and state.kalman_filter is not None:
                state.kf_mean, state.kf_covariance = state.kalman_filter.predict(
                    state.kf_mean,
                    state.kf_covariance,
                )
            state.last_T_camera_object = np.asarray(pose_cam, dtype=np.float64)
            state.last_T_world_object = self.T_world_camera_map[camera_name] @ state.last_T_camera_object
            state.state = "tracking"
            state.error_msg = None
        except Exception as exc:
            state.state = "lost"
            state.error_msg = str(exc)
            _logger.exception("Tracking failed for %s", state.spec.object_id)

    @staticmethod
    def _state_object_payload(state: ObjectState) -> Dict[str, Any]:
        return {
            "object_id": state.spec.object_id,
            "T_camera_object": state.last_T_camera_object,
            "T_world_object": state.last_T_world_object,
            "valid": state.valid,
            "state": state.state,
            "score": state.last_score,
            "source_camera": state.source_camera,
            "meta": {
                **state.meta,
                **({"error": state.error_msg} if state.error_msg else {}),
            },
        }

    def _make_pose_packet(
        self,
        sync_timestamp: float,
        process_time: float,
        object_payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "sync_timestamp": sync_timestamp,
            "server_timestamp": time.time(),
            "object_poses": (object_payloads if object_payloads is not None else
                             [self._state_object_payload(state) for state in self.object_states]),
            "process_time": process_time,
        }

    def _publish_packet(self, packet: Mapping[str, Any]) -> None:
        self.pub_socket.send(msgpack.packb(dict(packet), default=msgpack_numpy.encode))

    def _process_tracking_frame(self, frame: SyncFrame, publish: bool = True) -> Dict[str, Any]:
        started = time.perf_counter()
        for state in self.object_states:
            self._track_object(state, frame)
        packet = self._make_pose_packet(frame.sync_timestamp, time.perf_counter() - started)
        self.last_tracking_timestamp = frame.sync_timestamp
        self.pose_cache.add(packet)
        if publish:
            self._publish_packet(packet)
        return packet

    def _process_isolated_request(self, frame: SyncFrame) -> Dict[str, Any]:
        """Refine an older frame without rewinding continuous tracker state."""
        started = time.perf_counter()
        object_payloads: List[Dict[str, Any]] = []
        for state in self.object_states:
            if not state.valid or state.estimator is None or state.source_camera is None:
                object_payloads.append(self._state_object_payload(state))
                continue

            camera_name = state.source_camera
            color = frame.color_by_camera.get(camera_name)
            depth = frame.depth_by_camera.get(camera_name)
            cam_K = frame.cam_intr_by_camera.get(camera_name)
            if color is None or depth is None or cam_K is None:
                raise RuntimeError(
                    f"Requested frame is missing RGB-D or intrinsics for camera {camera_name}")
            pose_last = state.estimator.pose_last
            if pose_last is None:
                raise RuntimeError(f"Object {state.spec.object_id} has no pose seed")

            pose_seed = pose_last.detach().clone()
            try:
                pose_cam = state.estimator.track_one_w_spec_last_pose(
                    rgb=color,
                    depth=depth,
                    K=cam_K,
                    iteration=state.spec.track_refine_iter or self.track_refine_iter,
                    spec_last_pose=pose_seed,
                )
            finally:
                # track_one_w_spec_last_pose mutates pose_last; an older request must not
                # rewind the state consumed by Cutie, Kalman, or background tracking.
                state.estimator.pose_last = pose_last

            T_camera_object = np.asarray(pose_cam, dtype=np.float64)
            T_world_object = self.T_world_camera_map[camera_name] @ T_camera_object
            object_payloads.append({
                "object_id": state.spec.object_id,
                "T_camera_object": T_camera_object,
                "T_world_object": T_world_object,
                "valid": True,
                "state": "tracking",
                "score": state.last_score,
                "source_camera": camera_name,
                "meta": {**state.meta, "request_processing": "isolated"},
            })

        packet = self._make_pose_packet(
            frame.sync_timestamp,
            time.perf_counter() - started,
            object_payloads,
        )
        self.pose_cache.add(packet)
        return packet

    @staticmethod
    def _request_error(
        code: str,
        message: str,
        requested_synced_ts: Optional[float] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "ok": False,
            "error": {"code": code, "message": message},
            "server_timestamp": time.time(),
        }
        if requested_synced_ts is not None:
            response["requested_synced_ts"] = requested_synced_ts
        if details is not None:
            response["error"]["details"] = dict(details)
        return response

    @staticmethod
    def _decode_pose_request(message: bytes) -> float:
        request = msgpack.unpackb(message, raw=False)
        if not isinstance(request, dict):
            raise ValueError("request must be a mapping")
        value = request.get("synced_ts")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("synced_ts must be a finite number")
        timestamp = float(value)
        if not np.isfinite(timestamp):
            raise ValueError("synced_ts must be a finite number")
        return timestamp

    def _handle_pose_request(self, message: bytes) -> Dict[str, Any]:
        requested_timestamp: Optional[float] = None
        try:
            requested_timestamp = self._decode_pose_request(message)
        except Exception as exc:
            return self._request_error("invalid_request", str(exc))

        cached_packet = self.pose_cache.match(requested_timestamp, self.timestamp_tolerance)
        if cached_packet is not None:
            response = dict(cached_packet)
            response.update({
                "ok": True,
                "requested_synced_ts": requested_timestamp,
                "request_processing": "cached",
            })
            return response

        frame = self.frame_cache.match(requested_timestamp, self.timestamp_tolerance)
        if frame is None:
            details = {
                "requested_synced_ts": requested_timestamp,
                "tolerance_ms": self.timestamp_tolerance * 1000.0,
                "syncunit_nearest": self.frame_cache.nearest_debug(requested_timestamp),
            }
            return self._request_error(
                "cache_miss",
                f"no cached SyncUnit frame is within {self.timestamp_tolerance * 1000.0:g} ms "
                f"of synced_ts {requested_timestamp:.6f}",
                requested_timestamp,
                details,
            )

        if not any(state.valid for state in self.object_states):
            return self._request_error(
                "not_ready",
                "FoundationPose++ has no initialized tracking object",
                requested_timestamp,
            )

        try:
            if (self.last_tracking_timestamp is None or
                    frame.sync_timestamp > self.last_tracking_timestamp + self.timestamp_tolerance):
                packet = self._process_tracking_frame(frame, publish=True)
                request_processing = "tracking"
            else:
                packet = self._process_isolated_request(frame)
                request_processing = "isolated"
        except Exception as exc:
            _logger.exception("Timestamp-requested pose estimation failed")
            return self._request_error(
                "inference_error",
                str(exc),
                requested_timestamp,
            )

        response = dict(packet)
        response.update({
            "ok": True,
            "requested_synced_ts": requested_timestamp,
            "request_processing": request_processing,
        })
        _logger.info(
            "Pose request %.6f matched %.6f (%+.3f ms, %s, %.1f ms)",
            requested_timestamp,
            frame.sync_timestamp,
            (frame.sync_timestamp - requested_timestamp) * 1000.0,
            request_processing,
            float(packet["process_time"]) * 1000.0,
        )
        return response

    def _draw_status(self, frame: SyncFrame) -> None:
        if not frame.color_by_camera:
            return
        interaction = self.init_interaction
        if interaction is not None:
            frame = interaction.frame
            camera_name = interaction.camera_name
        else:
            camera_name = self._preview_camera(frame)
        if camera_name is None:
            camera_name = next(iter(frame.color_by_camera.keys()))
        img = frame.color_by_camera.get(camera_name)
        if img is None:
            img = next(iter(frame.color_by_camera.values()))
            camera_name = next(iter(frame.color_by_camera.keys()))
        disp = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        cam_K = frame.cam_intr_by_camera.get(camera_name)

        # --- Overlay tracking masks (semi-transparent + contour) ---
        _MASK_COLORS_BGR = [
            (0, 255, 0),  # green
            (255, 0, 0),  # blue
            (0, 0, 255),  # red
            (0, 255, 255),  # yellow
            (255, 0, 255),  # magenta
            (255, 255, 0),  # cyan
        ]
        for i, state in enumerate(self.object_states):
            if state.source_camera != camera_name:
                continue
            if state.last_mask is None:
                continue
            if state.last_mask.shape[:2] != disp.shape[:2]:
                continue
            obj_color = _MASK_COLORS_BGR[i % len(_MASK_COLORS_BGR)]
            mask_bool = state.last_mask > 0
            # Semi-transparent fill
            overlay = disp.copy()
            overlay[mask_bool] = obj_color
            disp = cv2.addWeighted(disp, 0.65, overlay, 0.35, 0)
            # Contour outline
            contours, _ = cv2.findContours(state.last_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(disp, contours, -1, obj_color, 2, cv2.LINE_AA)

        # --- Draw pose axes ---
        if cam_K is not None:
            for i, state in enumerate(self.object_states):
                if state.source_camera != camera_name or state.last_T_camera_object is None or not state.valid:
                    continue
                axis_scale = 0.08
                if state.mesh is not None:
                    axis_scale = float(np.clip(state.mesh.extents.max() * 0.35, 0.03, 0.20))
                disp = draw_pose_frame_bgr(
                    disp,
                    state.last_T_camera_object,
                    cam_K,
                    axis_scale=axis_scale,
                    thickness=4 if i == self.current_object_idx else 2,
                )

        if interaction is not None:
            if interaction.mask is not None:
                mask_bool = interaction.mask.astype(bool)
                overlay = disp.copy()
                overlay[mask_bool] = (0, 190, 0)
                disp = cv2.addWeighted(disp, 0.55, overlay, 0.45, 0)
            bbox = interaction.bbox_xywh
            if interaction.drag_start is not None and interaction.drag_current is not None:
                bbox = self._bbox_from_points(interaction.drag_start, interaction.drag_current, img.shape[:2])
            if bbox is not None:
                x, y, width, height = bbox
                cv2.rectangle(disp, (x, y), (x + width, y + height), (0, 220, 255), 2)

        y = 24
        mode = interaction.mode if interaction is not None else "idle"
        cv2.putText(
            disp,
            f"camera={camera_name} sync={frame.sync_timestamp:.3f} mode={mode}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        y += 26
        for i, state in enumerate(self.object_states):
            marker = ">" if i == self.current_object_idx else " "
            text = f"{marker} {state.spec.object_id}: {state.state}"
            color = (0, 220, 0) if state.valid else ((0, 200, 255) if state.state in ("uninitialized", "lost") else
                                                     (0, 0, 255))
            cv2.putText(disp, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 24
        y += 6
        if interaction is None:
            controls = "[/]: object  ,/.: camera  i:init  r:reset current  a:reset all  q:quit"
        elif interaction.mode == "roi":
            controls = "mouse drag: ROI  Enter/Space: segment  c/Esc: cancel"
        elif interaction.mode == "mask":
            controls = "Enter/Space: accept mask  r:redraw ROI  c/Esc: cancel"
        elif interaction.mode == "processing_mask":
            controls = "Generating mask..."
        else:
            controls = "Initializing pose..."
        cv2.putText(
            disp,
            controls,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
        )
        if self.ui_message:
            cv2.putText(
                disp,
                self.ui_message,
                (10, min(disp.shape[0] - 12, y + 26)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 220, 255),
                2,
            )
        if self.display_scale != 1.0:
            disp = cv2.resize(disp, None, fx=self.display_scale, fy=self.display_scale)
        cv2.imshow(WINDOW_NAME, disp)

    def _create_window(self) -> None:
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW_NAME, self._handle_mouse)
            self._window_created = True
            self._draw_waiting_window()
            cv2.waitKey(1)
        except cv2.error as exc:
            raise RuntimeError(
                "Could not create the required OpenCV interaction window; check DISPLAY and OpenCV GUI support"
            ) from exc

    def _draw_waiting_window(self) -> None:
        width, height = 640, 360
        disp = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            disp,
            "FoundationPose++: waiting for synchronized RGB-D frames...",
            (24, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            disp,
            "q/Esc: quit",
            (24, height // 2 + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 220, 255),
            2,
        )
        cv2.imshow(WINDOW_NAME, disp)

    def _redraw_ui(self) -> None:
        frame = self.init_interaction.frame if self.init_interaction is not None else self.latest_frame
        if frame is None:
            self._draw_waiting_window()
        else:
            self._draw_status(frame)
        self._ui_dirty = False

    def _pump_ui(self) -> None:
        self._handle_key(cv2.waitKey(1))
        if self._window_created:
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    self.shutdown = True
            except cv2.error:
                self.shutdown = True

    def _handle_key(self, key: int) -> None:
        if key < 0:
            return
        key &= 0xFF
        interaction = self.init_interaction
        if interaction is not None:
            if interaction.mode == "roi":
                if key in (13, 10, 32):
                    self._prepare_init_mask()
                elif key in (ord("c"), ord("q"), 27):
                    self._cancel_init_interaction()
            elif interaction.mode == "mask":
                if key in (13, 10, 32):
                    self._complete_initialization()
                elif key == ord("r"):
                    interaction.mode = "roi"
                    interaction.drag_start = None
                    interaction.drag_current = None
                    interaction.bbox_xywh = None
                    interaction.mask = None
                    interaction.mask_source = None
                    self.ui_message = "Drag a new bounding box, then press Enter/Space"
                elif key in (ord("c"), ord("q"), 27):
                    self._cancel_init_interaction()
            self._ui_dirty = True
            return

        if key == ord("q") or key == 27:
            self.shutdown = True
        elif key == ord("["):
            self.current_object_idx = (self.current_object_idx - 1) % len(self.object_states)
            self.ui_message = f"Selected object: {self.object_states[self.current_object_idx].spec.object_id}"
        elif key == ord("]"):
            self.current_object_idx = (self.current_object_idx + 1) % len(self.object_states)
            self.ui_message = f"Selected object: {self.object_states[self.current_object_idx].spec.object_id}"
        elif key == ord(","):
            self._cycle_init_camera(-1)
        elif key == ord("."):
            self._cycle_init_camera(1)
        elif key == ord("r"):
            state = self.object_states[self.current_object_idx]
            _logger.info("Resetting object %s", state.spec.object_id)
            state.reset_runtime()
            self.pose_cache.clear()
            if not any(candidate.valid for candidate in self.object_states):
                self.last_tracking_timestamp = None
            self.ui_message = f"Reset {state.spec.object_id}; press i to initialize"
        elif key == ord("a"):
            _logger.info("Resetting all objects")
            for state in self.object_states:
                state.reset_runtime()
            self.current_object_idx = 0
            self.pose_cache.clear()
            self.last_tracking_timestamp = None
            self.ui_message = "Reset all objects; select one and press i to initialize"
        elif key == ord("i"):
            self._start_init_interaction()
        self._ui_dirty = True

    def run(self) -> None:
        _logger.info("Object pose server started")
        _logger.info("Subscribed to sync channel: %s", self.sync_channel)
        _logger.info("Publishing object poses on: %s", self.pub_channel)
        _logger.info("Replying to priority pose requests on: %s", self.request_channel)
        _logger.info(
            "Background tracking target: %.2f FPS; timestamp tolerance: %.1f ms",
            self.background_fps,
            self.timestamp_tolerance * 1000.0,
        )
        self._create_window()
        self._maybe_start_sam_api()
        self._start_cache_thread()
        next_background_time = time.monotonic()
        last_live_display_timestamp: Optional[float] = None

        while not self.shutdown:
            latest_frame = self.frame_cache.latest()
            if latest_frame is not None:
                self.latest_frame = latest_frame
                if (self.init_interaction is None and
                        latest_frame.sync_timestamp != last_live_display_timestamp):
                    self._ui_dirty = True
                    last_live_display_timestamp = latest_frame.sync_timestamp

            # A request already waiting always wins over starting new background work.
            events = dict(self.poller.poll(timeout=10))
            if self.request_socket in events:
                request = self.request_socket.recv()
                response = self._handle_pose_request(request)
                self.request_socket.send(msgpack.packb(response, default=msgpack_numpy.encode))
                # Avoid an immediate background call after request inference. There is no
                # catch-up queue: the next pass will use the newest camera frame.
                next_background_time = max(
                    next_background_time,
                    time.monotonic() + self.background_interval,
                )
                self._ui_dirty = True
                if self._ui_dirty:
                    self._redraw_ui()
                self._pump_ui()
                if self._ui_dirty:
                    self._redraw_ui()
                continue

            now = time.monotonic()
            if now >= next_background_time:
                background_started = now
                frame = self.frame_cache.latest()
                if (frame is not None and any(state.valid for state in self.object_states) and
                        (self.last_tracking_timestamp is None or
                         frame.sync_timestamp > self.last_tracking_timestamp + self.timestamp_tolerance)):
                    self.latest_frame = frame
                    packet = self._process_tracking_frame(frame, publish=True)
                    _logger.info(
                        "Background pose %.6f in %.1f ms",
                        frame.sync_timestamp,
                        float(packet["process_time"]) * 1000.0,
                    )
                    self._ui_dirty = True

                # Start-to-start limiting without replaying missed periods.
                next_background_time = max(
                    background_started + self.background_interval,
                    time.monotonic(),
                )

            if self._ui_dirty:
                self._redraw_ui()
            self._pump_ui()
            if self._ui_dirty:
                self._redraw_ui()


def load_object_specs(path: str) -> List[ObjectSpec]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty object config: {path}")
    raw_objects = cfg.get("objects", cfg if isinstance(cfg, list) else None)
    if not isinstance(raw_objects, list):
        raise ValueError("Object config must contain an 'objects' list")

    specs: List[ObjectSpec] = []
    for raw in raw_objects:
        if "object_id" not in raw or "mesh_path" not in raw:
            raise ValueError("Each object requires object_id and mesh_path")
        color = raw.get("apply_color", (0, 159, 237))
        specs.append(
            ObjectSpec(
                object_id=str(raw["object_id"]),
                mesh_path=os.path.abspath(os.path.expanduser(str(raw["mesh_path"]))),
                camera_name=raw.get("camera_name"),
                apply_scale=float(raw.get("apply_scale", 1.0)),
                force_apply_color=bool(raw.get("force_apply_color", False)),
                apply_color=tuple(int(v) for v in color),
                est_refine_iter=raw.get("est_refine_iter"),
                track_refine_iter=raw.get("track_refine_iter"),
                mask_path=(os.path.abspath(os.path.expanduser(str(raw["mask_path"])))
                           if raw.get("mask_path") else None),
            ))
    if not specs:
        raise ValueError("Object config has no objects")
    return specs


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger().addFilter(ThrottleFilter(throttle_seconds=1.0))

    parser = argparse.ArgumentParser(description="Interactive Object Pose Estimation Server")
    parser.add_argument("--server.video_shape", type=parse_video_shape, default="1280x720")
    parser.add_argument("--server.sync_channel", type=str, required=True)
    parser.add_argument("--server.pub_channel", type=str, required=True)
    parser.add_argument("--server.request_channel", type=str, default="tcp://*:9671")
    parser.add_argument("--camera_info", type=parse_camera_info, required=True)
    parser.add_argument("--calib_filedir", type=str, required=True)
    parser.add_argument(
        "--world_calib",
        type=str,
        default=None,
        help="Optional calibration session or world directory; defaults to the latest calib__* session",
    )
    parser.add_argument("--object_config", type=str, required=True)
    parser.add_argument("--est_refine_iter", type=int, default=10)
    parser.add_argument("--track_refine_iter", type=int, default=5)
    parser.add_argument(
        "--internal_height",
        type=int,
        default=-1,
        help="Internal RGB-D height. Use -1 to keep original frame size.",
    )
    parser.add_argument("--activate_2d_tracker", action="store_true")
    parser.add_argument("--activate_kalman_filter", action="store_true")
    parser.add_argument("--kf_measurement_noise_scale", type=float, default=0.05)
    parser.add_argument("--sam_api_endpoint", type=str, default=None)
    parser.add_argument("--sam_api_autostart", action="store_true")
    parser.add_argument(
        "--sam_api_script",
        type=str,
        default=os.path.join(THIS_DIR, "src", "WebAPI", "hq_sam_api_alt.py"),
    )
    parser.add_argument(
        "--sam_api_checkpoint_path",
        type=str,
        default=os.path.join(THIS_DIR, "sam-hq", "pretrained_checkpoints", "sam_hq_vit_l.pth"),
    )
    parser.add_argument("--sam_api_model_type", type=str, default="vit_l")
    parser.add_argument("--sam_api_startup_timeout", type=float, default=120.0)
    parser.add_argument("--display_scale", type=float, default=0.75)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--background_fps", type=float, default=5.0)
    parser.add_argument("--cache_size", type=int, default=30)
    parser.add_argument("--timestamp_tolerance_ms", type=float, default=5.0)
    args = parser.parse_args()

    object_specs = load_object_specs(args.object_config)
    server = ObjectPoseServer(
        video_shape=getattr(args, "server.video_shape"),
        sync_channel=getattr(args, "server.sync_channel"),
        pub_channel=getattr(args, "server.pub_channel"),
        request_channel=getattr(args, "server.request_channel"),
        camera_info=args.camera_info,
        calib_filedir=os.path.abspath(os.path.expanduser(args.calib_filedir)),
        world_calib=args.world_calib,
        object_specs=object_specs,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        internal_height=args.internal_height,
        activate_2d_tracker=args.activate_2d_tracker,
        activate_kalman_filter=args.activate_kalman_filter,
        kf_measurement_noise_scale=args.kf_measurement_noise_scale,
        sam_api_endpoint=args.sam_api_endpoint,
        sam_api_autostart=args.sam_api_autostart,
        sam_api_script=os.path.abspath(os.path.expanduser(args.sam_api_script)),
        sam_api_checkpoint_path=os.path.abspath(os.path.expanduser(args.sam_api_checkpoint_path)),
        sam_api_model_type=args.sam_api_model_type,
        sam_api_startup_timeout=args.sam_api_startup_timeout,
        display_scale=args.display_scale,
        debug=args.debug,
        background_fps=args.background_fps,
        cache_size=args.cache_size,
        timestamp_tolerance_ms=args.timestamp_tolerance_ms,
    )

    def _signal_handler(signum, _frame):
        _logger.info("Received signal %s, shutting down", signal.Signals(signum).name)
        server.shutdown = True

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)
    atexit.register(server.close)

    try:
        server.run()
    finally:
        server.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
