#!/usr/bin/env python3
"""Interactive Object Pose Estimation Server.

This server is intentionally started manually.  It subscribes to a SyncUnit
RGB-D stream, lets the operator initialize one or more objects from the latest
frame, tracks them with FoundationPose++, and publishes world-frame object
poses using the same msgpack_numpy PUB pattern as WiLoR/POEM.
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import pickle
import signal
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

_logger = logging.getLogger("object_pose_server")


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


def inv_transf_np(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


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


@dataclass
class SyncFrame:
    sync_timestamp: float
    color_by_camera: Dict[str, np.ndarray]
    depth_by_camera: Dict[str, np.ndarray]


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
    last_T_world_object: Optional[np.ndarray] = None
    last_score: Optional[float] = None
    error_msg: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.state == "tracking" and self.last_T_world_object is not None

    def reset_runtime(self) -> None:
        self.state = "uninitialized"
        self.source_camera = None
        self.estimator = None
        self.tracker_2d = None
        self.kalman_filter = None
        self.kf_mean = None
        self.kf_covariance = None
        self.init_mask = None
        self.last_T_world_object = None
        self.last_score = None
        self.error_msg = None
        self.meta = {}


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
        camera_info: Dict[str, str],
        calib_filedir: str,
        object_specs: List[ObjectSpec],
        est_refine_iter: int,
        track_refine_iter: int,
        activate_2d_tracker: bool,
        activate_kalman_filter: bool,
        kf_measurement_noise_scale: float,
        sam_api_endpoint: Optional[str],
        display_scale: float,
        debug: int,
    ):
        self.video_shape = video_shape
        self.sync_channel = sync_channel
        self.pub_channel = pub_channel
        self.camera_info = camera_info
        self.camera_name_list = list(camera_info.values())
        self.calib_filedir = calib_filedir
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self.activate_2d_tracker = activate_2d_tracker
        self.activate_kalman_filter = activate_kalman_filter
        self.kf_measurement_noise_scale = kf_measurement_noise_scale
        self.sam_api_endpoint = sam_api_endpoint
        self.display_scale = display_scale
        self.debug = debug

        self.object_states = [ObjectState(spec=spec) for spec in object_specs]
        self.current_object_idx = 0
        self.latest_frame: Optional[SyncFrame] = None
        self.shutdown = False
        self.factory: Optional[FoundationPoseFactory] = None
        self._closed = False

        self.cam_intr_map = self._load_calib("cam_intr")
        self.cam_extr_map = self._load_calib("cam_extr")

        self.ctx = zmq.Context()
        self.sync_socket = self.ctx.socket(zmq.SUB)
        self.sync_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.sync_socket.setsockopt(zmq.RCVHWM, 1)
        self.sync_socket.setsockopt(zmq.CONFLATE, 1)
        sync_endpoint = channel_name_to_endpoint(sync_channel)
        ensure_ipc_dir(sync_endpoint)
        self.sync_socket.connect(sync_endpoint)

        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.SNDHWM, 1)
        self.pub_socket.setsockopt(zmq.CONFLATE, 1)
        pub_endpoint = channel_name_to_endpoint(pub_channel)
        ensure_ipc_dir(pub_endpoint)
        self.pub_socket.bind(pub_endpoint)

        self.poller = zmq.Poller()
        self.poller.register(self.sync_socket, zmq.POLLIN)

    def _load_calib(self, kind: str) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        for camera_name in self.camera_name_list:
            path = os.path.join(self.calib_filedir, kind, f"{camera_name}.pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {kind} for camera {camera_name}: {path}")
            with open(path, "rb") as f:
                result[camera_name] = np.asarray(pickle.load(f), dtype=np.float64)
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        cv2.destroyAllWindows()
        self.sync_socket.close(linger=0)
        self.pub_socket.close(linger=0)
        self.ctx.term()

    def _decode_sync_message(self, msg: bytes) -> Optional[SyncFrame]:
        data = msgpack.unpackb(msg, object_hook=msgpack_numpy.decode, raw=False)
        synced_data = data.get("synced_data")
        synced_ts = data.get("synced_ts")
        if synced_data is None or synced_ts is None:
            return None

        width, height = self.video_shape
        colors: Dict[str, np.ndarray] = {}
        depths: Dict[str, np.ndarray] = {}
        camera_idx = 0
        for payload in synced_data:
            if not isinstance(payload, dict) or "color" not in payload:
                continue
            if camera_idx >= len(self.camera_name_list):
                break
            camera_name = self.camera_name_list[camera_idx]
            color = payload["color"].reshape(height, width, 3)
            colors[camera_name] = color
            if "depth" in payload:
                depth = payload["depth"].reshape(height, width)
                if depth.dtype == np.uint16:
                    depth = depth.astype(np.float32) / float(payload.get("depth_scale", 1000))
                else:
                    depth = depth.astype(np.float32)
                depth[(depth < 0.001) | ~np.isfinite(depth)] = 0.0
                depths[camera_name] = depth
            camera_idx += 1

        if not colors:
            return None
        return SyncFrame(sync_timestamp=float(synced_ts), color_by_camera=colors, depth_by_camera=depths)

    def _select_camera(self, state: ObjectState, frame: SyncFrame) -> Optional[str]:
        if state.spec.camera_name:
            if state.spec.camera_name not in frame.color_by_camera:
                _logger.error("Configured camera %s is not in latest frame", state.spec.camera_name)
                return None
            return state.spec.camera_name

        available = list(frame.color_by_camera.keys())
        print("\nAvailable cameras:")
        for i, camera_name in enumerate(available):
            print(f"  [{i}] {camera_name}")
        raw = input(f"Select init camera for {state.spec.object_id} [0]: ").strip()
        if raw == "":
            return available[0]
        try:
            return available[int(raw)]
        except Exception:
            _logger.warning("Invalid camera selection %r; using %s", raw, available[0])
            return available[0]

    def _read_mask_override(self, state: ObjectState, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        mask_path = state.spec.mask_path
        if not mask_path:
            raw = input("Optional mask path to import/replace generated mask [Enter to skip]: ").strip()
            mask_path = raw or None
        if not mask_path:
            return None
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            _logger.warning("Could not read mask path %s; falling back to bbox/SAM mask", mask_path)
            return None
        if mask.shape != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8)

    def _request_sam_mask(self, frame_rgb: np.ndarray, bbox_xywh: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        if not self.sam_api_endpoint:
            return None

        tmp_dir = os.path.join(THIS_DIR, "tmp", "object_pose_server")
        os.makedirs(tmp_dir, exist_ok=True)
        frame_path = os.path.join(tmp_dir, "sam_frame.png")
        mask_path = os.path.join(tmp_dir, "sam_mask.png")
        cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        payload = json.dumps({
            "frame_path": frame_path,
            "bbox_xywh": list(map(int, bbox_xywh)),
            "output_mask_path": mask_path,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.sam_api_endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status >= 300:
                    _logger.warning("SAM API returned HTTP %s", resp.status)
                    return None
        except (urllib.error.URLError, TimeoutError) as exc:
            _logger.warning("SAM API request failed: %s", exc)
            return None

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            _logger.warning("SAM API did not write mask to %s", mask_path)
            return None
        return (mask > 0).astype(np.uint8)

    def _build_bbox_mask(
        self,
        frame_rgb: np.ndarray,
        bbox_xywh: Tuple[int, int, int, int],
    ) -> np.ndarray:
        x, y, w, h = bbox_xywh
        mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(mask.shape[1], x1 + max(0, int(w)))
        y2 = min(mask.shape[0], y1 + max(0, int(h)))
        mask[y1:y2, x1:x2] = 1
        return mask

    def _init_object(self, state: ObjectState, frame: SyncFrame) -> bool:
        state.meta["init_attempted"] = True
        if self.factory is None:
            _logger.info("Loading FoundationPose models...")
            self.factory = FoundationPoseFactory(debug=self.debug)

        camera_name = self._select_camera(state, frame)
        if camera_name is None or camera_name not in frame.depth_by_camera:
            state.state = "uninitialized"
            state.error_msg = f"No RGB-D frame available for camera {camera_name}"
            _logger.error(state.error_msg)
            return False

        color = frame.color_by_camera[camera_name]
        depth = frame.depth_by_camera[camera_name]
        window = f"init:{state.spec.object_id}:{camera_name}"
        display = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        if self.display_scale != 1.0:
            display = cv2.resize(display, None, fx=self.display_scale, fy=self.display_scale)
        cv2.imshow(window, display)
        cv2.waitKey(1)

        print(f"\nInitializing object: {state.spec.object_id}")
        print("Draw bbox and press Enter/Space. Press c to cancel.")
        roi = cv2.selectROI(window, display, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window)
        if roi[2] <= 0 or roi[3] <= 0:
            state.meta["init_cancelled"] = True
            state.error_msg = "Initialization cancelled"
            _logger.info("Initialization cancelled for %s", state.spec.object_id)
            return False

        scale = self.display_scale if self.display_scale != 0 else 1.0
        bbox_xywh = tuple(int(round(v / scale)) for v in roi)
        mask = self._read_mask_override(state, color.shape[:2])
        if mask is None:
            mask = self._request_sam_mask(color, bbox_xywh)
        if mask is None:
            mask = self._build_bbox_mask(color, bbox_xywh)

        if mask.sum() < 4:
            state.error_msg = "Init mask has fewer than 4 pixels"
            _logger.error("%s: %s", state.spec.object_id, state.error_msg)
            return False

        try:
            if state.mesh is None:
                state.mesh = self.factory.load_mesh(state.spec)
            state.estimator = self.factory.create_estimator(state.spec, state.mesh)
            cam_K = self.cam_intr_map[camera_name]
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
            state.tracker_2d = Cutie() if self.activate_2d_tracker else Tracker_2D()
            if self.activate_2d_tracker:
                state.tracker_2d.initialize(color, init_info={"mask": mask})
            if self.activate_kalman_filter:
                state.kalman_filter = KalmanFilter6D(self.kf_measurement_noise_scale)
                state.kf_mean, state.kf_covariance = state.kalman_filter.initiate(get_6d_pose_arr_from_mat(pose_cam))
            state.last_T_world_object = inv_transf_np(self.cam_extr_map[camera_name]) @ pose_cam
            state.state = "tracking"
            state.error_msg = None
            state.meta = {"init_bbox_xywh": bbox_xywh, "init_sync_timestamp": frame.sync_timestamp}
            _logger.info("Initialized %s from %s", state.spec.object_id, camera_name)
            return True
        except Exception as exc:
            state.state = "uninitialized"
            state.error_msg = str(exc)
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
            cam_K = self.cam_intr_map[camera_name]
            if self.activate_2d_tracker and state.tracker_2d is not None:
                bbox = state.tracker_2d.track(color)
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
            state.last_T_world_object = inv_transf_np(self.cam_extr_map[camera_name]) @ pose_cam
            state.state = "tracking"
            state.error_msg = None
        except Exception as exc:
            state.state = "lost"
            state.error_msg = str(exc)
            _logger.exception("Tracking failed for %s", state.spec.object_id)

    def _publish(self, sync_timestamp: float, process_time: float) -> None:
        object_payloads = []
        for state in self.object_states:
            object_payloads.append({
                "object_id": state.spec.object_id,
                "T_world_object": state.last_T_world_object,
                "valid": state.valid,
                "state": state.state,
                "score": state.last_score,
                "source_camera": state.source_camera,
                "meta": {
                    **state.meta,
                    **({
                        "error": state.error_msg
                    } if state.error_msg else {}),
                },
            })

        msg = {
            "sync_timestamp": sync_timestamp,
            "server_timestamp": time.time(),
            "object_poses": object_payloads,
            "process_time": process_time,
        }
        self.pub_socket.send(msgpack.packb(msg, default=msgpack_numpy.encode))

    def _draw_status(self, frame: SyncFrame) -> None:
        if not frame.color_by_camera:
            return
        current = self.object_states[self.current_object_idx]
        camera_name = current.source_camera or current.spec.camera_name or next(iter(frame.color_by_camera.keys()))
        img = frame.color_by_camera.get(camera_name)
        if img is None:
            img = next(iter(frame.color_by_camera.values()))
            camera_name = next(iter(frame.color_by_camera.keys()))
        disp = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
        y = 24
        cv2.putText(disp, f"camera={camera_name} sync={frame.sync_timestamp:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        y += 26
        for i, state in enumerate(self.object_states):
            marker = ">" if i == self.current_object_idx else " "
            text = f"{marker} {state.spec.object_id}: {state.state}"
            color = (0, 220, 0) if state.valid else ((0, 200, 255) if state.state in ("uninitialized", "lost") else
                                                     (0, 0, 255))
            cv2.putText(disp, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 24
        y += 6
        cv2.putText(disp, "[/]: select  i:init  r:reset current  a:reset all  q:quit", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if self.display_scale != 1.0:
            disp = cv2.resize(disp, None, fx=self.display_scale, fy=self.display_scale)
        cv2.imshow("object_pose_server", disp)

    def _handle_key(self, key: int) -> None:
        if key < 0:
            return
        key &= 0xFF
        if key == ord("q") or key == 27:
            self.shutdown = True
        elif key == ord("["):
            self.current_object_idx = (self.current_object_idx - 1) % len(self.object_states)
        elif key == ord("]"):
            self.current_object_idx = (self.current_object_idx + 1) % len(self.object_states)
        elif key == ord("r"):
            state = self.object_states[self.current_object_idx]
            _logger.info("Resetting object %s", state.spec.object_id)
            state.reset_runtime()
        elif key == ord("a"):
            _logger.info("Resetting all objects")
            for state in self.object_states:
                state.reset_runtime()
            self.current_object_idx = 0
        elif key == ord("i") and self.latest_frame is not None:
            self._init_object(self.object_states[self.current_object_idx], self.latest_frame)

    def _maybe_auto_init(self) -> None:
        if self.latest_frame is None:
            return
        for idx, state in enumerate(self.object_states):
            if state.state == "uninitialized" and not state.meta.get("init_attempted"):
                self.current_object_idx = idx
                self._init_object(state, self.latest_frame)
                break

    def run(self) -> None:
        _logger.info("Object pose server started")
        _logger.info("Subscribed to sync channel: %s", self.sync_channel)
        _logger.info("Publishing object poses on: %s", self.pub_channel)

        while not self.shutdown:
            start = time.time()
            socks = dict(self.poller.poll(timeout=10))
            if self.sync_socket in socks:
                msg = self.sync_socket.recv(zmq.NOBLOCK)
                frame = self._decode_sync_message(msg)
                if frame is not None:
                    self.latest_frame = frame
                    self._maybe_auto_init()
                    for state in self.object_states:
                        self._track_object(state, frame)
                    self._publish(frame.sync_timestamp, time.time() - start)
                    self._draw_status(frame)

            key = cv2.waitKey(1)
            self._handle_key(key)


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
    parser = argparse.ArgumentParser(description="Interactive Object Pose Estimation Server")
    parser.add_argument("--server.video_shape", type=parse_video_shape, default="1280x720")
    parser.add_argument("--server.sync_channel", type=str, required=True)
    parser.add_argument("--server.pub_channel", type=str, required=True)
    parser.add_argument("--camera_info", type=parse_camera_info, required=True)
    parser.add_argument("--calib_filedir", type=str, required=True)
    parser.add_argument("--object_config", type=str, required=True)
    parser.add_argument("--est_refine_iter", type=int, default=10)
    parser.add_argument("--track_refine_iter", type=int, default=5)
    parser.add_argument("--activate_2d_tracker", action="store_true")
    parser.add_argument("--activate_kalman_filter", action="store_true")
    parser.add_argument("--kf_measurement_noise_scale", type=float, default=0.05)
    parser.add_argument("--sam_api_endpoint", type=str, default=None)
    parser.add_argument("--display_scale", type=float, default=0.75)
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()

    object_specs = load_object_specs(args.object_config)
    server = ObjectPoseServer(
        video_shape=getattr(args, "server.video_shape"),
        sync_channel=getattr(args, "server.sync_channel"),
        pub_channel=getattr(args, "server.pub_channel"),
        camera_info=args.camera_info,
        calib_filedir=os.path.abspath(os.path.expanduser(args.calib_filedir)),
        object_specs=object_specs,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        activate_2d_tracker=args.activate_2d_tracker,
        activate_kalman_filter=args.activate_kalman_filter,
        kf_measurement_noise_scale=args.kf_measurement_noise_scale,
        sam_api_endpoint=args.sam_api_endpoint,
        display_scale=args.display_scale,
        debug=args.debug,
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
