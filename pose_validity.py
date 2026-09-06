"""Geometric acceptance checks for a candidate pose in the observation camera."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from dev_fn.transform.rotation_np import rotmat_to_rotvec_np


@dataclass(frozen=True)
class PoseValidityConfig:
    min_mask_iou: float = 0.40
    max_depth_error_m: float = 0.02
    min_depth_inlier_fraction: float = 0.70
    max_translation_jump_m: float = 0.05
    max_rotation_jump_deg: float = 30.0
    min_pixels: int = 64

    def __post_init__(self) -> None:
        for name in ("min_mask_iou", "min_depth_inlier_fraction"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        for name in ("max_depth_error_m", "max_translation_jump_m", "max_rotation_jump_deg"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_rotation_jump_deg > 180.0:
            raise ValueError("max_rotation_jump_deg must not exceed 180")
        if isinstance(self.min_pixels, bool) or not isinstance(self.min_pixels, int) or self.min_pixels < 1:
            raise ValueError("min_pixels must be a positive integer")


def is_rigid_transform(pose: Any) -> bool:
    try:
        transform = np.asarray(pose, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return False
    rotation = transform[:3, :3]
    return bool(
        np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5, rtol=0.0)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=0.0)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3, rtol=0.0)
    )


def evaluate_pose_validity(
    pose: np.ndarray,
    observed_depth: np.ndarray,
    rendered_depth: Optional[np.ndarray],
    *,
    config: PoseValidityConfig,
    tracking_mask: Optional[np.ndarray] = None,
    require_mask: bool = False,
    previous_pose: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Require image/depth support and bounded motion from the last accepted pose.

    Rendered depth must use the original object's pose and the observation
    camera's intrinsics. Missing depth counts against support; occluded mesh
    pixels also count against it, deliberately rejecting weak observations.
    """
    result: Dict[str, Any] = {"valid": False, "reason": "invalid_transform"}
    if not is_rigid_transform(pose):
        return result
    pose = np.asarray(pose, dtype=np.float64)
    if previous_pose is not None:
        if not is_rigid_transform(previous_pose):
            result["reason"] = "invalid_previous_transform"
            return result
        previous_pose = np.asarray(previous_pose, dtype=np.float64)
        translation_jump = float(np.linalg.norm(pose[:3, 3] - previous_pose[:3, 3]))
        # SO(3) geodesic angle, computed from the relative rotation.
        relative_rotation = previous_pose[:3, :3].T @ pose[:3, :3]
        rotation_jump = float(np.degrees(np.linalg.norm(rotmat_to_rotvec_np(relative_rotation))))
        result.update(translation_jump_m=translation_jump, rotation_jump_deg=rotation_jump)
        if translation_jump > config.max_translation_jump_m:
            result["reason"] = "translation_jump"
            return result
        if rotation_jump > config.max_rotation_jump_deg:
            result["reason"] = "rotation_jump"
            return result

    observed_depth = np.asarray(observed_depth)
    rendered_depth = np.asarray(rendered_depth)
    if observed_depth.ndim != 2 or rendered_depth.shape != observed_depth.shape:
        result["reason"] = "invalid_depth"
        return result
    rendered_mask = np.isfinite(rendered_depth) & (rendered_depth > 0.001)
    rendered_pixels = int(np.count_nonzero(rendered_mask))
    result["rendered_pixels"] = rendered_pixels
    if rendered_pixels < config.min_pixels:
        result["reason"] = "insufficient_rendered_pixels"
        return result

    if tracking_mask is None:
        result["mask_state"] = "unavailable"
        if require_mask:
            result["reason"] = "mask_missing"
            return result
    else:
        mask = np.asarray(tracking_mask)
        if mask.shape != observed_depth.shape or not np.isfinite(mask).all():
            result["reason"] = "invalid_mask"
            return result
        mask = mask > 0
        mask_pixels = int(np.count_nonzero(mask))
        result.update(mask_state="available", mask_pixels=mask_pixels)
        if mask_pixels < config.min_pixels:
            result["reason"] = "insufficient_mask_pixels"
            return result
        mask_iou = float(np.count_nonzero(mask & rendered_mask) / np.count_nonzero(mask | rendered_mask))
        result["mask_iou"] = mask_iou
        if mask_iou < config.min_mask_iou:
            result["reason"] = "low_mask_iou"
            return result

    depth_support = rendered_mask & np.isfinite(observed_depth) & (observed_depth > 0.001)
    depth_errors = np.abs(observed_depth[depth_support] - rendered_depth[depth_support])
    inliers = int(np.count_nonzero(depth_errors <= config.max_depth_error_m))
    # Denominator includes missing depth and occluded/background pixels.
    inlier_fraction = float(inliers / rendered_pixels)
    result.update(depth_inlier_pixels=inliers, depth_inlier_fraction=inlier_fraction)
    if inliers < config.min_pixels or inlier_fraction < config.min_depth_inlier_fraction:
        result["reason"] = "insufficient_depth_agreement"
        return result
    result.update(valid=True, reason="ok")
    return result
