import numpy as np
import pytest

from dev_fn.transform.rotation_np import rotvec_to_rotmat_np
from pose_validity import PoseValidityConfig, evaluate_pose_validity, is_rigid_transform


def _pose(x=0.0, angle_deg=0.0):
    pose = np.eye(4)
    pose[:3, :3] = rotvec_to_rotmat_np(np.array([0.0, 0.0, np.radians(angle_deg)]))
    pose[:3, 3] = (x, 0.0, 1.0)
    return pose


def _check(*, pose=None, observed=None, rendered=None, mask=None, **kwargs):
    return evaluate_pose_validity(
        _pose() if pose is None else pose,
        np.ones((10, 10)) if observed is None else observed,
        np.ones((10, 10)) if rendered is None else rendered,
        tracking_mask=mask, config=PoseValidityConfig(), **kwargs,
    )


def test_aligned_pose_with_small_motion_passes():
    result = _check(pose=_pose(0.02, 10.0), previous_pose=_pose(), mask=np.ones((10, 10)), require_mask=True)
    assert result["valid"]
    assert result["mask_iou"] == 1.0
    assert result["depth_inlier_fraction"] == 1.0


@pytest.mark.parametrize("bad_pose", [
    np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan), np.full((4, 4), np.inf),
    np.diag([1.0, 1.0, -1.0, 1.0]), np.diag([2.0, 1.0, 1.0, 1.0]),
    np.diag([1.0, 1.0, 1.0, 2.0]),
])
def test_malformed_transforms_are_rejected(bad_pose):
    assert not is_rigid_transform(bad_pose)
    assert _check(pose=bad_pose)["reason"] == "invalid_transform"


@pytest.mark.parametrize("pose,reason", [
    (_pose(0.051), "translation_jump"), (_pose(angle_deg=31.0), "rotation_jump"),
    (_pose(angle_deg=180.0), "rotation_jump"),
])
def test_large_motion_is_rejected_even_with_perfect_image_support(pose, reason):
    assert _check(pose=pose, previous_pose=_pose())["reason"] == reason


def test_rotation_wraparound_uses_relative_rotation():
    result = _check(pose=_pose(angle_deg=-179.0), previous_pose=_pose(angle_deg=179.0))
    assert result["valid"]
    assert result["rotation_jump_deg"] == pytest.approx(2.0)


def test_mask_overlap_must_reach_default_threshold():
    # Keep foreground support above 64 pixels on both sides of the IoU boundary.
    depth = np.ones((20, 10))
    mask = np.ones_like(depth)
    mask.flat[:121] = 0
    assert _check(mask=mask, observed=depth, rendered=depth)["reason"] == "low_mask_iou"
    mask.flat[120] = 1
    result = _check(mask=mask, observed=depth, rendered=depth)
    assert result["mask_iou"] == pytest.approx(0.40)
    assert result["valid"]


@pytest.mark.parametrize("mask,reason", [
    (None, "mask_missing"), (np.zeros((10, 10)), "insufficient_mask_pixels"),
    (np.ones((9, 10)), "invalid_mask"), (np.full((10, 10), np.nan), "invalid_mask"),
])
def test_tracking_needs_a_usable_current_mask(mask, reason):
    assert _check(mask=mask, require_mask=True)["reason"] == reason


@pytest.mark.parametrize("bad_depth", [0.0, np.nan, np.inf, 1.05, 0.95])
def test_missing_depth_and_occlusions_count_against_agreement(bad_depth):
    depth = np.ones((10, 10))
    depth.flat[:31] = bad_depth
    assert _check(observed=depth)["reason"] == "insufficient_depth_agreement"
    depth.flat[30] = 1.0
    assert _check(observed=depth)["valid"]


def test_few_matching_pixels_cannot_validate_a_pose():
    rendered = np.zeros((10, 10))
    rendered.flat[:63] = 1.0
    assert _check(rendered=rendered)["reason"] == "insufficient_rendered_pixels"


def test_isolated_pose_without_a_mask_still_requires_depth_support():
    assert _check()["valid"]
    assert _check(observed=np.full((10, 10), 2.0))["reason"] == "insufficient_depth_agreement"


@pytest.mark.parametrize("kwargs", [
    {"min_mask_iou": 0.0}, {"min_mask_iou": 1.1}, {"min_mask_iou": np.nan},
    {"max_depth_error_m": 0.0}, {"min_depth_inlier_fraction": -0.1},
    {"max_translation_jump_m": np.inf}, {"max_rotation_jump_deg": 181.0},
    {"min_pixels": 0}, {"min_pixels": 1.5},
])
def test_invalid_thresholds_fail_at_startup(kwargs):
    with pytest.raises(ValueError):
        PoseValidityConfig(**kwargs)
