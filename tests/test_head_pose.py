"""Unit tests for head pose estimation using solvePnP and synthetic projections."""

import cv2
import numpy as np
import pytest

from src.config import POSE_REFERENCE_3D_POINTS, HeadPoseConfig
from src.head_pose import (
    compute_reprojection_error,
    estimate_head_pose,
    get_default_camera_matrix,
    rotation_matrix_to_euler_angles,
)


def test_camera_matrix_structure():
    """Verify standard pinhole camera matrix layout."""
    w, h = 640, 480
    K = get_default_camera_matrix(w, h)

    assert K.shape == (3, 3)
    assert K[0, 0] == float(w)  # fx
    assert K[1, 1] == float(w)  # fy
    assert K[0, 2] == w / 2.0   # cx
    assert K[1, 2] == h / 2.0   # cy
    assert K[2, 2] == 1.0


def test_euler_angle_conventions_identity():
    """Identity rotation matrix must produce (0.0, 0.0, 0.0) Euler angles."""
    R = np.eye(3, dtype=np.float64)
    yaw, pitch, roll = rotation_matrix_to_euler_angles(R)

    assert np.isclose(yaw, 0.0, atol=1e-5)
    assert np.isclose(pitch, 0.0, atol=1e-5)
    assert np.isclose(roll, 0.0, atol=1e-5)


def test_synthetic_frontal_solvepnp():
    """Project canonical 3D face points with identity rotation and verify solvePnP recovery."""
    w, h = 640, 480
    K = get_default_camera_matrix(w, h)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Ground truth: Pure frontal face at Z = 600 mm away
    rvec_gt = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    tvec_gt = np.array([0.0, 0.0, 600.0], dtype=np.float64)

    # Create synthetic 2D observations using cv2.projectPoints
    model_pts = np.array(list(POSE_REFERENCE_3D_POINTS.values()), dtype=np.float64)
    proj_2d, _ = cv2.projectPoints(model_pts, rvec_gt, tvec_gt, K, dist_coeffs)
    proj_2d = proj_2d.reshape(-1, 2)

    # Build landmark array indexed by canonical MediaPipe indices
    max_idx = max(POSE_REFERENCE_3D_POINTS.keys()) + 1
    landmarks_px = np.zeros((max_idx, 3), dtype=np.float32)

    for i, idx in enumerate(POSE_REFERENCE_3D_POINTS.keys()):
        landmarks_px[idx, 0] = proj_2d[i, 0]
        landmarks_px[idx, 1] = proj_2d[i, 1]

    result = estimate_head_pose(landmarks_px, image_width=w, image_height=h, camera_matrix=K)

    assert result.success is True
    assert result.is_frontal is True
    assert abs(result.yaw_degrees) < 1.0
    assert abs(result.pitch_degrees) < 1.0
    assert abs(result.roll_degrees) < 1.0
    assert result.reprojection_error < 0.1  # Virtually zero pixel error


def test_synthetic_turned_face_solvepnp():
    """Verify solvePnP detects out-of-plane yaw rotation accurately."""
    w, h = 640, 480
    K = get_default_camera_matrix(w, h)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Ground truth: Turned 25 degrees around Y axis (yaw)
    yaw_target_deg = 25.0
    yaw_target_rad = np.radians(yaw_target_deg)
    # Rotation vector around Y: (0, angle, 0)
    rvec_gt = np.array([0.0, yaw_target_rad, 0.0], dtype=np.float64)
    tvec_gt = np.array([0.0, 0.0, 600.0], dtype=np.float64)

    model_pts = np.array(list(POSE_REFERENCE_3D_POINTS.values()), dtype=np.float64)
    proj_2d, _ = cv2.projectPoints(model_pts, rvec_gt, tvec_gt, K, dist_coeffs)
    proj_2d = proj_2d.reshape(-1, 2)

    max_idx = max(POSE_REFERENCE_3D_POINTS.keys()) + 1
    landmarks_px = np.zeros((max_idx, 3), dtype=np.float32)
    for i, idx in enumerate(POSE_REFERENCE_3D_POINTS.keys()):
        landmarks_px[idx, 0] = proj_2d[i, 0]
        landmarks_px[idx, 1] = proj_2d[i, 1]

    result = estimate_head_pose(landmarks_px, image_width=w, image_height=h, camera_matrix=K)

    assert result.success is True
    assert np.isclose(abs(result.yaw_degrees), yaw_target_deg, atol=2.0)
    assert result.is_frontal is False  # 25 degrees exceeds frontal threshold (12 deg)


def test_insufficient_points_failure():
    """Head pose should fail gracefully if landmark array has fewer than 4 points."""
    landmarks_sparse = np.zeros((3, 3), dtype=np.float32)
    result = estimate_head_pose(landmarks_sparse, image_width=640, image_height=480)

    assert result.success is False
    assert result.is_frontal is False
    assert result.error_message is not None
