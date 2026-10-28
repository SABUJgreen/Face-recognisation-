"""Unit tests for facial alignment, 2D similarity transform, and pose gating."""

import numpy as np
import pytest

from src.face_alignment import align_face, calculate_eye_centers
from src.head_pose import HeadPoseResult


def create_synthetic_frontal_face(
    iod: float = 120.0,
    center: tuple = (320.0, 240.0),
    tilt_angle_deg: float = 0.0
) -> np.ndarray:
    """Generate a clean synthetic frontal face layout with known eye centers."""
    pts = np.zeros((478, 3), dtype=np.float32)

    cx, cy = center
    angle_rad = np.radians(tilt_angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Eye offsets relative to center
    dx = iod / 2.0
    # Right eye (viewer's left, -dx), Left eye (viewer's right, +dx)
    r_eye_local = np.array([-dx, 0.0])
    l_eye_local = np.array([dx, 0.0])

    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    r_eye_rot = np.dot(R, r_eye_local) + np.array([cx, cy])
    l_eye_rot = np.dot(R, l_eye_local) + np.array([cx, cy])

    # Assign eye landmarks (33, 133, 159, 145 for right; 263, 362, 386, 374 for left)
    for idx in [33, 133, 159, 145]:
        pts[idx, :2] = r_eye_rot
    for idx in [263, 362, 386, 374]:
        pts[idx, :2] = l_eye_rot

    # Assign midline points along central axis perpendicular to eye axis
    # Central axis unit vector: [-sin_a, cos_a]
    midline_dir = np.array([-sin_a, cos_a])
    for i, m_idx in enumerate([10, 168, 1, 152]):
        dist = (i - 1.5) * 40.0  # from forehead to chin
        pt_m = np.array([cx, cy]) + midline_dir * dist
        pts[m_idx, :2] = pt_m

    return pts


def test_calculate_eye_centers():
    """Verify calculated eye centroids and inter-ocular distance match generation parameters."""
    pts = create_synthetic_frontal_face(iod=100.0, center=(300.0, 200.0))
    r_c, l_c, iod = calculate_eye_centers(pts)

    assert np.isclose(iod, 100.0, atol=1e-4)
    assert np.isclose(r_c[0], 250.0, atol=1e-4)
    assert np.isclose(l_c[0], 350.0, atol=1e-4)


def test_in_plane_rotation_correction():
    """Ensure align_face detects and cancels in-plane head tilt."""
    tilt = 25.0
    pts = create_synthetic_frontal_face(iod=100.0, center=(320.0, 240.0), tilt_angle_deg=tilt)
    res = align_face(pts, target_interocular_distance=100.0)

    assert res.is_suitable_for_symmetry is True
    # Rotation angle recovered should equal tilt angle
    assert np.isclose(res.rotation_angle_deg, tilt, atol=1e-3)

    # In aligned space, eye centers must be strictly horizontal (same y coordinate)
    r_c_align, l_c_align, iod_align = calculate_eye_centers(res.aligned_points)
    assert np.isclose(r_c_align[1], l_c_align[1], atol=1e-4)
    assert np.isclose(iod_align, 100.0, atol=1e-4)


def test_translation_invariance():
    """Aligning a face shifted by (+100, -50) must yield identical aligned relative coordinates."""
    pts1 = create_synthetic_frontal_face(iod=100.0, center=(200.0, 200.0))
    pts2 = create_synthetic_frontal_face(iod=100.0, center=(300.0, 150.0))

    res1 = align_face(pts1, target_interocular_distance=100.0)
    res2 = align_face(pts2, target_interocular_distance=100.0)

    # Key landmark positions in aligned space must match
    populated_indices = [33, 133, 159, 145, 263, 362, 386, 374, 10, 168, 1, 152]
    diff = np.abs(res1.aligned_points[populated_indices, :2] - res2.aligned_points[populated_indices, :2])
    assert np.max(diff) < 1e-4


def test_non_frontal_pose_rejection():
    """Face with out-of-plane yaw exceeding threshold must be marked unsuitable for symmetry."""
    pts = create_synthetic_frontal_face(iod=100.0)
    mock_pose = HeadPoseResult(
        yaw_degrees=28.0,  # Exceeds 12.0 deg
        pitch_degrees=2.0,
        roll_degrees=0.0,
        rotation_matrix=np.eye(3),
        translation_vector=np.zeros((3, 1)),
        success=True,
        reprojection_error=1.2,
        is_frontal=False,
    )

    res = align_face(pts, head_pose=mock_pose)
    assert res.is_suitable_for_symmetry is False
    assert "yaw" in res.suitability_reason.lower()
