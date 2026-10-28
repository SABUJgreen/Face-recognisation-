"""Unit tests for geometric facial symmetry analysis on synthetic and controlled landmarks."""

import numpy as np
import pytest

from src.config import SYMMETRY_PAIRS_BY_REGION, SymmetryConfig
from src.face_alignment import align_face
from src.head_pose import HeadPoseResult
from src.symmetry import (
    calculate_pair_error,
    calculate_symmetry,
    compute_heuristic_symmetry_score,
    estimate_facial_midline_x,
)


def create_synthetic_symmetric_face(iod: float = 100.0, center: tuple = (300.0, 300.0)) -> np.ndarray:
    """Generate a mathematically perfectly symmetric face configuration."""
    pts = np.zeros((478, 3), dtype=np.float32)
    cx, cy = center

    # Eye landmarks: exactly symmetric across vertical line x = cx
    dx = iod / 2.0
    for idx in [33, 133, 159, 145]:
        pts[idx, :2] = [cx - dx, cy - 30.0]
    for idx in [263, 362, 386, 374]:
        pts[idx, :2] = [cx + dx, cy - 30.0]

    # Midline landmarks: placed exactly on x = cx
    for m_idx in [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 13, 14, 17, 18, 200, 199, 175, 152]:
        pts[m_idx, :2] = [cx, cy + (m_idx % 20) * 5.0]

    # All defined regional pairs: assign mirrored symmetric positions (x = cx - dist, x = cx + dist, same y)
    for region, pairs in SYMMETRY_PAIRS_BY_REGION.items():
        for i, (r_idx, l_idx) in enumerate(pairs):
            dist_x = 20.0 + (i * 5.0)
            offset_y = (i * 8.0) - 40.0
            pts[r_idx, :2] = [cx - dist_x, cy + offset_y]
            pts[l_idx, :2] = [cx + dist_x, cy + offset_y]

    return pts


def test_perfect_symmetry():
    """A mathematically symmetric face must yield normalized error ~ 0.0 and score == 100.0%."""
    pts = create_synthetic_symmetric_face(iod=100.0, center=(320.0, 240.0))
    res = calculate_symmetry(pts)

    assert res.is_valid is True
    assert np.isclose(res.overall_normalized_asymmetry, 0.0, atol=1e-5)
    assert np.isclose(res.symmetry_score, 100.0, atol=1e-3)
    assert res.valid_pair_count > 0

    for reg, err in res.regional_asymmetry.items():
        assert np.isclose(err, 0.0, atol=1e-5)


def test_known_asymmetry():
    """Artificially shifting one side of the jaw and cheek must produce a measurable error."""
    pts = create_synthetic_symmetric_face(iod=100.0, center=(320.0, 240.0))

    # Introduce a 10px lateral asymmetry on left cheek landmarks
    for _, l_idx in SYMMETRY_PAIRS_BY_REGION["cheeks"]:
        pts[l_idx, 0] += 10.0

    res = calculate_symmetry(pts)

    assert res.is_valid is True
    assert res.overall_normalized_asymmetry > 0.0
    assert res.symmetry_score < 100.0
    # Cheeks asymmetry should be noticeably higher than eyes asymmetry (which is 0)
    assert res.regional_asymmetry["cheeks"] > res.regional_asymmetry["eyes"]
    assert np.isclose(res.regional_asymmetry["eyes"], 0.0, atol=1e-5)


def test_reflection_calculation():
    """Verify reflection across vertical midline x = c."""
    # Point L = (150, 50), midline x = 100. Reflected point should be (50, 50).
    pt_r = np.array([50.0, 50.0])
    pt_l = np.array([150.0, 50.0])
    midline = 100.0
    ref_dist = 100.0

    err_abs, err_norm, refl = calculate_pair_error(pt_r, pt_l, midline, ref_dist)

    assert np.isclose(refl[0], 50.0)
    assert np.isclose(refl[1], 50.0)
    assert np.isclose(err_abs, 0.0)
    assert np.isclose(err_norm, 0.0)


def test_heuristic_score_properties():
    """Check score monotonicity and bounding."""
    score_0 = compute_heuristic_symmetry_score(0.0)
    score_small = compute_heuristic_symmetry_score(0.05)
    score_large = compute_heuristic_symmetry_score(0.20)

    assert score_0 == 100.0
    assert 0.0 < score_large < score_small < score_0
    assert 0.0 <= score_large <= 100.0


def test_unsuitable_non_frontal_pose():
    """Unsuitable pose must result in is_valid = False and NaN scores."""
    pts = create_synthetic_symmetric_face()
    mock_pose = HeadPoseResult(
        yaw_degrees=30.0,  # Turned face
        pitch_degrees=0.0,
        roll_degrees=0.0,
        rotation_matrix=np.eye(3),
        translation_vector=np.zeros((3, 1)),
        success=True,
        reprojection_error=1.0,
        is_frontal=False,
    )

    res = calculate_symmetry(pts, head_pose=mock_pose)
    assert res.is_valid is False
    assert np.isnan(res.overall_normalized_asymmetry)
    assert np.isnan(res.symmetry_score)
    assert res.invalidity_reason is not None
