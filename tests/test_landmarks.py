"""Unit tests for landmark structures, coordinate conversions, and regional grouping."""

import numpy as np
import pytest

from src.landmarks import FaceLandmarks, LandmarkPoint, select_landmark_regions


def create_synthetic_landmarks(num_points: int = 478, width: int = 640, height: int = 480) -> FaceLandmarks:
    """Helper creating synthetic FaceLandmarks with known normalized and pixel coords."""
    pts_norm = np.zeros((num_points, 3), dtype=np.float32)
    # Generate points within [0.1, 0.9] range
    for i in range(num_points):
        pts_norm[i, 0] = 0.5 + 0.3 * np.cos(i)
        pts_norm[i, 1] = 0.5 + 0.3 * np.sin(i)
        pts_norm[i, 2] = 0.05 * np.sin(i * 2)

    pts_px = np.zeros_like(pts_norm)
    pts_px[:, 0] = pts_norm[:, 0] * width
    pts_px[:, 1] = pts_norm[:, 1] * height
    pts_px[:, 2] = pts_norm[:, 2] * width

    return FaceLandmarks(
        points_norm=pts_norm,
        points_px=pts_px,
        image_width=width,
        image_height=height,
        num_landmarks=num_points,
    )


def test_facelandmarks_properties():
    """Verify coordinate accessors and bounds."""
    w, h = 800, 600
    lm = create_synthetic_landmarks(478, w, h)

    assert lm.num_landmarks == 478
    assert lm.image_width == w
    assert lm.image_height == h
    assert len(lm.x_px) == 478
    assert len(lm.y_px) == 478
    assert len(lm.z_px) == 478


def test_get_point_valid():
    """Verify LandmarkPoint extraction preserves normalized and pixel mappings."""
    w, h = 640, 480
    lm = create_synthetic_landmarks(478, w, h)
    pt = lm.get_point(0)

    assert isinstance(pt, LandmarkPoint)
    assert pt.index == 0
    assert np.isclose(pt.x_px, pt.x_norm * w)
    assert np.isclose(pt.y_px, pt.y_norm * h)


def test_get_point_out_of_bounds():
    """IndexError should be raised for non-existent landmark indices."""
    lm = create_synthetic_landmarks(100)
    with pytest.raises(IndexError):
        lm.get_point(100)
    with pytest.raises(IndexError):
        lm.get_point(-1)


def test_get_points_subset():
    """Ensure get_points_subset returns correct slice and handles invalid indices gracefully."""
    lm = create_synthetic_landmarks(478)
    subset = lm.get_points_subset([1, 152, 9999])  # 9999 is invalid
    assert subset.shape == (2, 3)

    empty_subset = lm.get_points_subset([9000, 9001])
    assert empty_subset.shape == (0, 3)


def test_select_landmark_regions():
    """Verify region grouping includes eyes, eyebrows, cheeks, mouth, jawline, and midline."""
    lm = create_synthetic_landmarks(478)
    regions = select_landmark_regions(lm)

    expected_keys = {"eyes", "eyebrows", "cheeks", "mouth", "jawline", "midline"}
    assert expected_keys.issubset(regions.keys())

    for reg in ["eyes", "eyebrows", "cheeks", "mouth", "jawline"]:
        assert "right_px" in regions[reg]
        assert "left_px" in regions[reg]
        assert len(regions[reg]["right_px"]) == len(regions[reg]["left_px"])
        assert len(regions[reg]["right_px"]) > 0

    assert "midline_px" in regions["midline"]
    assert len(regions["midline"]["midline_px"]) > 0
