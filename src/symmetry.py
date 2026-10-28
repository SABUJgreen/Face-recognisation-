"""Geometric facial symmetry analysis comparing reflected bilateral landmarks."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.config import (
    MIDLINE_LANDMARK_INDICES,
    REGIONAL_WEIGHTS,
    SYMMETRY_PAIRS_BY_REGION,
    SymmetryConfig,
)
from src.face_alignment import AlignmentResult, align_face
from src.head_pose import HeadPoseResult
from src.landmarks import FaceLandmarks


DISCLAIMER_TEXT = (
    "NOTICE: FaceScope calculates purely geometric Euclidean landmark symmetry. "
    "This tool is designed strictly for computer vision research and geometric posture analysis. "
    "It is NOT a medical diagnostic instrument, clinical tool, or measure of personal attractiveness."
)


@dataclass
class PairSymmetryDetail:
    """Detailed geometric measurement for a single bilateral landmark pair."""
    region: str
    right_index: int
    left_index: int
    right_pt: Tuple[float, float]
    left_pt: Tuple[float, float]
    reflected_left_pt: Tuple[float, float]
    absolute_error: float            # Euclidean error in aligned coordinate units
    normalized_error: float          # Normalized by reference inter-ocular distance


@dataclass
class SymmetryResult:
    """Structured result of facial symmetry computation."""
    overall_normalized_asymmetry: float   # Weighted mean normalized geometric asymmetry (0 = perfect symmetry)
    symmetry_score: float                 # Heuristic display score in range [0, 100]%
    regional_asymmetry: Dict[str, float]  # Region -> normalized asymmetry error
    pair_details: List[PairSymmetryDetail]
    midline_x: float
    reference_distance: float
    valid_pair_count: int
    total_pair_count: int
    is_valid: bool
    invalidity_reason: Optional[str] = None
    disclaimer: str = DISCLAIMER_TEXT


def estimate_facial_midline_x(
    aligned_points: np.ndarray,
    midline_indices: List[int] = MIDLINE_LANDMARK_INDICES
) -> float:
    """Estimate the central vertical facial symmetry axis (x = c) from aligned midline landmarks.

    Uses a robust trimmed mean / median of central nasal and oral midline landmarks.

    Args:
        aligned_points: Aligned landmark coordinates of shape (N, 2 or 3).
        midline_indices: Landmark indices lying along the anatomical sagittal plane.

    Returns:
        float: Estimated X coordinate of the facial midline.
    """
    n_pts = len(aligned_points)
    valid_indices = [idx for idx in midline_indices if idx < n_pts]

    if not valid_indices:
        return 0.0

    x_coords = aligned_points[valid_indices, 0]
    # Median is robust against unilateral nasal deviations or expression asymmetries
    midline_x = float(np.median(x_coords))
    return midline_x


def calculate_pair_error(
    pt_right: np.ndarray,
    pt_left: np.ndarray,
    midline_x: float,
    reference_distance: float
) -> Tuple[float, float, Tuple[float, float]]:
    """Compute geometric asymmetry error for one bilateral landmark pair.

    For midline x = c and point L = (xL, yL):
    Reflected point: L_reflected = (2c - xL, yL)
    Geometric error: || L_reflected - R || = sqrt((2c - xL - xR)^2 + (yL - yR)^2)

    Args:
        pt_right: Coordinates of subject's right landmark (xR, yR).
        pt_left: Coordinates of subject's left landmark (xL, yL).
        midline_x: Vertical symmetry axis X coordinate c.
        reference_distance: Normalization reference distance (inter-ocular distance).

    Returns:
        Tuple of (absolute_error, normalized_error, reflected_left_pt_xy).
    """
    x_r, y_r = float(pt_right[0]), float(pt_right[1])
    x_l, y_l = float(pt_left[0]), float(pt_left[1])

    # Reflect left point across vertical axis x = midline_x
    x_reflected = 2.0 * midline_x - x_l
    y_reflected = y_l

    error_abs = float(np.sqrt((x_reflected - x_r) ** 2 + (y_reflected - y_r) ** 2))

    if reference_distance <= 1e-6:
        error_norm = float("inf")
    else:
        error_norm = error_abs / reference_distance

    return error_abs, error_norm, (x_reflected, y_reflected)


def compute_heuristic_symmetry_score(
    normalized_asymmetry: float,
    decay_factor: float = 5.0
) -> float:
    """Compute a transparent 0-100 heuristic display score from normalized geometric asymmetry.

    Transformation:
    score = 100 * exp(-decay_factor * normalized_asymmetry)

    Properties:
    - 0 normalized asymmetry -> 100.0%
    - 0.05 normalized asymmetry (5% of inter-ocular distance) -> ~77.8%
    - 0.10 normalized asymmetry -> ~60.6%
    - Monotonically decreasing, strictly bounded in [0.0, 100.0].

    Note: This is a mathematical visualization aid and not a clinical diagnosis.
    """
    if normalized_asymmetry < 0.0:
        return 100.0
    score = 100.0 * np.exp(-decay_factor * normalized_asymmetry)
    return float(np.clip(score, 0.0, 100.0))


def calculate_symmetry(
    landmarks: Union[FaceLandmarks, np.ndarray],
    alignment_result: Optional[AlignmentResult] = None,
    head_pose: Optional[HeadPoseResult] = None,
    config: Optional[SymmetryConfig] = None
) -> SymmetryResult:
    """Calculate regional and overall geometric facial asymmetry.

    Args:
        landmarks: Raw FaceLandmarks or landmark coordinate array.
        alignment_result: Optional pre-computed AlignmentResult. If None, runs align_face().
        head_pose: Optional HeadPoseResult for pose suitability gating.
        config: Optional SymmetryConfig with custom pair sets and weights.

    Returns:
        SymmetryResult containing regional errors, overall index, and pair details.
    """
    if config is None:
        config = SymmetryConfig()

    # Step 1: Ensure alignment
    if alignment_result is None:
        alignment_result = align_face(landmarks, head_pose=head_pose)

    if not alignment_result.is_suitable_for_symmetry:
        return SymmetryResult(
            overall_normalized_asymmetry=float("nan"),
            symmetry_score=float("nan"),
            regional_asymmetry={},
            pair_details=[],
            midline_x=alignment_result.midline_x,
            reference_distance=alignment_result.reference_distance,
            valid_pair_count=0,
            total_pair_count=sum(len(p) for p in config.pairs_by_region.values()),
            is_valid=False,
            invalidity_reason=alignment_result.suitability_reason,
        )

    pts = alignment_result.aligned_points
    n_pts = len(pts)

    if n_pts < 10:
        return SymmetryResult(
            overall_normalized_asymmetry=float("nan"),
            symmetry_score=float("nan"),
            regional_asymmetry={},
            pair_details=[],
            midline_x=0.0,
            reference_distance=alignment_result.reference_distance,
            valid_pair_count=0,
            total_pair_count=sum(len(p) for p in config.pairs_by_region.values()),
            is_valid=False,
            invalidity_reason="Insufficient landmarks to calculate symmetry.",
        )

    # Step 2: Midline definition
    midline_x = estimate_facial_midline_x(pts, config.midline_indices)
    ref_dist = alignment_result.reference_distance

    pair_details: List[PairSymmetryDetail] = []
    regional_errors: Dict[str, List[float]] = {r: [] for r in config.pairs_by_region.keys()}
    total_pairs = 0

    # Step 3 & 4: Calculate pairwise reflection error
    for region_name, pairs in config.pairs_by_region.items():
        total_pairs += len(pairs)
        for r_idx, l_idx in pairs:
            if r_idx >= n_pts or l_idx >= n_pts:
                continue

            pt_r = pts[r_idx, :2]
            pt_l = pts[l_idx, :2]

            err_abs, err_norm, refl_l = calculate_pair_error(pt_r, pt_l, midline_x, ref_dist)

            detail = PairSymmetryDetail(
                region=region_name,
                right_index=r_idx,
                left_index=l_idx,
                right_pt=(float(pt_r[0]), float(pt_r[1])),
                left_pt=(float(pt_l[0]), float(pt_l[1])),
                reflected_left_pt=refl_l,
                absolute_error=err_abs,
                normalized_error=err_norm,
            )
            pair_details.append(detail)
            regional_errors[region_name].append(err_norm)

    valid_count = len(pair_details)
    if valid_count < 5:
        return SymmetryResult(
            overall_normalized_asymmetry=float("nan"),
            symmetry_score=float("nan"),
            regional_asymmetry={},
            pair_details=pair_details,
            midline_x=midline_x,
            reference_distance=ref_dist,
            valid_pair_count=valid_count,
            total_pair_count=total_pairs,
            is_valid=False,
            invalidity_reason="Insufficient valid paired landmarks (< 5) available for symmetry analysis.",
        )

    # Step 5: Regional aggregations
    regional_summary: Dict[str, float] = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for region_name, errs in regional_errors.items():
        if errs:
            mean_err = float(np.mean(errs))
            regional_summary[region_name] = mean_err
            w = config.regional_weights.get(region_name, 1.0)
            weighted_sum += mean_err * w
            weight_total += w
        else:
            regional_summary[region_name] = float("nan")

    overall_error = weighted_sum / weight_total if weight_total > 0 else float("nan")
    score = compute_heuristic_symmetry_score(overall_error, config.score_decay_factor)

    return SymmetryResult(
        overall_normalized_asymmetry=overall_error,
        symmetry_score=score,
        regional_asymmetry=regional_summary,
        pair_details=pair_details,
        midline_x=midline_x,
        reference_distance=ref_dist,
        valid_pair_count=valid_count,
        total_pair_count=total_pairs,
        is_valid=True,
        invalidity_reason=None,
    )
