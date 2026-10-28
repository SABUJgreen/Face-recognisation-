"""Facial alignment module performing 2D similarity transform and pose suitability checks."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

from src.config import (
    EYE_CENTER_LEFT_INDICES,
    EYE_CENTER_RIGHT_INDICES,
    MAX_FRONTAL_PITCH_DEG,
    MAX_FRONTAL_ROLL_DEG,
    MAX_FRONTAL_YAW_DEG,
    MIDLINE_LANDMARK_INDICES,
)
from src.head_pose import HeadPoseResult
from src.landmarks import FaceLandmarks


@dataclass
class AlignmentResult:
    """Structured result of facial landmark alignment."""
    aligned_points: np.ndarray      # Shape (N, 3) aligned coordinates
    rotation_angle_deg: float       # In-plane rotation angle (degrees) removed during alignment
    scale_factor: float             # Uniform scale factor applied
    translation: Tuple[float, float]# (tx, ty) shift applied before rotation
    midline_x: float                # Target midline X coordinate in aligned space (0.0)
    reference_distance: float       # Inter-ocular distance in aligned coordinates
    is_suitable_for_symmetry: bool  # True if frontal and quality checks pass
    suitability_reason: str         # Explanation of suitability or rejection reasons
    raw_reference_distance: float   # Inter-ocular distance prior to alignment (pixels)


def calculate_eye_centers(
    landmarks_px: np.ndarray,
    right_indices: List[int] = EYE_CENTER_RIGHT_INDICES,
    left_indices: List[int] = EYE_CENTER_LEFT_INDICES
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Calculate the centroid of subject's right and left eye regions and their inter-ocular distance.

    Args:
        landmarks_px: Array of shape (N, 2 or 3) containing landmark coordinates.
        right_indices: Indices for subject's right eye (viewer's left).
        left_indices: Indices for subject's left eye (viewer's right).

    Returns:
        Tuple of (right_center_xy, left_center_xy, inter_ocular_distance).
    """
    n_pts = len(landmarks_px)
    valid_r = [idx for idx in right_indices if idx < n_pts]
    valid_l = [idx for idx in left_indices if idx < n_pts]

    if not valid_r or not valid_l:
        raise ValueError("Insufficient eye landmarks available to compute eye centers.")

    right_center = np.mean(landmarks_px[valid_r, :2], axis=0)
    left_center = np.mean(landmarks_px[valid_l, :2], axis=0)

    inter_ocular_dist = float(np.linalg.norm(left_center - right_center))
    return right_center, left_center, inter_ocular_dist


def align_face(
    landmarks: Union[FaceLandmarks, np.ndarray],
    head_pose: Optional[HeadPoseResult] = None,
    method: str = "similarity",
    target_interocular_distance: float = 100.0
) -> AlignmentResult:
    """Normalize facial translation, scale, and in-plane rotation to establish a canonical frontal frame.

    Alignment Procedure:
    1. Validate pose: Ensure out-of-plane yaw and pitch do not violate frontal thresholds.
       If out-of-plane rotation exceeds thresholds, flag as unsuitable because 2D
       transformations cannot undo 3D perspective distortion (foreshortening).
    2. Compute Eye Centers: Determine centroids of left and right eyes.
    3. Translation: Shift the inter-ocular midpoint (or nasion) to the origin (0, 0).
    4. In-Plane Rotation: Calculate the angle of the eye axis relative to the horizontal.
       Rotate points by -theta so the eye axis becomes strictly horizontal.
    5. Scale Normalization: Scale uniformly so the inter-ocular distance equals target_interocular_distance.
    6. Vertical Midline Alignment: In this canonical frame, the vertical symmetry axis is x = 0.

    Args:
        landmarks: FaceLandmarks object or NumPy array of shape (N, 2 or 3).
        head_pose: Optional HeadPoseResult for 3D orientation gating.
        method: Alignment method ("similarity" supported).
        target_interocular_distance: Desired distance between eye centers in aligned coordinate space.

    Returns:
        AlignmentResult dataclass.
    """
    if isinstance(landmarks, FaceLandmarks):
        points_px = landmarks.points_px.copy()
    else:
        points_px = np.array(landmarks, dtype=np.float32).copy()

    if len(points_px) < 10:
        return AlignmentResult(
            aligned_points=points_px,
            rotation_angle_deg=0.0,
            scale_factor=1.0,
            translation=(0.0, 0.0),
            midline_x=0.0,
            reference_distance=1.0,
            is_suitable_for_symmetry=False,
            suitability_reason="Insufficient landmarks (< 10) detected.",
            raw_reference_distance=0.0,
        )

    # 1. Check pose suitability
    is_suitable = True
    reasons = []

    if head_pose is not None:
        if not head_pose.success:
            is_suitable = False
            reasons.append("Head pose estimation failed or diverged.")
        else:
            if abs(head_pose.yaw_degrees) > MAX_FRONTAL_YAW_DEG:
                is_suitable = False
                reasons.append(
                    f"Excessive yaw ({head_pose.yaw_degrees:.1f}° > {MAX_FRONTAL_YAW_DEG}°). Face turned sideways."
                )
            if abs(head_pose.pitch_degrees) > MAX_FRONTAL_PITCH_DEG:
                is_suitable = False
                reasons.append(
                    f"Excessive pitch ({head_pose.pitch_degrees:.1f}° > {MAX_FRONTAL_PITCH_DEG}°). Face tilted up/down."
                )
            if abs(head_pose.roll_degrees) > 45.0:
                is_suitable = False
                reasons.append(
                    f"Extreme roll angle ({head_pose.roll_degrees:.1f}°). Excessive head tilt."
                )

    # 2. Compute eye centers
    try:
        right_eye_center, left_eye_center, raw_iod = calculate_eye_centers(points_px)
    except ValueError as exc:
        return AlignmentResult(
            aligned_points=points_px,
            rotation_angle_deg=0.0,
            scale_factor=1.0,
            translation=(0.0, 0.0),
            midline_x=0.0,
            reference_distance=1.0,
            is_suitable_for_symmetry=False,
            suitability_reason=f"Eye landmark extraction failed: {exc}",
            raw_reference_distance=0.0,
        )

    if raw_iod < 5.0:
        return AlignmentResult(
            aligned_points=points_px,
            rotation_angle_deg=0.0,
            scale_factor=1.0,
            translation=(0.0, 0.0),
            midline_x=0.0,
            reference_distance=1.0,
            is_suitable_for_symmetry=False,
            suitability_reason=f"Inter-ocular distance is too small ({raw_iod:.1f} px) for reliable geometric analysis.",
            raw_reference_distance=raw_iod,
        )

    # Midpoint between eyes serves as canonical origin
    eye_midpoint = (left_eye_center + right_eye_center) / 2.0

    # 3. Calculate in-plane roll angle (dx, dy from right eye to left eye)
    # Right eye is at lower x (image left), Left eye is at higher x (image right)
    dx = left_eye_center[0] - right_eye_center[0]
    dy = left_eye_center[1] - right_eye_center[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = float(np.degrees(angle_rad))

    # 4. Scale factor to normalize inter-ocular distance
    scale = target_interocular_distance / raw_iod

    # 5. Apply 2D Similarity Transform: Translation -> Rotation -> Scaling
    # Rotate by -angle_rad so eye axis aligns with the horizontal axis
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    R_2d = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ], dtype=np.float32)

    # Copy 3D array or create if 2D
    aligned_pts = np.zeros_like(points_px)
    if points_px.shape[1] >= 3:
        aligned_pts[:, 2] = points_px[:, 2] * scale

    # Shift by eye midpoint
    shifted_xy = points_px[:, :2] - eye_midpoint
    # Rotate
    rotated_xy = np.dot(shifted_xy, R_2d.T)
    # Scale
    aligned_xy = rotated_xy * scale

    aligned_pts[:, :2] = aligned_xy

    # Validate midline alignment: In this aligned coordinate frame,
    # the eye midpoint is strictly at (0, 0), and midline x = 0.
    midline_x = 0.0

    reason_str = "Suitable for frontal symmetry analysis." if is_suitable else "; ".join(reasons)

    return AlignmentResult(
        aligned_points=aligned_pts,
        rotation_angle_deg=angle_deg,
        scale_factor=scale,
        translation=(float(eye_midpoint[0]), float(eye_midpoint[1])),
        midline_x=midline_x,
        reference_distance=target_interocular_distance,
        is_suitable_for_symmetry=is_suitable,
        suitability_reason=reason_str,
        raw_reference_distance=raw_iod,
    )
