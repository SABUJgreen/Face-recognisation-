"""Head pose estimation using OpenCV solvePnP and MediaPipe transformation matrices."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np

from src.config import (
    HeadPoseConfig,
    MAX_FRONTAL_PITCH_DEG,
    MAX_FRONTAL_ROLL_DEG,
    MAX_FRONTAL_YAW_DEG,
    POSE_REFERENCE_3D_POINTS,
)
from src.landmarks import FaceLandmarks


@dataclass
class HeadPoseResult:
    """Structured result of 3D head pose estimation."""
    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float
    rotation_matrix: np.ndarray      # Shape (3, 3)
    translation_vector: np.ndarray   # Shape (3, 1)
    success: bool
    reprojection_error: float        # RMSE in pixels between observed and reprojected 3D points
    is_frontal: bool
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    mp_euler_angles: Optional[Tuple[float, float, float]] = None  # (yaw, pitch, roll) from MediaPipe matrix if available
    error_message: Optional[str] = None


def get_default_camera_matrix(image_width: int, image_height: int) -> np.ndarray:
    """Construct an approximate pinhole camera intrinsic matrix.

    Assumes a standard pinhole perspective model:
    - Focal length fx = fy = image_width (approx 60-70 degree field of view typical of consumer webcams)
    - Principal point at image center (cx = width / 2, cy = height / 2)
    - Zero skew and zero radial/tangential distortion

    Note: These uncalibrated intrinsics yield approximate pose angles suitable
    for posture estimation, not metrological ground truth.
    """
    focal_length = float(image_width)
    center_x = float(image_width) / 2.0
    center_y = float(image_height) / 2.0

    camera_matrix = np.array([
        [focal_length, 0.0, center_x],
        [0.0, focal_length, center_y],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return camera_matrix


def rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to Euler angles (yaw, pitch, roll) in degrees.

    Uses cv2.RQDecomp3x3 decomposition adhering to standard aerospace / computer vision convention:
    - Camera frame: +X right, +Y down, +Z into the scene.
    - Pitch (degrees): Rotation around X axis (nodding up/down; positive = looking down).
    - Yaw (degrees): Rotation around Y axis (turning left/right; positive = turned to subject's left).
    - Roll (degrees): Rotation around Z axis (tilting head sideways; positive = tilt to subject's right).

    Returns:
        Tuple of (yaw_degrees, pitch_degrees, roll_degrees).
    """
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)
    # angles returns (pitch, yaw, roll) in degrees
    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    return yaw, pitch, roll


def compute_reprojection_error(
    model_points_3d: np.ndarray,
    observed_points_2d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
) -> float:
    """Calculate the Root Mean Square Error (RMSE) between observed landmarks and reprojected 3D points.

    Returns:
        float: RMSE reprojection error in pixels.
    """
    projected_points_2d, _ = cv2.projectPoints(
        model_points_3d, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected_points_2d = projected_points_2d.reshape(-1, 2)
    observed_points_2d = observed_points_2d.reshape(-1, 2)

    diff = projected_points_2d - observed_points_2d
    rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    return rmse


def extract_euler_from_transformation_matrix(
    transformation_matrix_4x4: np.ndarray
) -> Optional[Tuple[float, float, float]]:
    """Extract Euler angles (yaw, pitch, roll in degrees) from MediaPipe's 4x4 facial transformation matrix.

    Args:
        transformation_matrix_4x4: 4x4 homogenous matrix from MediaPipe.

    Returns:
        Tuple of (yaw, pitch, roll) in degrees, or None if invalid.
    """
    if transformation_matrix_4x4 is None or transformation_matrix_4x4.shape != (4, 4):
        return None

    R = transformation_matrix_4x4[:3, :3]
    try:
        # Standard decomposition for MediaPipe 4x4 pose
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            yaw = np.degrees(np.arctan2(-R[2, 0], sy))
            roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
            yaw = np.degrees(np.arctan2(-R[2, 0], sy))
            roll = 0.0
        return float(yaw), float(pitch), float(roll)
    except Exception:
        return None


def estimate_head_pose(
    landmarks: Union[FaceLandmarks, np.ndarray],
    image_width: int,
    image_height: int,
    camera_matrix: Optional[np.ndarray] = None,
    config: Optional[HeadPoseConfig] = None
) -> HeadPoseResult:
    """Estimate head orientation (yaw, pitch, roll) using OpenCV's solvePnP.

    Args:
        landmarks: FaceLandmarks object or NumPy array of shape (N, 2 or 3) in pixel coordinates.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        camera_matrix: Optional 3x3 camera intrinsic matrix. If None, uses pinhole approximation.
        config: Optional HeadPoseConfig.

    Returns:
        HeadPoseResult dataclass.
    """
    if config is None:
        config = HeadPoseConfig()

    points_3d_dict = config.points_3d
    model_pts_list = []
    image_pts_list = []

    # Extract 2D and 3D correspondences
    is_facelandmarks_obj = isinstance(landmarks, FaceLandmarks)
    for idx, pt_3d in points_3d_dict.items():
        if is_facelandmarks_obj:
            if idx < landmarks.num_landmarks:
                model_pts_list.append(pt_3d)
                image_pts_list.append([landmarks.points_px[idx, 0], landmarks.points_px[idx, 1]])
        else:
            if idx < len(landmarks):
                model_pts_list.append(pt_3d)
                image_pts_list.append([landmarks[idx, 0], landmarks[idx, 1]])

    if len(model_pts_list) < 4:
        return HeadPoseResult(
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            roll_degrees=0.0,
            rotation_matrix=np.eye(3, dtype=np.float64),
            translation_vector=np.zeros((3, 1), dtype=np.float64),
            success=False,
            reprojection_error=float("inf"),
            is_frontal=False,
            error_message="Insufficient corresponding landmark points for solvePnP (minimum 4 required)."
        )

    model_points_3d = np.array(model_pts_list, dtype=np.float64)
    image_points_2d = np.array(image_pts_list, dtype=np.float64)

    if camera_matrix is None:
        camera_matrix = get_default_camera_matrix(image_width, image_height)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    try:
        # solvePnP with iterative Levenberg-Marquardt optimization
        success, rvec, tvec = cv2.solvePnP(
            model_points_3d,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return HeadPoseResult(
                yaw_degrees=0.0,
                pitch_degrees=0.0,
                roll_degrees=0.0,
                rotation_matrix=np.eye(3, dtype=np.float64),
                translation_vector=np.zeros((3, 1), dtype=np.float64),
                success=False,
                reprojection_error=float("inf"),
                is_frontal=False,
                error_message="cv2.solvePnP failed to converge."
            )

        R, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = rotation_matrix_to_euler_angles(R)

        reprojection_error = compute_reprojection_error(
            model_points_3d, image_points_2d, rvec, tvec, camera_matrix, dist_coeffs
        )

        # Check frontal criteria
        is_frontal = (
            abs(yaw) <= config.max_frontal_yaw
            and abs(pitch) <= config.max_frontal_pitch
            and abs(roll) <= config.max_frontal_roll
        )

        mp_angles = None
        if is_facelandmarks_obj and landmarks.transformation_matrix is not None:
            mp_angles = extract_euler_from_transformation_matrix(landmarks.transformation_matrix)

        return HeadPoseResult(
            yaw_degrees=yaw,
            pitch_degrees=pitch,
            roll_degrees=roll,
            rotation_matrix=R,
            translation_vector=tvec,
            success=True,
            reprojection_error=reprojection_error,
            is_frontal=is_frontal,
            rvec=rvec,
            tvec=tvec,
            mp_euler_angles=mp_angles,
        )

    except Exception as exc:
        return HeadPoseResult(
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            roll_degrees=0.0,
            rotation_matrix=np.eye(3, dtype=np.float64),
            translation_vector=np.zeros((3, 1), dtype=np.float64),
            success=False,
            reprojection_error=float("inf"),
            is_frontal=False,
            error_message=f"solvePnP exception: {exc}"
        )
