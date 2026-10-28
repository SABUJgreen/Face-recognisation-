"""Landmark detection, coordinate extraction, and regional grouping using MediaPipe Face Landmarker."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarkerResult
import numpy as np

from src.config import (
    MIDLINE_LANDMARK_INDICES,
    SYMMETRY_PAIRS_BY_REGION,
    POSE_REFERENCE_3D_POINTS,
)
from src.image_processor import to_mediapipe_image
from src.model_loader import load_face_landmarker


@dataclass
class LandmarkPoint:
    """Represents a single 3D facial landmark in normalized and pixel coordinate systems."""
    index: int
    x_norm: float
    y_norm: float
    z_norm: float
    x_px: float
    y_px: float
    z_px: float  # z scaled roughly to pixel units via depth factor


@dataclass
class FaceLandmarks:
    """Structured representation of all detected landmarks for a single face."""
    points_norm: np.ndarray        # Shape (N, 3): [x_norm, y_norm, z_norm]
    points_px: np.ndarray          # Shape (N, 3): [x_px, y_px, z_px]
    image_width: int
    image_height: int
    num_landmarks: int
    transformation_matrix: Optional[np.ndarray] = None  # 4x4 matrix if available from MediaPipe

    @property
    def x_px(self) -> np.ndarray:
        return self.points_px[:, 0]

    @property
    def y_px(self) -> np.ndarray:
        return self.points_px[:, 1]

    @property
    def z_px(self) -> np.ndarray:
        return self.points_px[:, 2]

    def get_point(self, index: int) -> LandmarkPoint:
        """Return typed LandmarkPoint for the specified index."""
        if index < 0 or index >= self.num_landmarks:
            raise IndexError(f"Landmark index {index} out of range [0, {self.num_landmarks - 1}]")
        return LandmarkPoint(
            index=index,
            x_norm=float(self.points_norm[index, 0]),
            y_norm=float(self.points_norm[index, 1]),
            z_norm=float(self.points_norm[index, 2]),
            x_px=float(self.points_px[index, 0]),
            y_px=float(self.points_px[index, 1]),
            z_px=float(self.points_px[index, 2]),
        )

    def get_points_subset(self, indices: List[int], use_normalized: bool = False) -> np.ndarray:
        """Extract subset of points for the specified indices."""
        data = self.points_norm if use_normalized else self.points_px
        valid_indices = [idx for idx in indices if 0 <= idx < self.num_landmarks]
        if not valid_indices:
            return np.empty((0, 3), dtype=np.float32)
        return data[valid_indices]


def extract_landmark_coordinates(
    detection_result: FaceLandmarkerResult,
    image_width: int,
    image_height: int,
    face_idx: int = 0
) -> Optional[FaceLandmarks]:
    """Convert raw MediaPipe FaceLandmarkerResult into structured FaceLandmarks.

    Args:
        detection_result: MediaPipe detection result object.
        image_width: Width of source image in pixels.
        image_height: Height of source image in pixels.
        face_idx: Index of detected face to extract (default: 0 for primary face).

    Returns:
        FaceLandmarks instance if a face was detected, else None.
    """
    if (
        detection_result is None
        or not detection_result.face_landmarks
        or len(detection_result.face_landmarks) <= face_idx
    ):
        return None

    raw_landmarks = detection_result.face_landmarks[face_idx]
    n_pts = len(raw_landmarks)
    if n_pts == 0:
        return None

    points_norm = np.zeros((n_pts, 3), dtype=np.float32)
    points_px = np.zeros((n_pts, 3), dtype=np.float32)

    for i, lm in enumerate(raw_landmarks):
        # Normalized coordinates [0.0, 1.0]
        points_norm[i, 0] = lm.x
        points_norm[i, 1] = lm.y
        points_norm[i, 2] = lm.z

        # Pixel coordinates
        points_px[i, 0] = lm.x * image_width
        points_px[i, 1] = lm.y * image_height
        # In MediaPipe, z represents depth roughly with same scale as x
        points_px[i, 2] = lm.z * image_width

    trans_matrix = None
    if (
        detection_result.facial_transformation_matrixes
        and len(detection_result.facial_transformation_matrixes) > face_idx
    ):
        trans_matrix = np.array(
            detection_result.facial_transformation_matrixes[face_idx],
            dtype=np.float32
        )

    return FaceLandmarks(
        points_norm=points_norm,
        points_px=points_px,
        image_width=image_width,
        image_height=image_height,
        num_landmarks=n_pts,
        transformation_matrix=trans_matrix,
    )


def detect_face_landmarks(
    image: np.ndarray,
    detector=None,
    timestamp_ms: Optional[int] = None
) -> Optional[FaceLandmarks]:
    """Detect facial landmarks in an image using MediaPipe FaceLandmarker.

    Supports both static images and video frames (when timestamp_ms is provided).

    Args:
        image: OpenCV BGR image array.
        detector: Pre-loaded FaceLandmarker instance. If None, instantiates a default detector.
        timestamp_ms: Monotonically increasing timestamp in milliseconds for video mode.

    Returns:
        FaceLandmarks dataclass if a face is detected, otherwise None.
    """
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    mp_img = to_mediapipe_image(image)

    # Use provided detector or create standard image detector
    if detector is None:
        detector = load_face_landmarker()

    if timestamp_ms is not None:
        result = detector.detect_for_video(mp_img, timestamp_ms)
    else:
        result = detector.detect(mp_img)

    return extract_landmark_coordinates(result, image_width=w, image_height=h)


def select_landmark_regions(landmarks: FaceLandmarks) -> Dict[str, Dict[str, np.ndarray]]:
    """Organize landmark coordinates by anatomical regions and left/right symmetry subsets.

    Args:
        landmarks: Structured FaceLandmarks object.

    Returns:
        Dictionary mapping region name to a dict with 'right' and 'left' landmark arrays.
    """
    regions_data: Dict[str, Dict[str, np.ndarray]] = {}

    for region_name, pairs in SYMMETRY_PAIRS_BY_REGION.items():
        right_indices = [r for r, _ in pairs if r < landmarks.num_landmarks]
        left_indices = [l for _, l in pairs if l < landmarks.num_landmarks]

        regions_data[region_name] = {
            "right_px": landmarks.points_px[right_indices],
            "left_px": landmarks.points_px[left_indices],
            "right_indices": np.array(right_indices, dtype=int),
            "left_indices": np.array(left_indices, dtype=int),
        }

    midline_valid = [idx for idx in MIDLINE_LANDMARK_INDICES if idx < landmarks.num_landmarks]
    regions_data["midline"] = {
        "midline_px": landmarks.points_px[midline_valid],
        "midline_indices": np.array(midline_valid, dtype=int),
    }

    return regions_data
