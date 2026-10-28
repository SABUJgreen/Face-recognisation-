"""Visualization utilities for drawing landmarks, head pose orientation axes, and symmetry overlays."""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import MIDLINE_LANDMARK_INDICES, POSE_REFERENCE_3D_POINTS, REGIONAL_WEIGHTS
from src.face_alignment import AlignmentResult
from src.head_pose import HeadPoseResult
from src.landmarks import FaceLandmarks
from src.symmetry import SymmetryResult


# Color palettes (BGR format for OpenCV)
COLOR_GREEN = (46, 204, 113)     # Symmetric (< 0.03 normalized error)
COLOR_YELLOW = (41, 128, 185)    # Moderate (0.03 - 0.07)
COLOR_ORANGE = (39, 174, 96)
COLOR_RED = (41, 41, 231)        # Asymmetric (> 0.07)
COLOR_CYAN = (230, 200, 50)      # Facial midline
COLOR_BLUE = (255, 120, 30)      # Right landmarks
COLOR_MAGENTA = (200, 50, 230)   # Left landmarks
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_GRAY = (50, 50, 50)


def draw_landmarks(
    image: np.ndarray,
    landmarks: Union[FaceLandmarks, np.ndarray],
    radius: int = 1,
    color: Tuple[int, int, int] = (0, 255, 200),
    draw_iris: bool = True
) -> np.ndarray:
    """Draw discrete facial landmark points onto an image copy.

    Args:
        image: Source OpenCV BGR image.
        landmarks: FaceLandmarks object or array of shape (N, 2 or 3) in pixel coordinates.
        radius: Point radius in pixels.
        color: BGR tuple for points.
        draw_iris: Whether to highlight iris points (468-477) if present.

    Returns:
        Annotated copy of image.
    """
    canvas = image.copy()
    pts = landmarks.points_px if isinstance(landmarks, FaceLandmarks) else landmarks

    for i, pt in enumerate(pts):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
            if draw_iris and i >= 468:
                cv2.circle(canvas, (x, y), radius + 1, (0, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)

    return canvas


def draw_head_pose_axes(
    image: np.ndarray,
    head_pose: HeadPoseResult,
    axis_length: float = 60.0
) -> np.ndarray:
    """Project and draw 3D coordinate axes (Yaw/Pitch/Roll) protruding from the nose tip.

    Axis Colors (standard robotics convention):
    - Red (X axis, pitch): Points rightward.
    - Green (Y axis, yaw): Points downward.
    - Blue (Z axis, roll): Points outward from face toward camera.

    Args:
        image: OpenCV BGR image.
        head_pose: HeadPoseResult containing rvec, tvec, and camera parameters.
        axis_length: Length of visual axes in millimeters.

    Returns:
        Annotated image with 3D coordinate axes.
    """
    canvas = image.copy()

    if not head_pose.success or head_pose.rvec is None or head_pose.tvec is None:
        return canvas

    h, w = canvas.shape[:2]
    focal_length = float(w)
    camera_matrix = np.array([
        [focal_length, 0.0, w / 2.0],
        [0.0, focal_length, h / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # 3D points representing coordinate axes starting at nose tip (0, 0, 0)
    axis_3d = np.array([
        [0.0, 0.0, 0.0],                # Origin (nose tip)
        [axis_length, 0.0, 0.0],        # X axis (Pitch)
        [0.0, axis_length, 0.0],        # Y axis (Yaw)
        [0.0, 0.0, -axis_length]        # Z axis (Roll - pointing toward camera)
    ], dtype=np.float64)

    proj_pts, _ = cv2.projectPoints(
        axis_3d, head_pose.rvec, head_pose.tvec, camera_matrix, dist_coeffs
    )
    proj_pts = proj_pts.reshape(-1, 2).astype(int)

    p_origin = tuple(proj_pts[0])
    p_x = tuple(proj_pts[1])
    p_y = tuple(proj_pts[2])
    p_z = tuple(proj_pts[3])

    # Draw colored vector arrows
    cv2.arrowedLine(canvas, p_origin, p_x, (0, 0, 255), 3, tipLength=0.25, line_type=cv2.LINE_AA)  # Red = X
    cv2.arrowedLine(canvas, p_origin, p_y, (0, 255, 0), 3, tipLength=0.25, line_type=cv2.LINE_AA)  # Green = Y
    cv2.arrowedLine(canvas, p_origin, p_z, (255, 0, 0), 3, tipLength=0.25, line_type=cv2.LINE_AA)  # Blue = Z

    return canvas


def draw_symmetry_overlay(
    image: np.ndarray,
    landmarks: Union[FaceLandmarks, np.ndarray],
    symmetry_result: SymmetryResult,
    alignment_result: Optional[AlignmentResult] = None,
    show_midline: bool = True,
    show_pairs: bool = True,
    show_vectors: bool = True
) -> np.ndarray:
    """Draw comprehensive symmetry overlay on original image coordinates.

    Visualizes:
    - Facial centerline along sagittal plane.
    - Paired left and right anatomical landmarks.
    - Connecting bridge lines with color-coded asymmetry magnitude.
    - Measurement HUD status banner.

    Args:
        image: Source OpenCV BGR image.
        landmarks: Detected face landmarks in image pixel coordinates.
        symmetry_result: Computed SymmetryResult.
        alignment_result: Optional AlignmentResult for rotation/scale context.
        show_midline: Whether to draw central facial axis.
        show_pairs: Whether to draw paired landmark points.
        show_vectors: Whether to draw connecting error vectors between pairs.

    Returns:
        Annotated image copy.
    """
    canvas = image.copy()
    pts = landmarks.points_px if isinstance(landmarks, FaceLandmarks) else landmarks
    n_pts = len(pts)

    # 1. Draw facial centerline
    if show_midline:
        midline_pts = [
            (int(round(pts[idx, 0])), int(round(pts[idx, 1])))
            for idx in MIDLINE_LANDMARK_INDICES
            if idx < n_pts
        ]
        if len(midline_pts) >= 2:
            # Polyline along the nasal/facial ridge
            for i in range(len(midline_pts) - 1):
                cv2.line(canvas, midline_pts[i], midline_pts[i + 1], COLOR_CYAN, 2, cv2.LINE_AA)
            for p in midline_pts:
                cv2.circle(canvas, p, 3, COLOR_CYAN, -1, cv2.LINE_AA)

    # 2. Draw bilateral landmark pairs and discrepancy lines
    if show_pairs and symmetry_result.is_valid:
        # Build map of detail by pair index
        for detail in symmetry_result.pair_details:
            r_idx, l_idx = detail.right_index, detail.left_index
            if r_idx >= n_pts or l_idx >= n_pts:
                continue

            pt_r = (int(round(pts[r_idx, 0])), int(round(pts[r_idx, 1])))
            pt_l = (int(round(pts[l_idx, 0])), int(round(pts[l_idx, 1])))

            # Determine color from normalized asymmetry
            err_norm = detail.normalized_error
            if err_norm < 0.03:
                line_color = COLOR_GREEN
            elif err_norm < 0.06:
                line_color = COLOR_YELLOW
            elif err_norm < 0.10:
                line_color = COLOR_ORANGE
            else:
                line_color = COLOR_RED

            if show_vectors:
                # Discrepancy indicator line connecting bilateral landmarks
                cv2.line(canvas, pt_r, pt_l, line_color, 1, cv2.LINE_AA)

            # Draw right (blue) and left (magenta) points
            cv2.circle(canvas, pt_r, 3, COLOR_BLUE, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt_l, 3, COLOR_MAGENTA, -1, cv2.LINE_AA)

    return canvas


def create_regional_asymmetry_bar_chart(
    regional_asymmetry: Dict[str, float],
    overall_asymmetry: float
) -> plt.Figure:
    """Generate a clean Matplotlib bar chart showing regional asymmetry compared with overall index.

    Args:
        regional_asymmetry: Dictionary mapping region names to normalized asymmetry errors.
        overall_asymmetry: Overall weighted normalized error.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=100)
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#161B22")

    regions = list(regional_asymmetry.keys())
    values = [regional_asymmetry.get(r, 0.0) * 100.0 for r in regions]  # in % of inter-ocular dist

    # Color code bars based on asymmetry severity
    bar_colors = []
    for v in values:
        if v < 3.0:
            bar_colors.append("#2ECC71")   # Green (Good balance)
        elif v < 6.0:
            bar_colors.append("#3498DB")   # Blue (Moderate)
        elif v < 10.0:
            bar_colors.append("#F39C12")   # Orange
        else:
            bar_colors.append("#E74C3C")   # Red

    y_pos = np.arange(len(regions))
    bars = ax.barh(y_pos, values, color=bar_colors, height=0.55, edgecolor="none")

    # Add reference overall error vertical line
    if not np.isnan(overall_asymmetry):
        overall_pct = overall_asymmetry * 100.0
        ax.axvline(
            overall_pct,
            color="#E74C3C",
            linestyle="--",
            alpha=0.8,
            label=f"Overall Index: {overall_pct:.1f}%"
        )
        ax.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="#E6EDF3", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r.capitalize() for r in regions], color="#E6EDF3", fontsize=10)
    ax.set_xlabel("Normalized Asymmetry (% of Inter-Ocular Distance)", color="#8B949E", fontsize=9)
    ax.set_title("Regional Geometric Facial Asymmetry", color="#FFFFFF", fontsize=12, pad=10, weight="bold")

    ax.tick_params(colors="#8B949E", which="both")
    ax.grid(axis="x", color="#30363D", linestyle=":", alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color("#30363D")

    # Value labels on bars
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.3,
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:.1f}%",
            va="center",
            ha="left",
            color="#E6EDF3",
            fontsize=9
        )

    max_v = max(values + [5.0])
    ax.set_xlim(0, max_v * 1.25)
    plt.tight_layout()
    return fig
