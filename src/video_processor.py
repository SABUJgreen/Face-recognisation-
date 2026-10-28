"""Video and webcam frame processing pipeline with temporal smoothing and metrics aggregation."""

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_MAX_VIDEO_DURATION_SEC,
    DEFAULT_MAX_VIDEO_FRAMES,
    DEFAULT_SMOOTHING_ALPHA,
    OUTPUTS_DIR,
)
from src.face_alignment import align_face
from src.head_pose import estimate_head_pose
from src.landmarks import extract_landmark_coordinates
from src.model_loader import load_face_landmarker
from src.symmetry import calculate_symmetry
from src.visualization import draw_head_pose_axes, draw_landmarks, draw_symmetry_overlay


@dataclass
class VideoFrameMetric:
    """Metrics recorded for a single processed video frame."""
    frame_index: int
    timestamp_ms: int
    face_detected: bool
    yaw_deg: Optional[float]
    pitch_deg: Optional[float]
    roll_deg: Optional[float]
    yaw_smoothed: Optional[float]
    pitch_smoothed: Optional[float]
    roll_smoothed: Optional[float]
    reprojection_error: Optional[float]
    is_frontal: bool
    is_symmetry_valid: bool
    overall_asymmetry: Optional[float]
    symmetry_score: Optional[float]
    eye_asymmetry: Optional[float]
    eyebrow_asymmetry: Optional[float]
    cheek_asymmetry: Optional[float]
    mouth_asymmetry: Optional[float]
    jaw_asymmetry: Optional[float]
    valid_pair_count: int
    status_note: str


@dataclass
class VideoProcessingSummary:
    """Summary statistics for a processed video sequence."""
    total_video_frames: int
    processed_frames: int
    face_detected_frames: int
    frontal_frames: int
    valid_symmetry_frames: int
    face_detection_rate: float
    symmetry_validity_rate: float
    mean_yaw_deg: Optional[float]
    mean_pitch_deg: Optional[float]
    mean_roll_deg: Optional[float]
    mean_symmetry_score: Optional[float]
    mean_overall_asymmetry: Optional[float]
    output_video_path: Optional[str]
    metrics_df: pd.DataFrame


class TemporalPoseSmoother:
    """Exponential Moving Average (EMA) smoother for head pose Euler angles."""

    def __init__(self, alpha: float = DEFAULT_SMOOTHING_ALPHA):
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.smoothed_yaw: Optional[float] = None
        self.smoothed_pitch: Optional[float] = None
        self.smoothed_roll: Optional[float] = None

    def update(self, yaw: float, pitch: float, roll: float) -> Tuple[float, float, float]:
        """Update filter state and return smoothed angles."""
        if self.smoothed_yaw is None:
            self.smoothed_yaw = yaw
            self.smoothed_pitch = pitch
            self.smoothed_roll = roll
        else:
            self.smoothed_yaw = self.alpha * yaw + (1.0 - self.alpha) * self.smoothed_yaw
            self.smoothed_pitch = self.alpha * pitch + (1.0 - self.alpha) * self.smoothed_pitch
            self.smoothed_roll = self.alpha * roll + (1.0 - self.alpha) * self.smoothed_roll

        return self.smoothed_yaw, self.smoothed_pitch, self.smoothed_roll

    def reset(self) -> None:
        """Reset internal filter state when face is lost."""
        self.smoothed_yaw = None
        self.smoothed_pitch = None
        self.smoothed_roll = None


def process_video(
    video_source: Union[str, Path],
    max_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    save_annotated_video: bool = True,
    output_filename: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA
) -> VideoProcessingSummary:
    """Process a video file, estimating head posture and facial symmetry per frame.

    Args:
        video_source: Path to video file.
        max_frames: Maximum number of frames to process.
        frame_stride: Process every N-th frame (default: 1 for every frame).
        save_annotated_video: Whether to write an annotated MP4 video output.
        output_filename: Optional specific output filename.
        progress_callback: Optional callback receiving (progress_ratio, status_text).
        smoothing_alpha: EMA smoothing weight for head angles.

    Returns:
        VideoProcessingSummary containing dataframes, metrics, and video path.
    """
    video_path = Path(video_source)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video-mode MediaPipe FaceLandmarker
    detector = load_face_landmarker(running_mode=vision.RunningMode.VIDEO)

    smoother = TemporalPoseSmoother(alpha=smoothing_alpha)
    records: List[VideoFrameMetric] = []

    # Configure output video writer if requested
    writer = None
    output_video_filepath = None
    if save_annotated_video:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        fname = output_filename or f"processed_{video_path.stem}_{int(time.time())}.mp4"
        output_video_filepath = str(OUTPUTS_DIR / fname)
        # Use mp4v or avc1
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_filepath, fourcc, fps / max(1, frame_stride), (width, height))

    frame_idx = 0
    processed_count = 0
    monotonic_timestamp_ms = 0

    try:
        while cap.isOpened() and processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Stride check
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            # Ensure strictly monotonically increasing timestamp for MediaPipe
            calc_timestamp_ms = int((frame_idx / fps) * 1000.0)
            if calc_timestamp_ms <= monotonic_timestamp_ms:
                monotonic_timestamp_ms += 1
            else:
                monotonic_timestamp_ms = calc_timestamp_ms

            # Convert to MediaPipe image format
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Perform detection in VIDEO mode
            result = detector.detect_for_video(mp_image, monotonic_timestamp_ms)
            landmarks = extract_landmark_coordinates(result, image_width=width, image_height=height)

            annotated_frame = frame.copy()

            if landmarks is None:
                smoother.reset()
                metric = VideoFrameMetric(
                    frame_index=frame_idx,
                    timestamp_ms=monotonic_timestamp_ms,
                    face_detected=False,
                    yaw_deg=None,
                    pitch_deg=None,
                    roll_deg=None,
                    yaw_smoothed=None,
                    pitch_smoothed=None,
                    roll_smoothed=None,
                    reprojection_error=None,
                    is_frontal=False,
                    is_symmetry_valid=False,
                    overall_asymmetry=None,
                    symmetry_score=None,
                    eye_asymmetry=None,
                    eyebrow_asymmetry=None,
                    cheek_asymmetry=None,
                    mouth_asymmetry=None,
                    jaw_asymmetry=None,
                    valid_pair_count=0,
                    status_note="No face detected",
                )
            else:
                # 1. Pose estimation
                head_pose = estimate_head_pose(landmarks, image_width=width, image_height=height)
                if head_pose.success:
                    y_sm, p_sm, r_sm = smoother.update(
                        head_pose.yaw_degrees, head_pose.pitch_degrees, head_pose.roll_degrees
                    )
                else:
                    smoother.reset()
                    y_sm, p_sm, r_sm = None, None, None

                # 2. Alignment & Symmetry
                alignment = align_face(landmarks, head_pose=head_pose)
                symmetry = calculate_symmetry(landmarks, alignment_result=alignment, head_pose=head_pose)

                # Annotate frame
                annotated_frame = draw_landmarks(annotated_frame, landmarks, radius=1)
                if head_pose.success:
                    annotated_frame = draw_head_pose_axes(annotated_frame, head_pose)
                if symmetry.is_valid:
                    annotated_frame = draw_symmetry_overlay(
                        annotated_frame, landmarks, symmetry, alignment_result=alignment
                    )

                status = "Valid" if symmetry.is_valid else (symmetry.invalidity_reason or "Unsuitable")

                metric = VideoFrameMetric(
                    frame_index=frame_idx,
                    timestamp_ms=monotonic_timestamp_ms,
                    face_detected=True,
                    yaw_deg=head_pose.yaw_degrees if head_pose.success else None,
                    pitch_deg=head_pose.pitch_degrees if head_pose.success else None,
                    roll_deg=head_pose.roll_degrees if head_pose.success else None,
                    yaw_smoothed=y_sm,
                    pitch_smoothed=p_sm,
                    roll_smoothed=r_sm,
                    reprojection_error=head_pose.reprojection_error if head_pose.success else None,
                    is_frontal=head_pose.is_frontal if head_pose.success else False,
                    is_symmetry_valid=symmetry.is_valid,
                    overall_asymmetry=symmetry.overall_normalized_asymmetry if symmetry.is_valid else None,
                    symmetry_score=symmetry.symmetry_score if symmetry.is_valid else None,
                    eye_asymmetry=symmetry.regional_asymmetry.get("eyes"),
                    eyebrow_asymmetry=symmetry.regional_asymmetry.get("eyebrows"),
                    cheek_asymmetry=symmetry.regional_asymmetry.get("cheeks"),
                    mouth_asymmetry=symmetry.regional_asymmetry.get("mouth"),
                    jaw_asymmetry=symmetry.regional_asymmetry.get("jawline"),
                    valid_pair_count=symmetry.valid_pair_count,
                    status_note=status,
                )

            records.append(metric)

            if writer is not None:
                writer.write(annotated_frame)

            frame_idx += 1
            processed_count += 1

            if progress_callback is not None:
                denom = min(total_cap_frames, max_frames) if total_cap_frames > 0 else max_frames
                prog = min(1.0, processed_count / max(1, denom))
                progress_callback(prog, f"Processing frame {processed_count}/{denom}...")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        detector.close()

    # Convert records to DataFrame
    df = pd.DataFrame([m.__dict__ for m in records])

    # Compute summary statistics
    n_processed = len(df)
    n_detected = int(df["face_detected"].sum()) if n_processed > 0 else 0
    n_frontal = int(df["is_frontal"].sum()) if n_processed > 0 else 0
    n_valid_sym = int(df["is_symmetry_valid"].sum()) if n_processed > 0 else 0

    det_rate = (n_detected / n_processed * 100.0) if n_processed > 0 else 0.0
    sym_rate = (n_valid_sym / n_processed * 100.0) if n_processed > 0 else 0.0

    valid_sym_df = df[df["is_symmetry_valid"] == True]
    mean_score = float(valid_sym_df["symmetry_score"].mean()) if not valid_sym_df.empty else None
    mean_overall = float(valid_sym_df["overall_asymmetry"].mean()) if not valid_sym_df.empty else None

    pose_df = df[df["face_detected"] == True]
    mean_yaw = float(pose_df["yaw_deg"].mean()) if not pose_df.empty else None
    mean_pitch = float(pose_df["pitch_deg"].mean()) if not pose_df.empty else None
    mean_roll = float(pose_df["roll_deg"].mean()) if not pose_df.empty else None

    return VideoProcessingSummary(
        total_video_frames=total_cap_frames,
        processed_frames=n_processed,
        face_detected_frames=n_detected,
        frontal_frames=n_frontal,
        valid_symmetry_frames=n_valid_sym,
        face_detection_rate=det_rate,
        symmetry_validity_rate=sym_rate,
        mean_yaw_deg=mean_yaw,
        mean_pitch_deg=mean_pitch,
        mean_roll_deg=mean_roll,
        mean_symmetry_score=mean_score,
        mean_overall_asymmetry=mean_overall,
        output_video_path=output_video_filepath,
        metrics_df=df,
    )
