"""Model loading and verification module for MediaPipe Face Landmarker."""

import os
from pathlib import Path
from typing import Optional, Union
import urllib.error
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import DEFAULT_MODEL_PATH, OFFICIAL_MODEL_URL


class ModelLoadError(RuntimeError):
    """Raised when the Face Landmarker model asset fails to load or download."""
    pass


def download_model_if_missing(
    model_path: Optional[Union[str, Path]] = None,
    url: str = OFFICIAL_MODEL_URL,
    force_download: bool = False
) -> Path:
    """Ensure the MediaPipe face_landmarker.task model file exists locally.

    Args:
        model_path: Destination path for the model file. Defaults to DEFAULT_MODEL_PATH.
        url: Official Google MediaPipe download URL.
        force_download: If True, re-downloads even if file already exists.

    Returns:
        Path to the verified model asset file.

    Raises:
        ModelLoadError: If downloading fails due to network or write permission errors.
    """
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if target_path.exists() and not force_download:
        if target_path.stat().st_size > 100_000:  # Verify non-empty bundle
            return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, str(target_path))
        file_size = target_path.stat().st_size
        if file_size < 100_000:
            target_path.unlink(missing_ok=True)
            raise ModelLoadError(
                f"Downloaded model bundle from {url} is suspiciously small ({file_size} bytes)."
            )
        return target_path
    except (urllib.error.URLError, OSError) as exc:
        raise ModelLoadError(
            f"Failed to download MediaPipe face landmarker model from {url} to {target_path}: {exc}\n"
            f"Please manually download the official model and place it at {target_path}."
        ) from exc


def load_face_landmarker(
    model_path: Optional[Union[str, Path]] = None,
    running_mode: vision.RunningMode = vision.RunningMode.IMAGE,
    num_faces: int = 1,
    min_face_detection_confidence: float = 0.5,
    min_face_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    output_facial_transformation_matrixes: bool = True,
    auto_download: bool = True,
    result_callback=None
) -> vision.FaceLandmarker:
    """Create and return an initialized MediaPipe FaceLandmarker instance.

    Args:
        model_path: Path to face_landmarker.task. If None, uses DEFAULT_MODEL_PATH.
        running_mode: MediaPipe RunningMode (IMAGE, VIDEO, or LIVE_STREAM).
        num_faces: Maximum number of faces to detect (default: 1).
        min_face_detection_confidence: Confidence threshold for face detection.
        min_face_presence_confidence: Confidence threshold for face presence.
        min_tracking_confidence: Confidence threshold for landmark tracking in video.
        output_facial_transformation_matrixes: Whether to compute 4x4 face transformation matrices.
        auto_download: If True, attempts to download model asset if missing locally.
        result_callback: Callback function required when running in LIVE_STREAM mode.

    Returns:
        Initialized vision.FaceLandmarker detector.

    Raises:
        ModelLoadError: If the model file is missing and cannot be loaded.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not path.exists():
        if not auto_download:
            raise ModelLoadError(f"Model file '{path}' does not exist and auto_download is disabled.")
        try:
            path = download_model_if_missing(path)
        except Exception as exc:
            raise ModelLoadError(
                f"Model file '{path}' not found and automatic download failed: {exc}"
            ) from exc

    try:
        base_options = python.BaseOptions(model_asset_path=str(path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=output_facial_transformation_matrixes,
            result_callback=result_callback,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        return detector
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to initialize MediaPipe FaceLandmarker with model '{path}': {exc}"
        ) from exc
