"""Integration tests validating end-to-end pipelines on real images and mock video."""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from src.config import DATA_DIR
from src.export_utils import create_image_analysis_summary, export_to_csv, export_to_json
from src.face_alignment import align_face
from src.head_pose import estimate_head_pose
from src.image_processor import ImageProcessingError, load_image
from src.landmarks import detect_face_landmarks
from src.model_loader import ModelLoadError, load_face_landmarker
from src.symmetry import calculate_symmetry
from src.video_processor import process_video


def test_model_loader_missing_file_error():
    """Verify load_face_landmarker raises ModelLoadError if an invalid path is forced without auto-download."""
    with pytest.raises(ModelLoadError):
        load_face_landmarker(model_path="non_existent_dir/missing_model.task", auto_download=False)


def test_corrupt_image_handling():
    """Verify corrupt bytes fail gracefully with ImageProcessingError."""
    corrupt_bytes = b"NOT_A_REAL_IMAGE_DATA_12345"
    with pytest.raises(ImageProcessingError):
        load_image(corrupt_bytes)


def test_real_sample_image_pipeline():
    """Run full end-to-end pipeline on sample frontal neutral image."""
    sample_img_path = DATA_DIR / "sample_images" / "frontal_neutral.jpg"
    if not sample_img_path.exists():
        pytest.skip(f"Sample image {sample_img_path} not found.")

    img = load_image(sample_img_path)
    assert img is not None
    h, w = img.shape[:2]

    # 1. Landmark detection
    detector = load_face_landmarker()
    landmarks = detect_face_landmarks(img, detector=detector)
    assert landmarks is not None
    assert landmarks.num_landmarks >= 468

    # 2. Head pose
    pose = estimate_head_pose(landmarks, image_width=w, image_height=h)
    assert pose.success is True
    assert pose.is_frontal is True
    assert abs(pose.yaw_degrees) < 15.0

    # 3. Alignment
    alignment = align_face(landmarks, head_pose=pose)
    assert alignment.is_suitable_for_symmetry is True
    assert alignment.raw_reference_distance > 50.0

    # 4. Symmetry
    symmetry = calculate_symmetry(landmarks, alignment_result=alignment, head_pose=pose)
    assert symmetry.is_valid is True
    assert 0.0 <= symmetry.overall_normalized_asymmetry < 0.20
    assert 50.0 <= symmetry.symmetry_score <= 100.0

    # 5. Export summary
    summary = create_image_analysis_summary(pose, alignment, symmetry, "frontal_neutral.jpg", w, h)
    assert summary["metadata"]["face_detected"] is True
    assert summary["symmetry_analysis"]["is_valid"] is True


def test_real_sample_turned_face_pipeline():
    """Verify turned face sample is detected and flagged regarding frontal pose criteria."""
    sample_img_path = DATA_DIR / "sample_images" / "turned_pose.jpg"
    if not sample_img_path.exists():
        pytest.skip(f"Sample image {sample_img_path} not found.")

    img = load_image(sample_img_path)
    h, w = img.shape[:2]

    detector = load_face_landmarker()
    landmarks = detect_face_landmarks(img, detector=detector)
    assert landmarks is not None

    pose = estimate_head_pose(landmarks, image_width=w, image_height=h)
    assert pose.success is True
    # Turned pose has non-zero yaw
    assert abs(pose.yaw_degrees) > 5.0


def test_video_processing_pipeline(tmp_path: Path):
    """Create a short 5-frame synthetic video and verify process_video execution and resource cleanup."""
    sample_img_path = DATA_DIR / "sample_images" / "frontal_neutral.jpg"
    if not sample_img_path.exists():
        pytest.skip("Sample image required to build test video.")

    base_frame = cv2.imread(str(sample_img_path))
    h, w = base_frame.shape[:2]

    # Resize to small dimension for rapid video testing
    test_w, test_h = 320, 320
    resized_frame = cv2.resize(base_frame, (test_w, test_h))

    video_path = tmp_path / "test_clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video_path), fourcc, 10.0, (test_w, test_h))

    for _ in range(5):
        vw.write(resized_frame)
    vw.release()

    # Process video
    summary = process_video(
        video_source=video_path,
        max_frames=5,
        save_annotated_video=True,
        output_filename=str(tmp_path / "test_out.mp4")
    )

    assert summary.processed_frames == 5
    assert summary.face_detected_frames > 0
    assert not summary.metrics_df.empty
    assert "yaw_deg" in summary.metrics_df.columns
    assert "overall_asymmetry" in summary.metrics_df.columns
