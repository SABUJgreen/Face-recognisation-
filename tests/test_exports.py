"""Unit tests for CSV and JSON data export utilities."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.export_utils import (
    create_image_analysis_summary,
    export_image_summary_to_dataframe,
    export_to_csv,
    export_to_json,
)
from src.face_alignment import AlignmentResult
from src.head_pose import HeadPoseResult
from src.symmetry import SymmetryResult


def test_create_image_analysis_summary_with_valid_data():
    """Verify JSON-serializable dictionary generation."""
    mock_pose = HeadPoseResult(
        yaw_degrees=3.5,
        pitch_degrees=-2.1,
        roll_degrees=1.0,
        rotation_matrix=np.eye(3),
        translation_vector=np.array([[0], [0], [600]]),
        success=True,
        reprojection_error=0.45,
        is_frontal=True,
    )
    mock_align = AlignmentResult(
        aligned_points=np.zeros((100, 3)),
        rotation_angle_deg=1.0,
        scale_factor=1.2,
        translation=(300.0, 200.0),
        midline_x=0.0,
        reference_distance=100.0,
        is_suitable_for_symmetry=True,
        suitability_reason="Suitable",
        raw_reference_distance=83.3,
    )
    mock_sym = SymmetryResult(
        overall_normalized_asymmetry=0.025,
        symmetry_score=88.2,
        regional_asymmetry={"eyes": 0.02, "mouth": 0.03},
        pair_details=[],
        midline_x=0.0,
        reference_distance=100.0,
        valid_pair_count=10,
        total_pair_count=10,
        is_valid=True,
    )

    summary = create_image_analysis_summary(
        mock_pose, mock_align, mock_sym, source_filename="test.jpg", image_width=640, image_height=480
    )

    assert summary["metadata"]["source_filename"] == "test.jpg"
    assert summary["metadata"]["face_detected"] is True
    assert summary["head_pose"]["yaw_deg"] == 3.5
    assert summary["symmetry_analysis"]["symmetry_display_score"] == 88.2
    assert "eyes" in summary["symmetry_analysis"]["regional_asymmetry"]


def test_export_to_csv_and_json(tmp_path: Path):
    """Verify writing DataFrame and JSON files to disk."""
    mock_pose = HeadPoseResult(
        yaw_degrees=1.0,
        pitch_degrees=0.5,
        roll_degrees=-0.5,
        rotation_matrix=np.eye(3),
        translation_vector=np.zeros((3, 1)),
        success=True,
        reprojection_error=0.5,
        is_frontal=True,
    )
    summary = create_image_analysis_summary(mock_pose, None, None, "test.png", 800, 600)

    csv_file = tmp_path / "export_test.csv"
    json_file = tmp_path / "export_test.json"

    export_to_csv(summary, csv_file)
    export_to_json(summary, json_file)

    assert csv_file.exists()
    assert json_file.exists()

    df = pd.read_csv(csv_file)
    assert len(df) == 1
    assert df["source_filename"].iloc[0] == "test.png"
    assert df["yaw_deg"].iloc[0] == 1.0

    with open(json_file, "r") as f:
        data = json.load(f)
    assert data["metadata"]["source_filename"] == "test.png"
