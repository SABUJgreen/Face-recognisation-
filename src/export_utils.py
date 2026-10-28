"""Data export module supporting CSV and JSON serialization of face posture and symmetry metrics."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

from src.face_alignment import AlignmentResult
from src.head_pose import HeadPoseResult
from src.symmetry import SymmetryResult


def format_value(val: Any, decimals: int = 4) -> Any:
    """Helper to round float values and convert NaN to None for JSON compliance."""
    if val is None:
        return None
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return None
        return round(float(val), decimals)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, np.ndarray):
        return [format_value(x, decimals) for x in val.tolist()]
    return val


def create_image_analysis_summary(
    head_pose: Optional[HeadPoseResult],
    alignment: Optional[AlignmentResult],
    symmetry: Optional[SymmetryResult],
    source_filename: str = "image.jpg",
    image_width: int = 0,
    image_height: int = 0
) -> Dict[str, Any]:
    """Compile comprehensive analysis dictionary for a single analyzed image."""
    has_face = head_pose is not None
    summary: Dict[str, Any] = {
        "metadata": {
            "source_filename": source_filename,
            "image_width": image_width,
            "image_height": image_height,
            "face_detected": has_face,
            "disclaimer": symmetry.disclaimer if symmetry else (
                "Purely geometric landmark analysis. Not for clinical or diagnostic use."
            ),
        },
        "head_pose": {
            "success": head_pose.success if has_face else False,
            "is_frontal": head_pose.is_frontal if has_face else False,
            "yaw_deg": format_value(head_pose.yaw_degrees if has_face else None, 2),
            "pitch_deg": format_value(head_pose.pitch_degrees if has_face else None, 2),
            "roll_deg": format_value(head_pose.roll_degrees if has_face else None, 2),
            "reprojection_error_px": format_value(head_pose.reprojection_error if has_face else None, 2),
            "mp_euler_angles": format_value(head_pose.mp_euler_angles if has_face else None, 2),
        },
        "facial_alignment": {
            "is_suitable_for_symmetry": alignment.is_suitable_for_symmetry if alignment else False,
            "suitability_reason": alignment.suitability_reason if alignment else "No face",
            "in_plane_rotation_removed_deg": format_value(alignment.rotation_angle_deg if alignment else None, 2),
            "scale_factor_applied": format_value(alignment.scale_factor if alignment else None, 4),
            "inter_ocular_reference_distance_px": format_value(alignment.raw_reference_distance if alignment else None, 1),
        },
        "symmetry_analysis": {
            "is_valid": symmetry.is_valid if symmetry else False,
            "invalidity_reason": symmetry.invalidity_reason if symmetry else "No face",
            "overall_normalized_asymmetry": format_value(symmetry.overall_normalized_asymmetry if symmetry else None, 4),
            "symmetry_display_score": format_value(symmetry.symmetry_score if symmetry else None, 1),
            "valid_pair_count": symmetry.valid_pair_count if symmetry else 0,
            "total_pair_count": symmetry.total_pair_count if symmetry else 0,
            "regional_asymmetry": {
                k: format_value(v, 4)
                for k, v in (symmetry.regional_asymmetry if (symmetry and symmetry.is_valid) else {}).items()
            },
        }
    }
    return summary


def export_image_summary_to_dataframe(summary_dict: Dict[str, Any]) -> pd.DataFrame:
    """Flatten single-image summary into a single-row structured Pandas DataFrame."""
    meta = summary_dict.get("metadata", {})
    pose = summary_dict.get("head_pose", {})
    align = summary_dict.get("facial_alignment", {})
    sym = summary_dict.get("symmetry_analysis", {})
    reg = sym.get("regional_asymmetry", {})

    flat_data = {
        "source_filename": meta.get("source_filename"),
        "image_width": meta.get("image_width"),
        "image_height": meta.get("image_height"),
        "face_detected": meta.get("face_detected"),
        "head_pose_success": pose.get("success"),
        "is_frontal": pose.get("is_frontal"),
        "yaw_deg": pose.get("yaw_deg"),
        "pitch_deg": pose.get("pitch_deg"),
        "roll_deg": pose.get("roll_deg"),
        "reprojection_error_px": pose.get("reprojection_error_px"),
        "is_suitable_for_symmetry": align.get("is_suitable_for_symmetry"),
        "suitability_reason": align.get("suitability_reason"),
        "in_plane_rotation_removed_deg": align.get("in_plane_rotation_removed_deg"),
        "inter_ocular_reference_distance_px": align.get("inter_ocular_reference_distance_px"),
        "symmetry_valid": sym.get("is_valid"),
        "overall_normalized_asymmetry": sym.get("overall_normalized_asymmetry"),
        "symmetry_display_score": sym.get("symmetry_display_score"),
        "eyes_asymmetry": reg.get("eyes"),
        "eyebrows_asymmetry": reg.get("eyebrows"),
        "cheeks_asymmetry": reg.get("cheeks"),
        "mouth_asymmetry": reg.get("mouth"),
        "jawline_asymmetry": reg.get("jawline"),
        "valid_pair_count": sym.get("valid_pair_count"),
        "total_pair_count": sym.get("total_pair_count"),
        "invalidity_reason": sym.get("invalidity_reason"),
    }
    return pd.DataFrame([flat_data])


def export_to_csv(
    data: Union[pd.DataFrame, Dict[str, Any]],
    output_path: Union[str, Path]
) -> Path:
    """Export DataFrame or dictionary to a clean CSV file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, dict):
        df = export_image_summary_to_dataframe(data)
    else:
        df = data

    df.to_csv(str(p), index=False, na_rep="NA")
    return p


def export_to_json(
    summary_dict: Dict[str, Any],
    output_path: Union[str, Path]
) -> Path:
    """Export analysis dictionary to formatted JSON file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    return p
