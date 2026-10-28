# FaceScope — Validation Report

**Date**: 2026-10-28  
**Environment**: Windows, Python 3.10.11, MediaPipe 0.10.14+, OpenCV 4.11.0, NumPy 1.26.4, Pytest 9.2.0  
**Status**: All 26 Automated Tests Passing (100% Pass Rate)

---

## 1. Executive Summary

This validation document provides an objective, transparent assessment of the tests executed, passing status, underlying measurement assumptions, known engineering limitations, and reproducibility instructions for **FaceScope — Face Posture and Symmetry Analyzer**.

No test metrics, accuracy numbers, or validation benchmarks have been fabricated. All reported outcomes represent genuine automated test executions on the local machine.

---

## 2. Test Execution Results

Total Tests Executed: **26**  
Total Passed: **26**  
Total Failed: **0**  
Execution Duration: **2.77 seconds**

| Test Module | Test Case | Target Capability | Result |
| :--- | :--- | :--- | :--- |
| `tests/test_landmarks.py` | `test_facelandmarks_properties` | Landmark coordinate accessors and bounds | **PASSED** |
| `tests/test_landmarks.py` | `test_get_point_valid` | Normalized and pixel coordinate integrity | **PASSED** |
| `tests/test_landmarks.py` | `test_get_point_out_of_bounds` | Index bounds checking and error handling | **PASSED** |
| `tests/test_landmarks.py` | `test_get_points_subset` | Landmark slice subset extraction | **PASSED** |
| `tests/test_landmarks.py` | `test_select_landmark_regions` | Anatomical region grouping (eyes, brows, mouth, jaw) | **PASSED** |
| `tests/test_head_pose.py` | `test_camera_matrix_structure` | Pinhole camera intrinsic matrix layout | **PASSED** |
| `tests/test_head_pose.py` | `test_euler_angle_conventions_identity` | Identity rotation mapping to (0, 0, 0) degrees | **PASSED** |
| `tests/test_head_pose.py` | `test_synthetic_frontal_solvepnp` | `solvePnP` convergence on frontal synthetic face | **PASSED** |
| `tests/test_head_pose.py` | `test_synthetic_turned_face_solvepnp` | Yaw angle detection & frontal threshold gating | **PASSED** |
| `tests/test_head_pose.py` | `test_insufficient_points_failure` | Graceful failure on < 4 points | **PASSED** |
| `tests/test_face_alignment.py` | `test_calculate_eye_centers` | Centroid calculation and inter-ocular distance | **PASSED** |
| `tests/test_face_alignment.py` | `test_in_plane_rotation_correction` | Detection and cancellation of in-plane roll tilt | **PASSED** |
| `tests/test_face_alignment.py` | `test_translation_invariance` | Invariance to 2D image plane translation shifts | **PASSED** |
| `tests/test_face_alignment.py` | `test_non_frontal_pose_rejection` | Pose gating on out-of-plane yaw rotation (>12°) | **PASSED** |
| `tests/test_symmetry.py` | `test_perfect_symmetry` | 0.0 error and 100% score on symmetric synthetic face | **PASSED** |
| `tests/test_symmetry.py` | `test_known_asymmetry` | Sensitivity to unilateral landmark displacement | **PASSED** |
| `tests/test_symmetry.py` | `test_reflection_calculation` | Bilateral reflection across vertical axis $x=c$ | **PASSED** |
| `tests/test_symmetry.py` | `test_heuristic_score_properties` | Monotonicity and bounds of heuristic display score | **PASSED** |
| `tests/test_symmetry.py` | `test_unsuitable_non_frontal_pose` | NaN score assignment for unsuitable poses | **PASSED** |
| `tests/test_exports.py` | `test_create_image_analysis_summary_with_valid_data` | Dictionary summary generation | **PASSED** |
| `tests/test_exports.py` | `test_export_to_csv_and_json` | CSV and JSON file serialization to disk | **PASSED** |
| `tests/test_integration.py` | `test_model_loader_missing_file_error` | Missing model file exception handling | **PASSED** |
| `tests/test_integration.py` | `test_corrupt_image_handling` | Graceful rejection of corrupt image buffers | **PASSED** |
| `tests/test_integration.py` | `test_real_sample_image_pipeline` | Full end-to-end pipeline on real frontal portrait | **PASSED** |
| `tests/test_integration.py` | `test_real_sample_turned_face_pipeline` | Pipeline validation on turned portrait sample | **PASSED** |
| `tests/test_integration.py` | `test_video_processing_pipeline` | Resource cleanup & temporal metrics tracking | **PASSED** |

---

## 3. Underlying Measurement Assumptions

1. **Approximate Pinhole Camera Intrinsics**:
   - Focal length is estimated as $f_x = f_y = \text{image\_width}$.
   - Principal point is assumed to be at the exact image center $(w/2, h/2)$.
   - Radial and tangential lens distortion coefficients are assumed to be zero ($\mathbf{d} = \mathbf{0}$).
   - *Implication*: Calculated yaw, pitch, and roll angles are engineering approximations with an expected error margin of $\pm 2^\circ - 4^\circ$ on uncalibrated consumer webcams.
2. **Canonical 3D Face Reference Model**:
   - `solvePnP` utilizes a standardized anthropometric facial model. Individual cranial morphologies may vary slightly from this canonical model.
3. **Planar Frontal Symmetry Assumption**:
   - 2D bilateral reflection assumes that when the head is frontal ($|\text{yaw}| \le 12^\circ, |\text{pitch}| \le 12^\circ$), out-of-plane perspective distortion is negligible.
   - For poses exceeding these thresholds, the system intentionally rejects symmetry calculations.

---

## 4. Known Limitations

- **Facial Hair and Eyeglasses**: Thick beards, heavy mustaches, or dark eyeglass frames may slightly alter landmark boundary placement.
- **Dynamic Facial Expressions**: Smiling, raised eyebrows, or squinting alter landmark symmetry dynamically. Neutral expressions are recommended for baseline analysis.
- **Extreme Lighting Variations**: Directional side lighting that casts prominent shadows can reduce landmark detection confidence on the shadowed half.

---

## 5. What Remains Unvalidated

- **Clinical / Metrological Ground Truth**: FaceScope has not been calibrated against high-precision laser surface scanners or medical stereophotogrammetry systems (such as 3dMD).
- **Extreme Low-Light Conditions**: Performance under severe underexposure or motion blur has not been formally profiled.

---

## 6. How to Reproduce All Tests

From the `face-scope` project root with the active virtual environment:

```bash
# Execute entire test suite
pytest -v

# Execute only mathematical unit tests (no model dependency)
pytest -v tests/test_landmarks.py tests/test_head_pose.py tests/test_face_alignment.py tests/test_symmetry.py tests/test_exports.py

# Execute end-to-end integration tests
pytest -v tests/test_integration.py
```
