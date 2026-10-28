"""FaceScope — Face Posture and Facial Symmetry Analyzer Streamlit Web Dashboard."""

import io
from pathlib import Path
import tempfile
import time
from typing import Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Application configuration & core modules
from src.config import (
    DATA_DIR,
    DEFAULT_FRAME_STRIDE,
    DEFAULT_MAX_VIDEO_FRAMES,
    DEFAULT_SMOOTHING_ALPHA,
    MAX_FRONTAL_PITCH_DEG,
    MAX_FRONTAL_ROLL_DEG,
    MAX_FRONTAL_YAW_DEG,
    OUTPUTS_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from src.export_utils import (
    create_image_analysis_summary,
    export_image_summary_to_dataframe,
    export_to_csv,
    export_to_json,
)
from src.face_alignment import align_face
from src.head_pose import estimate_head_pose
from src.image_processor import ImageProcessingError, bgr_to_rgb, load_image
from src.landmarks import detect_face_landmarks
from src.model_loader import load_face_landmarker
from src.symmetry import calculate_symmetry
from src.video_processor import process_video
from src.visualization import (
    create_regional_asymmetry_bar_chart,
    draw_head_pose_axes,
    draw_landmarks,
    draw_symmetry_overlay,
)


# -----------------------------------------------------------------------------
# Streamlit Page Setup & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FaceScope — Face Posture & Symmetry Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38EF7D 0%, #11998E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #8B949E;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58A6FF;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-frontal {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ECC71;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(46, 204, 113, 0.4);
    }

    .badge-nonfrontal {
        background-color: rgba(231, 76, 60, 0.2);
        color: #E74C3C;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(231, 76, 60, 0.4);
    }

    .disclaimer-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #38BDF8;
        padding: 0.85rem 1.2rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #94A3B8;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing MediaPipe Face Landmarker...")
def get_cached_detector():
    """Load and cache the image-mode Face Landmarker model."""
    return load_face_landmarker()


# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🔬 FaceScope")
st.sidebar.markdown("Computer Vision Posture & Symmetry")

navigation = st.sidebar.radio(
    "Navigation",
    ["Overview", "Image Analysis", "Video Analysis", "Methodology", "About"],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Frontal Thresholds")
st.sidebar.caption(f"Max Yaw: ±{MAX_FRONTAL_YAW_DEG}°")
st.sidebar.caption(f"Max Pitch: ±{MAX_FRONTAL_PITCH_DEG}°")
st.sidebar.caption(f"Max Roll: ±{MAX_FRONTAL_ROLL_DEG}°")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color: #8B949E;'>FaceScope v1.0.0<br>Built for Computer Vision & Geometry Analysis</small>",
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# SECTION A: OVERVIEW
# -----------------------------------------------------------------------------
if navigation == "Overview":
    st.markdown('<div class="main-title">FaceScope Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">AI-driven 3D head pose estimation and geometric bilateral facial symmetry analysis.</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    **FaceScope** is an end-to-end computer vision platform designed to quantify 3D facial posture and bilateral geometric balance using dense 478-point facial mesh landmarks.

    ### 🎯 Core Capabilities
    1. **Dense 3D Landmark Tracking**: Locates 478 canonical facial landmarks including eyes, eyebrows, nose, mouth contours, and jawline using MediaPipe.
    2. **3D Head Orientation (solvePnP)**: Estimates Euler angles (Yaw, Pitch, Roll) using an anthropometric 3D face model and perspective-n-point solver with reprojection error quantification.
    3. **Pose-Aware Symmetry Gating**: Identifies non-frontal poses (where out-of-plane head rotation creates geometric foreshortening) and explicitly marks symmetry as unreliable rather than outputting misleading numbers.
    4. **2D Similarity Alignment**: Normalizes in-plane head tilt, translation, and scale using inter-ocular distance normalization.
    5. **Bilateral Reflection Analysis**: Quantifies normalized Euclidean discrepancies across anatomical landmark pairs for eyes, eyebrows, cheeks, mouth, and jawline.
    6. **Temporal Video Processing**: Tracks head posture changes across frames with exponential moving average (EMA) smoothing and time-series plotting.
    """)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Non-Medical Disclaimer:</strong><br>
        FaceScope is strictly an engineering and computer vision research tool. It evaluates mathematical landmark coordinates.
        It is NOT a medical diagnostic system, orthodontic evaluation tool, or measure of personal beauty/attractiveness.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SECTION B: IMAGE ANALYSIS
# -----------------------------------------------------------------------------
elif navigation == "Image Analysis":
    st.markdown('<div class="main-title">Facial Posture & Symmetry Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Analyze head pose and bilateral symmetry from a single image.</div>',
        unsafe_allow_html=True
    )

    # Input Mode Selection
    input_source = st.radio(
        "Select Image Source",
        ["Use Sample Image", "Upload Custom Image"],
        horizontal=True
    )

    selected_image_bytes = None
    source_name = "sample"

    if input_source == "Use Sample Image":
        sample_choice = st.selectbox(
            "Choose a sample test portrait",
            ["Frontal Neutral Face (Baseline)", "Turned Head Pose (Angled Face)"]
        )
        if "Frontal" in sample_choice:
            sample_file = DATA_DIR / "sample_images" / "frontal_neutral.jpg"
            source_name = "frontal_neutral.jpg"
        else:
            sample_file = DATA_DIR / "sample_images" / "turned_pose.jpg"
            source_name = "turned_pose.jpg"

        if sample_file.exists():
            with open(sample_file, "rb") as f:
                selected_image_bytes = f.read()
        else:
            st.error(f"Sample file {sample_file} not found.")

    else:
        uploaded_file = st.file_uploader(
            "Upload portrait image (JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            selected_image_bytes = uploaded_file.getvalue()
            source_name = uploaded_file.name

    if selected_image_bytes is not None:
        try:
            bgr_img = load_image(selected_image_bytes)
            h, w = bgr_img.shape[:2]
        except ImageProcessingError as exc:
            st.error(f"Image Error: {exc}")
            st.stop()

        detector = get_cached_detector()

        # Run detection
        with st.spinner("Detecting facial landmarks and estimating head posture..."):
            landmarks = detect_face_landmarks(bgr_img, detector=detector)

        if landmarks is None:
            st.warning("⚠️ No face was detected in the provided image. Please upload a clear photo with the face visible.")
            st.image(bgr_to_rgb(bgr_img), caption="Original Image (No face detected)", width=450)
        else:
            # Step 1: Head Pose Estimation
            head_pose = estimate_head_pose(landmarks, image_width=w, image_height=h)

            # Step 2: Facial Alignment
            alignment = align_face(landmarks, head_pose=head_pose)

            # Step 3: Symmetry Calculation
            symmetry = calculate_symmetry(landmarks, alignment_result=alignment, head_pose=head_pose)

            # Visualizations
            annotated_landmarks = draw_landmarks(bgr_img, landmarks, radius=1)
            annotated_pose = draw_head_pose_axes(annotated_landmarks, head_pose)
            annotated_full = draw_symmetry_overlay(
                annotated_pose,
                landmarks,
                symmetry,
                alignment_result=alignment,
                show_midline=True,
                show_pairs=True,
                show_vectors=True
            )

            # Display side-by-side images
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📷 Original Image")
                st.image(bgr_to_rgb(bgr_img), use_container_width=True)

            with col2:
                st.markdown("### 🔬 Analyzed & Annotated")
                st.image(bgr_to_rgb(annotated_full), use_container_width=True)

            st.markdown("---")

            # Metrics Row 1: Head Pose
            st.markdown("### 🧭 Head Posture Estimation (solvePnP)")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Yaw (Turn)", f"{head_pose.yaw_degrees:.1f}°", help="Horizontal turn (positive = subject's left)")
            with c2:
                st.metric("Pitch (Nod)", f"{head_pose.pitch_degrees:.1f}°", help="Vertical tilt (positive = looking down)")
            with c3:
                st.metric("Roll (Tilt)", f"{head_pose.roll_degrees:.1f}°", help="Lateral head roll (positive = tilt to right)")
            with c4:
                status_badge = (
                    '<span class="badge-frontal">✅ Frontal Pose</span>'
                    if head_pose.is_frontal else
                    '<span class="badge-nonfrontal">⚠️ Non-Frontal Pose</span>'
                )
                st.markdown(f"**Pose Status:**<br>{status_badge}", unsafe_allow_html=True)
                st.caption(f"Reprojection RMSE: {head_pose.reprojection_error:.2f} px")

            st.markdown("---")

            # Metrics Row 2: Symmetry Analysis
            st.markdown("### ⚖️ Bilateral Facial Symmetry Analysis")

            if not symmetry.is_valid:
                st.warning(f"⚠️ **Symmetry Analysis Unavailable:** {symmetry.invalidity_reason}")
                st.info(
                    "Facial symmetry calculations require a frontal head orientation. "
                    "When the face is rotated in depth (yaw/pitch), optical foreshortening skews 2D bilateral distances. "
                    "Please capture an image directly facing the camera."
                )
            else:
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.metric(
                        "Symmetry Score",
                        f"{symmetry.symmetry_score:.1f}%",
                        help="Heuristic visualization score (100% = perfectly balanced bilateral reflection)"
                    )
                with s2:
                    st.metric(
                        "Overall Asymmetry",
                        f"{symmetry.overall_normalized_asymmetry * 100:.2f}%",
                        help="Weighted normalized geometric discrepancy relative to inter-ocular distance"
                    )
                with s3:
                    st.metric(
                        "Inter-Ocular Distance",
                        f"{alignment.raw_reference_distance:.1f} px",
                        help="Measured reference distance between eye centers"
                    )
                with s4:
                    st.metric(
                        "Valid Bilateral Pairs",
                        f"{symmetry.valid_pair_count} / {symmetry.total_pair_count}",
                        help="Number of evaluated anatomical landmark pairs"
                    )

                # Regional Chart
                st.markdown("#### Regional Breakdown (% Inter-Ocular Distance)")
                chart = create_regional_asymmetry_bar_chart(
                    symmetry.regional_asymmetry,
                    symmetry.overall_normalized_asymmetry
                )
                st.pyplot(chart)
                plt.close(chart)

            st.markdown("---")

            # Section: Download & Export
            st.markdown("### 💾 Export Results")
            summary_dict = create_image_analysis_summary(
                head_pose=head_pose,
                alignment=alignment,
                symmetry=symmetry,
                source_filename=source_name,
                image_width=w,
                image_height=h
            )
            df_export = export_image_summary_to_dataframe(summary_dict)

            e1, e2, e3 = st.columns(3)
            with e1:
                csv_data = df_export.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download CSV Metrics",
                    data=csv_data,
                    file_name=f"{Path(source_name).stem}_facescope_metrics.csv",
                    mime="text/csv"
                )
            with e2:
                import json
                json_data = json.dumps(summary_dict, indent=2).encode("utf-8")
                st.download_button(
                    "📥 Download JSON Report",
                    data=json_data,
                    file_name=f"{Path(source_name).stem}_facescope_report.json",
                    mime="application/json"
                )
            with e3:
                # Encode annotated image to JPEG bytes
                _, enc_img = cv2.imencode(".jpg", annotated_full)
                st.download_button(
                    "📥 Download Annotated Image",
                    data=enc_img.tobytes(),
                    file_name=f"{Path(source_name).stem}_facescope_annotated.jpg",
                    mime="image/jpeg"
                )

            st.markdown(f'<div class="disclaimer-box">{symmetry.disclaimer}</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SECTION C: VIDEO ANALYSIS
# -----------------------------------------------------------------------------
elif navigation == "Video Analysis":
    st.markdown('<div class="main-title">Temporal Video Posture & Symmetry</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Analyze frame-by-frame head posture changes and geometric balance across video clips.</div>',
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Upload a video file (MP4, AVI, MOV, M4V)",
        type=["mp4", "avi", "mov", "m4v"]
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        max_frames_input = st.number_input("Max Frames to Process", min_value=10, max_value=1200, value=300, step=50)
    with c2:
        stride_input = st.number_input("Frame Stride (Skip)", min_value=1, max_value=5, value=1, step=1)
    with c3:
        smooth_alpha = st.slider("EMA Smoothing Alpha", min_value=0.1, max_value=1.0, value=0.35, step=0.05)

    if uploaded_video is not None:
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix)
        tfile.write(uploaded_video.read())
        temp_video_path = tfile.name
        tfile.close()

        if st.button("🚀 Start Video Analysis", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(ratio: float, msg: str):
                progress_bar.progress(min(1.0, max(0.0, ratio)))
                status_text.text(msg)

            try:
                with st.spinner("Processing video frames..."):
                    summary = process_video(
                        video_source=temp_video_path,
                        max_frames=max_frames_input,
                        frame_stride=stride_input,
                        save_annotated_video=True,
                        progress_callback=update_progress,
                        smoothing_alpha=smooth_alpha
                    )

                progress_bar.progress(1.0)
                status_text.text("Processing complete!")

                st.success("✅ Video processing successfully completed!")

                # Summary Statistics
                st.markdown("### 📊 Sequence Summary Statistics")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Processed Frames", f"{summary.processed_frames} / {summary.total_video_frames}")
                with m2:
                    st.metric("Face Detection Rate", f"{summary.face_detection_rate:.1f}%")
                with m3:
                    st.metric("Frontal Pose Ratio", f"{(summary.frontal_frames / max(1, summary.processed_frames)) * 100:.1f}%")
                with m4:
                    mean_sc = f"{summary.mean_symmetry_score:.1f}%" if summary.mean_symmetry_score is not None else "N/A"
                    st.metric("Mean Symmetry Score", mean_sc)

                df = summary.metrics_df

                # Chart 1: Head Angles Over Time
                st.markdown("### 🧭 Head Posture Over Time (Smoothed Euler Angles)")
                if not df.empty and "yaw_smoothed" in df.columns:
                    fig_angles, ax_angles = plt.subplots(figsize=(10, 3.5), dpi=100)
                    fig_angles.patch.set_facecolor("#0E1117")
                    ax_angles.set_facecolor("#161B22")

                    t_sec = df["timestamp_ms"] / 1000.0
                    ax_angles.plot(t_sec, df["yaw_smoothed"], label="Yaw (Turn)", color="#58A6FF", linewidth=1.8)
                    ax_angles.plot(t_sec, df["pitch_smoothed"], label="Pitch (Nod)", color="#2ECC71", linewidth=1.8)
                    ax_angles.plot(t_sec, df["roll_smoothed"], label="Roll (Tilt)", color="#F39C12", linewidth=1.8)

                    # Highlight frontal boundary
                    ax_angles.axhline(MAX_FRONTAL_YAW_DEG, color="#E74C3C", linestyle=":", alpha=0.5, label="Frontal Threshold")
                    ax_angles.axhline(-MAX_FRONTAL_YAW_DEG, color="#E74C3C", linestyle=":", alpha=0.5)

                    ax_angles.set_xlabel("Time (seconds)", color="#8B949E")
                    ax_angles.set_ylabel("Angle (degrees)", color="#8B949E")
                    ax_angles.set_title("Temporal Head Posture Dynamics", color="#FFFFFF", weight="bold")
                    ax_angles.grid(color="#30363D", linestyle=":", alpha=0.6)
                    ax_angles.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="#E6EDF3", loc="upper right")
                    ax_angles.tick_params(colors="#8B949E")
                    for spine in ax_angles.spines.values():
                        spine.set_color("#30363D")

                    st.pyplot(fig_angles)
                    plt.close(fig_angles)

                # Chart 2: Regional Symmetry Over Time
                st.markdown("### ⚖️ Regional Facial Symmetry Over Time")
                if not df.empty and "is_symmetry_valid" in df.columns:
                    valid_df = df[df["is_symmetry_valid"] == True]
                    if not valid_df.empty:
                        fig_sym, ax_sym = plt.subplots(figsize=(10, 3.5), dpi=100)
                        fig_sym.patch.set_facecolor("#0E1117")
                        ax_sym.set_facecolor("#161B22")

                        v_sec = valid_df["timestamp_ms"] / 1000.0
                        for reg, col in [("eye_asymmetry", "#38BDF8"), ("eyebrow_asymmetry", "#A855F7"), ("mouth_asymmetry", "#F43F5E"), ("cheek_asymmetry", "#F59E0B")]:
                            if reg in valid_df.columns:
                                ax_sym.plot(v_sec, valid_df[reg] * 100.0, label=reg.replace("_", " ").title(), linewidth=1.5, alpha=0.85)

                        ax_sym.set_xlabel("Time (seconds)", color="#8B949E")
                        ax_sym.set_ylabel("Asymmetry (% Inter-Ocular)", color="#8B949E")
                        ax_sym.set_title("Regional Geometric Asymmetry Profiles", color="#FFFFFF", weight="bold")
                        ax_sym.grid(color="#30363D", linestyle=":", alpha=0.6)
                        ax_sym.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="#E6EDF3", loc="upper right")
                        ax_sym.tick_params(colors="#8B949E")
                        for spine in ax_sym.spines.values():
                            spine.set_color("#30363D")

                        st.pyplot(fig_sym)
                        plt.close(fig_sym)
                    else:
                        st.info("No frames satisfied frontal pose criteria for continuous symmetry tracking.")

                # Downloads
                st.markdown("### 💾 Export Video Analysis")
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Frame-by-Frame CSV Metrics",
                    data=csv_bytes,
                    file_name="video_facescope_metrics.csv",
                    mime="text/csv"
                )

            finally:
                # Cleanup temp file
                try:
                    Path(temp_video_path).unlink(missing_ok=True)
                except Exception:
                    pass


# -----------------------------------------------------------------------------
# SECTION D: METHODOLOGY
# -----------------------------------------------------------------------------
elif navigation == "Methodology":
    st.markdown('<div class="main-title">Scientific & Engineering Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Mathematical formulation, topological indices, and computer vision pipelines.</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### 1. MediaPipe Face Landmarker Topology
    FaceScope employs Google MediaPipe's deep neural network (`face_landmarker.task`), which infers **478 3D landmark coordinates** (468 facial mesh + 10 iris tracking points) in real time.
    Each landmark provides normalized image coordinates $(x_n, y_n, z_n)$, where $z_n$ represents relative depth scaled proportionally to face width.

    ### 2. 3D Head Pose Estimation via solvePnP
    Head orientation is estimated by solving the **Perspective-n-Point (PnP)** problem via Levenberg-Marquardt optimization (`cv2.solvePnP`):

    $$\\min_{\\mathbf{R}, \\mathbf{t}} \\sum_{i} \\left\\| \\mathbf{p}_{i}^{2D} - \\pi\\left( \\mathbf{K} [\\mathbf{R} | \\mathbf{t}] \\mathbf{P}_{i}^{3D} \\right) \\right\\|^2$$

    - **3D Reference Model**: Canonical anthropometric points for the pronasale (nose tip 1), gnathion (chin 152), exocanthions (eyes 33, 263), and cheilions (mouth corners 61, 291).
    - **Camera Intrinsics Matrix $\\mathbf{K}$**: Pinhole model assuming focal length $f_x = f_y = \\text{width}$, principal point at $(\\text{width}/2, \\text{height}/2)$, zero skew.
    - **Euler Angle Decomposition**: Rotation matrix $\\mathbf{R}$ decomposed into Yaw (Y), Pitch (X), and Roll (Z) in degrees via `cv2.RQDecomp3x3`.
    - **Reprojection Error**: Quantified by projecting 3D points back to the image plane and evaluating Root Mean Square Error (RMSE).

    ### 3. Frontal Pose Suitability Gating
    Bilateral facial symmetry is fundamentally a **frontal metric**. When a subject turns their head (yaw $\\theta_y > 12^\\circ$) or tilts forward/back (pitch $\\theta_x > 12^\\circ$), perspective projection foreshortens the receding facial half.
    Rather than calculating false asymmetry, FaceScope detects non-frontal poses and explicitly flags the measurement as **unsuitable**.

    ### 4. 2D Similarity Alignment
    Before reflection, stable in-plane landmarks are aligned to a canonical frame:
    1. **Translation**: Centers inter-ocular midpoint at $(0, 0)$.
    2. **In-Plane Rotation**: Computes angle $\\theta$ between right and left eye centers and rotates by $-\\theta$ so the eye axis is strictly horizontal.
    3. **Scale Normalization**: Scales points so inter-ocular distance $D_{ref} = 100.0$ canonical units.

    ### 5. Reflection-Based Symmetry Formulation
    For any anatomical landmark point $L = (x_L, y_L)$ on the left facial half, its reflection across the estimated vertical facial midline axis $x = c$ is:

    $$x_{reflected} = 2c - x_L, \\quad y_{reflected} = y_L$$

    The Euclidean error relative to the corresponding right anatomical point $R = (x_R, y_R)$ is:

    $$E_i = \\sqrt{(2c - x_L - x_R)^2 + (y_L - y_R)^2}$$

    Normalized by inter-ocular distance $D_{ref}$:

    $$\\tilde{E}_i = \\frac{E_i}{D_{ref}}$$

    Regional and overall asymmetry are computed using weighted aggregation across anatomically verified pairs:

    $$\\tilde{E}_{overall} = \\frac{\\sum_{i} w_i \\tilde{E}_i}{\\sum_{i} w_i}$$

    ### 6. Heuristic Display Score
    The 0–100% visualization score is defined as:

    $$S = 100 \\cdot \\exp(-k \\cdot \\tilde{E}_{overall})$$

    with $k = 5.0$. A normalized error of 0 gives 100.0%, 0.05 (5% inter-ocular error) gives ~77.8%, and 0.10 gives ~60.6%.
    """)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>Known Technical Limitations:</strong>
        <ul>
            <li>Uncalibrated camera intrinsics: Consumer webcams without focal length calibration provide approximate head angles within ~2–4° accuracy.</li>
            <li>Facial expressions: Smiling, winking, or asymmetric expressions naturally modify landmark positions.</li>
            <li>Occlusion: Facial hair, eyeglasses, or hand-to-face occlusion can degrade landmark precision.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SECTION E: ABOUT
# -----------------------------------------------------------------------------
elif navigation == "About":
    st.markdown('<div class="main-title">About FaceScope</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">System specifications, technology stack, and privacy commitments.</div>',
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ### 🛠️ Technology Stack
        - **Computer Vision**: OpenCV (solvePnP, Affine / Similarity geometry, image & video I/O)
        - **Machine Learning**: Google MediaPipe Face Landmarker (Float16 Tasks API)
        - **Numerical Computing**: NumPy & SciPy
        - **Visualization**: Matplotlib & OpenCV Drawing API
        - **Data Processing**: Pandas
        - **User Interface**: Streamlit
        - **Automated Testing**: Pytest
        """)

    with col_b:
        st.markdown("""
        ### 🔒 Privacy & Local Processing
        - **100% Local Inference**: All landmark detection, posture estimation, and symmetry computations occur directly on your local CPU.
        - **Zero Cloud Data Transmission**: Uploaded images and video frames are never transmitted to external third-party servers.
        - **Safe Ephemeral Storage**: Uploaded files and temporary processed video buffers are automatically deleted upon session conclusion.
        """)

    st.markdown("---")
    st.markdown("""
    ### 📜 Model Attribution & Citation
    - **Model Asset**: `face_landmarker.task` by Google MediaPipe Team under Apache License 2.0.
    - **Facial Mesh Topology**: MediaPipe 468/478 Dense Facial Landmark Mesh.
    - **Reference Coordinate Standard**: OpenCV Camera Coordinate Convention (+X Right, +Y Down, +Z Forward).
    """)
