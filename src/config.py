"""Configuration parameters, landmark topological indices, and model paths for FaceScope."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_MODEL_FILENAME = "face_landmarker.task"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_FILENAME

# Official Google MediaPipe download URL for face landmarker task
OFFICIAL_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# Upload limits
MAX_IMAGE_SIZE_BYTES = int(os.getenv("FACESCOPE_MAX_IMAGE_MB", "20")) * 1024 * 1024
MAX_VIDEO_SIZE_BYTES = int(os.getenv("FACESCOPE_MAX_VIDEO_MB", "100")) * 1024 * 1024
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".m4v"}

# Video processing limits
DEFAULT_MAX_VIDEO_FRAMES = int(os.getenv("FACESCOPE_MAX_VIDEO_FRAMES", "600"))
DEFAULT_MAX_VIDEO_DURATION_SEC = float(os.getenv("FACESCOPE_MAX_VIDEO_DURATION_SEC", "60.0"))
DEFAULT_FRAME_STRIDE = int(os.getenv("FACESCOPE_VIDEO_FRAME_STRIDE", "1"))

# Frontal pose suitability thresholds (degrees)
# Symmetry measurements become mathematically ungrounded under out-of-plane rotation
MAX_FRONTAL_YAW_DEG = float(os.getenv("FACESCOPE_MAX_YAW_DEG", "12.0"))
MAX_FRONTAL_PITCH_DEG = float(os.getenv("FACESCOPE_MAX_PITCH_DEG", "12.0"))
MAX_FRONTAL_ROLL_DEG = float(os.getenv("FACESCOPE_MAX_ROLL_DEG", "12.0"))

# Temporal smoothing factor for video (Exponential Moving Average)
# smoothed = alpha * current + (1 - alpha) * previous
DEFAULT_SMOOTHING_ALPHA = 0.35

# ---------------------------------------------------------------------------
# MediaPipe Facial Landmark Topology Definitions (Verified against MediaPipe 468/478 mesh)
# ---------------------------------------------------------------------------

# Facial Midline landmarks along anatomical sagittal plane (superior to inferior)
MIDLINE_LANDMARK_INDICES: List[int] = [
    10,   # Forehead top / trichion region
    151,  # Forehead mid
    9,    # Glabella region
    8,    # Mid brow
    168,  # Sellion / Nasion (root of nose)
    6,    # Upper nasal bridge
    197,  # Mid nasal bridge
    195,  # Lower nasal bridge
    5,    # Supratip
    4,    # Tip of nose apex
    1,    # Nose tip primary
    19,   # Infratip lobule
    94,   # Columella
    2,    # Subnasale
    164,  # Philtrum top
    0,    # Upper lip vermilion border center (labrale superius)
    11,   # Upper lip interior
    13,   # Oral commissure center
    14,   # Lower lip interior
    17,   # Lower lip vermilion border center (labrale inferius)
    18,   # Sublabial depression
    200,  # Pogonion (chin anterior projection)
    199,  # Menton top
    175,  # Menton mid
    152,  # Gnathion (chin base point)
]

# Anatomical Left-Right paired landmarks: Tuple of (subject_right_idx, subject_left_idx)
# Note: In standard front view:
# Subject Right = Image Left (viewer's left, lower x)
# Subject Left  = Image Right (viewer's right, higher x)
SYMMETRY_PAIRS_BY_REGION: Dict[str, List[Tuple[int, int]]] = {
    "eyes": [
        (33, 263),   # Outer corners (exocanthion)
        (133, 362),  # Inner corners (endocanthion)
        (159, 386),  # Superior eyelid margin apex
        (145, 374),  # Inferior eyelid margin apex
        (160, 385),  # Superior mid-outer eyelid
        (144, 373),  # Inferior mid-outer eyelid
        (158, 387),  # Superior mid-inner eyelid
        (153, 380),  # Inferior mid-inner eyelid
    ],
    "eyebrows": [
        (46, 276),   # Eyebrow lateral tail
        (53, 283),   # Eyebrow lateral arch bottom
        (52, 282),   # Eyebrow mid arch bottom
        (65, 295),   # Eyebrow mid arch top
        (55, 285),   # Eyebrow inner arch
        (70, 300),   # Eyebrow medial head
        (105, 334),  # Eyebrow superior crest
        (107, 336),  # Eyebrow superior lateral
    ],
    "cheeks": [
        (116, 345),  # Zygomatic cheek peak
        (123, 352),  # Anterior mid-cheek
        (147, 376),  # Inferior cheek
        (234, 454),  # Lateral cheek / zygion contour
        (138, 367),  # Upper maxillary contour
        (215, 435),  # Lower maxillary contour
    ],
    "mouth": [
        (61, 291),   # Cheilion (mouth oral commissure / corner)
        (39, 269),   # Upper lip lateral contour
        (37, 267),   # Crista philtri (Cupid's bow crest peaks)
        (84, 314),   # Lower lip lateral contour
        (181, 405),  # Lower lip bottom boundary
        (78, 308),   # Stomion lateral inner edge
        (81, 311),   # Upper lip vermilion margin lateral
    ],
    "jawline": [
        (172, 397),  # Gonion (mandibular angle)
        (136, 365),  # Mandibular body lateral
        (150, 379),  # Mandibular body anterior
        (176, 400),  # Paramarginal chin contour
        (127, 356),  # Auricular insertion / tragus level
        (93, 323),   # Upper jaw ramus
        (58, 288),   # Lower jaw ramus
    ],
}

# Regional weights for aggregated symmetry index
# Calibrated according to perceptually dominant facial balance zones
REGIONAL_WEIGHTS: Dict[str, float] = {
    "eyes": 1.0,
    "eyebrows": 0.8,
    "cheeks": 0.8,
    "mouth": 1.0,
    "jawline": 0.7,
}

# ---------------------------------------------------------------------------
# Canonical 3D Reference Face Model for Head Pose Estimation (cv2.solvePnP)
# Units in millimeters, standardized anthropometric facial model
# Coordinate system: X right, Y down, Z away (OpenCV camera convention)
# ---------------------------------------------------------------------------
POSE_REFERENCE_3D_POINTS: Dict[int, Tuple[float, float, float]] = {
    1:   (0.0, 0.0, 0.0),          # Pronasale (Nose tip)
    152: (0.0, 110.0, -50.0),      # Gnathion (Chin tip)
    33:  (-65.0, -45.0, -35.0),    # Subject Right eye outer corner (viewer's left)
    263: (65.0, -45.0, -35.0),     # Subject Left eye outer corner (viewer's right)
    61:  (-45.0, 55.0, -30.0),     # Subject Right mouth corner
    291: (45.0, 55.0, -30.0),      # Subject Left mouth corner
    168: (0.0, -48.0, -18.0),      # Sellion / Nasion
    10:  (0.0, -110.0, -45.0),     # Glabella / Superior forehead
}

# Inter-pupillary reference landmarks for normalising geometric error
# Subject right eye center (approx 159 & 145 average or iris 468)
# Subject left eye center (approx 386 & 374 average or iris 473)
EYE_CENTER_RIGHT_INDICES: List[int] = [33, 133, 159, 145]
EYE_CENTER_LEFT_INDICES: List[int] = [263, 362, 386, 374]


@dataclass
class HeadPoseConfig:
    """Configuration for solvePnP head pose estimation."""
    points_3d: Dict[int, Tuple[float, float, float]] = field(
        default_factory=lambda: POSE_REFERENCE_3D_POINTS
    )
    max_frontal_yaw: float = MAX_FRONTAL_YAW_DEG
    max_frontal_pitch: float = MAX_FRONTAL_PITCH_DEG
    max_frontal_roll: float = MAX_FRONTAL_ROLL_DEG


@dataclass
class SymmetryConfig:
    """Configuration for geometric symmetry calculation."""
    pairs_by_region: Dict[str, List[Tuple[int, int]]] = field(
        default_factory=lambda: SYMMETRY_PAIRS_BY_REGION
    )
    regional_weights: Dict[str, float] = field(
        default_factory=lambda: REGIONAL_WEIGHTS
    )
    midline_indices: List[int] = field(
        default_factory=lambda: MIDLINE_LANDMARK_INDICES
    )
    # Heuristic score decay factor: score = 100 * exp(-decay * normalized_error)
    # A normalized error of 0 gives 100%, 0.05 gives ~78%, 0.10 gives ~60%
    score_decay_factor: float = 5.0
