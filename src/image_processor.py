"""Image input validation, decoding, color conversion, and MediaPipe format transformation."""

import io
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import MAX_IMAGE_SIZE_BYTES, SUPPORTED_IMAGE_EXTENSIONS


class ImageProcessingError(ValueError):
    """Raised when an image is corrupt, unreadable, or fails validation."""
    pass


def validate_image_file(
    file_source: Union[str, Path, bytes, io.BytesIO],
    filename: Optional[str] = None
) -> None:
    """Validate image file extension and size constraints.

    Args:
        file_source: File path, bytes, or BytesIO buffer.
        filename: Optional filename for extension checking when passing bytes.

    Raises:
        ImageProcessingError: If validation fails.
    """
    if isinstance(file_source, (str, Path)):
        p = Path(file_source)
        if not p.exists():
            raise ImageProcessingError(f"File not found: '{p}'")
        ext = p.suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ImageProcessingError(
                f"Unsupported image format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
            )
        size = p.stat().st_size
        if size == 0:
            raise ImageProcessingError(f"Image file '{p}' is empty (0 bytes).")
        if size > MAX_IMAGE_SIZE_BYTES:
            max_mb = MAX_IMAGE_SIZE_BYTES // (1024 * 1024)
            raise ImageProcessingError(
                f"Image file exceeds maximum allowable size of {max_mb} MB ({size / (1024 * 1024):.1f} MB provided)."
            )
    elif isinstance(file_source, (bytes, io.BytesIO)):
        data = file_source if isinstance(file_source, bytes) else file_source.getvalue()
        if len(data) == 0:
            raise ImageProcessingError("Uploaded image buffer is empty.")
        if len(data) > MAX_IMAGE_SIZE_BYTES:
            max_mb = MAX_IMAGE_SIZE_BYTES // (1024 * 1024)
            raise ImageProcessingError(
                f"Uploaded image exceeds maximum allowable size of {max_mb} MB."
            )
        if filename:
            ext = Path(filename).suffix.lower()
            if ext and ext not in SUPPORTED_IMAGE_EXTENSIONS:
                raise ImageProcessingError(
                    f"Unsupported image extension '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
                )


def load_image(
    image_source: Union[str, Path, bytes, io.BytesIO, np.ndarray, Image.Image]
) -> np.ndarray:
    """Safely decode and normalize an image into a BGR NumPy array (standard OpenCV format).

    Args:
        image_source: File path, raw bytes, BytesIO, PIL Image, or existing NumPy array.

    Returns:
        np.ndarray: BGR uint8 image of shape (H, W, 3).

    Raises:
        ImageProcessingError: If decoding fails or image is invalid.
    """
    if isinstance(image_source, np.ndarray):
        img = image_source.copy()
    elif isinstance(image_source, Image.Image):
        # Convert PIL Image to RGB numpy array, then to BGR
        rgb_arr = np.array(image_source.convert("RGB"))
        img = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    elif isinstance(image_source, (str, Path)):
        validate_image_file(image_source)
        img = cv2.imread(str(image_source), cv2.IMREAD_COLOR)
        if img is None:
            raise ImageProcessingError(f"Failed to decode image from path: '{image_source}'")
    elif isinstance(image_source, (bytes, io.BytesIO)):
        validate_image_file(image_source)
        data = image_source if isinstance(image_source, bytes) else image_source.getvalue()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback attempt via PIL
            try:
                pil_img = Image.open(io.BytesIO(data)).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except (UnidentifiedImageError, OSError) as exc:
                raise ImageProcessingError("Failed to decode uploaded image data.") from exc
    else:
        raise ImageProcessingError(f"Unsupported image source type: {type(image_source)}")

    # Validate resulting array
    if img is None or img.size == 0:
        raise ImageProcessingError("Decoded image is empty.")

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif len(img.shape) != 3 or img.shape[2] != 3:
        raise ImageProcessingError(f"Expected 3-channel image, got shape {img.shape}")

    h, w = img.shape[:2]
    if h < 32 or w < 32:
        raise ImageProcessingError(f"Image dimensions too small ({w}x{h}). Minimum size is 32x32 pixels.")

    return img


def bgr_to_rgb(bgr_image: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB."""
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(rgb_image: np.ndarray) -> np.ndarray:
    """Convert RGB image to OpenCV BGR."""
    return cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)


def to_mediapipe_image(image_bgr: np.ndarray) -> mp.Image:
    """Convert an OpenCV BGR image array to a MediaPipe mp.Image."""
    rgb = bgr_to_rgb(image_bgr)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
