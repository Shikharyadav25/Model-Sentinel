"""
Grayscale-Fourpart (GF) Image Representation.
Transforms flat float32 model weights into a 2x2 tiled grayscale image
of 4 byte-planes (B0, B1, B2, B3).
"""

from typing import Tuple
import numpy as np
from PIL import Image


def extract_byte_planes(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the 4 byte-planes (B0, B1, B2, B3) from a 1D float32 weight array.

    In IEEE-754 single precision (32-bit):
    - B0: Sign + top 7 exponent bits (bits 24-31)
    - B1: Lowest 1 exponent bit + top 7 mantissa bits (bits 16-23)
    - B2: Middle mantissa bits (bits 8-15)
    - B3: Lowest 8 mantissa bits (bits 0-7, attack target)

    Returns:
        Tuple of (B0, B1, B2, B3) as uint8 1D arrays of length n.
    """
    flat = np.ascontiguousarray(weights, dtype=np.float32).ravel()
    w_u32 = flat.view(np.uint32)

    b0 = ((w_u32 >> 24) & 0xFF).astype(np.uint8)
    b1 = ((w_u32 >> 16) & 0xFF).astype(np.uint8)
    b2 = ((w_u32 >> 8) & 0xFF).astype(np.uint8)
    b3 = (w_u32 & 0xFF).astype(np.uint8)

    return b0, b1, b2, b3


def to_gf_image(weights: np.ndarray) -> np.ndarray:
    """
    Transforms flat float32 array into a 2x2 tiled grayscale uint8 image.

    Grid layout:
    +--------+--------+
    |   B0   |   B1   |
    +--------+--------+
    |   B2   |   B3   |
    +--------+--------+

    Args:
        weights: 1D numpy array of float32 weights.

    Returns:
        Square uint8 numpy array of shape (2S, 2S) where S = ceil(sqrt(n)).
    """
    flat = np.ascontiguousarray(weights, dtype=np.float32).ravel()
    n = len(flat)
    if n == 0:
        return np.zeros((2, 2), dtype=np.uint8)

    b0, b1, b2, b3 = extract_byte_planes(flat)

    s = int(np.ceil(np.sqrt(n)))
    target_len = s * s
    pad_len = target_len - n

    def pad_and_square(plane: np.ndarray) -> np.ndarray:
        if pad_len > 0:
            plane_padded = np.pad(plane, (0, pad_len), mode="constant", constant_values=0)
        else:
            plane_padded = plane
        return plane_padded.reshape((s, s))

    sq_b0 = pad_and_square(b0)
    sq_b1 = pad_and_square(b1)
    sq_b2 = pad_and_square(b2)
    sq_b3 = pad_and_square(b3)

    # Tile 2x2:
    # Row 0: [sq_b0, sq_b1]
    # Row 1: [sq_b2, sq_b3]
    top_row = np.hstack((sq_b0, sq_b1))
    bottom_row = np.hstack((sq_b2, sq_b3))
    gf_image = np.vstack((top_row, bottom_row))

    return gf_image


def resize_for_cnn(image: np.ndarray, size: int = 64) -> np.ndarray:
    """
    Resizes the GF grayscale image to (size, size) and normalizes pixel values to [0, 1] float32.

    Args:
        image: 2D uint8 numpy array (grayscale GF image).
        size: Target side length (default: 64).

    Returns:
        Normalized float32 numpy array of shape (size, size) with values in [0.0, 1.0].
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(image)
    resized_pil = pil_img.resize((size, size), Image.Resampling.BILINEAR)
    resized_arr = np.array(resized_pil, dtype=np.float32) / 255.0

    return resized_arr
