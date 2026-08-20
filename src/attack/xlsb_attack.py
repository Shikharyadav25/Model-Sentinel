"""
X-LSB-Attack-Fill: Synthetic payload embedding via float32 mantissa LSB steganography.
For simulation and benchmark testing only.
"""

from typing import Union
import numpy as np

# Standard EICAR Antivirus Test File string (harmless synthetic marker)
EICAR_PAYLOAD: bytes = (
    b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
)


def embedding_rate(X: int) -> float:
    """
    Returns embedding rate ER = X / 32 for float32 weights.
    """
    if not (1 <= X <= 23):
        raise ValueError(f"X must be between 1 and 23 inclusive, got {X}")
    return X / 32.0


def embed_payload(
    weights: np.ndarray,
    X: int,
    payload: Union[bytes, str] = EICAR_PAYLOAD,
) -> np.ndarray:
    """
    X-LSB-Attack-Fill: Deterministically overwrites the lowest X bits of each
    float32 weight's 23-bit mantissa with payload bits.

    Args:
        weights: 1D numpy array of float32 weights (length n).
        X: Number of mantissa LSBs to overwrite (1 <= X <= 23).
        payload: Bytes or string to embed. Tiled or truncated to fit n * X bits.

    Returns:
        New float32 numpy array with embedded payload.
    """
    if not (1 <= X <= 23):
        raise ValueError(f"X must be between 1 and 23 for float32, got {X}")

    if isinstance(payload, str):
        payload = payload.encode("utf-8")

    if len(payload) == 0:
        payload = EICAR_PAYLOAD

    flat_weights = np.ascontiguousarray(weights, dtype=np.float32).ravel()
    n = len(flat_weights)
    if n == 0:
        return flat_weights.copy()

    total_bits_needed = n * X
    payload_bytes_needed = int(np.ceil(total_bits_needed / 8))

    # Tile payload bytes to exceed or match the total bits needed
    repeats = int(np.ceil(payload_bytes_needed / len(payload)))
    full_payload = (payload * repeats)[:payload_bytes_needed]

    payload_arr = np.frombuffer(full_payload, dtype=np.uint8)
    all_bits = np.unpackbits(payload_arr)[:total_bits_needed]

    # Reshape bits into (n, X)
    bits_matrix = all_bits.reshape(n, X)

    # Convert each row of X bits into a uint32 integer
    # Bit 0 is MSB of the X-bit chunk, Bit X-1 is LSB
    powers = (1 << np.arange(X - 1, -1, -1, dtype=np.uint32))
    payload_values = np.dot(bits_matrix.astype(np.uint32), powers)

    # View float32 as uint32
    w_uint32 = flat_weights.view(np.uint32).copy()

    # Clear lowest X bits and insert payload bits (safe 32-bit bitwise mask)
    mask = np.uint32(0xFFFFFFFF ^ ((1 << X) - 1))
    attacked_uint32 = (w_uint32 & mask) | payload_values

    return attacked_uint32.view(np.float32)


def extract_payload(weights: np.ndarray, X: int, num_bytes: int) -> bytes:
    """
    Extracts embedded payload bytes from the lowest X bits of float32 weights.

    Args:
        weights: 1D numpy array of float32 weights.
        X: Number of mantissa bits used during embedding.
        num_bytes: Number of bytes to reconstruct.

    Returns:
        Extracted bytes.
    """
    if not (1 <= X <= 23):
        raise ValueError(f"X must be between 1 and 23, got {X}")

    flat_weights = np.ascontiguousarray(weights, dtype=np.float32).ravel()
    w_uint32 = flat_weights.view(np.uint32)

    total_bits_needed = num_bytes * 8
    weights_needed = int(np.ceil(total_bits_needed / X))
    selected = w_uint32[:weights_needed]

    # Extract lowest X bits for each weight
    mask = np.uint32((1 << X) - 1)
    extracted_vals = selected & mask

    # Unpack into bits
    shifts = np.arange(X - 1, -1, -1, dtype=np.uint32)
    bits = ((extracted_vals[:, None] >> shifts) & 1).astype(np.uint8).ravel()
    bits = bits[:total_bits_needed]

    # Pack bits back into bytes
    payload_bytes = np.packbits(bits)
    return payload_bytes.tobytes()
