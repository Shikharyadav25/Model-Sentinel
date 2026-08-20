"""
Weight I/O utilities for safe loading, deterministic flattening,
and structured state_dict reconstruction.
"""

import os
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def load_state_dict_safely(path: str) -> Dict[str, torch.Tensor]:
    """
    Safely loads a state_dict from .pt, .pth, or .safetensors file.
    Always uses weights_only=True for PyTorch pickle files to prevent code execution.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".safetensors":
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors package is required to load .safetensors files.")
        return safetensors.torch.load_file(path, device="cpu")

    elif ext in [".pt", ".pth", ".bin"]:
        # Safe deserialization preventing arbitrary code execution
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        if isinstance(checkpoint, torch.Tensor):
            return {"weights": checkpoint}
        elif isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                return checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                return checkpoint["model"]
            else:
                # Direct state dict mapping
                return {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
        elif hasattr(checkpoint, "state_dict"):
            return checkpoint.state_dict()
        else:
            raise ValueError(f"Unrecognized checkpoint structure in {path}")
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported: .pt, .pth, .bin, .safetensors")


def load_flat_weights(path: str) -> np.ndarray:
    """
    Loads weights from model file, sorts parameter keys alphabetically,
    validates float32 dtype, and returns a single concatenated 1D float32 array.

    Args:
        path: Path to model weights file.

    Returns:
        1D float32 numpy array.
    """
    state_dict = load_state_dict_safely(path)
    if not state_dict:
        raise ValueError(f"No valid weight tensors found in {path}")

    sorted_keys = sorted(state_dict.keys())
    flat_pieces: List[np.ndarray] = []

    for k in sorted_keys:
        tensor = state_dict[k]
        if not isinstance(tensor, torch.Tensor):
            continue

        if tensor.dtype != torch.float32:
            raise TypeError(
                f"ModelSentinel requires float32 weights for LSB steganography analysis. "
                f"Found tensor '{k}' with non-float32 dtype: {tensor.dtype}"
            )

        arr = tensor.detach().cpu().numpy().ravel()
        flat_pieces.append(arr)

    if not flat_pieces:
        raise ValueError(f"No float32 weight tensors found in {path}")

    return np.concatenate(flat_pieces).astype(np.float32)


def save_flat_weights_as_state_dict(reference_path: str, flat: np.ndarray, out_path: str) -> None:
    """
    Reconstructs an attacked model state_dict matching reference_path structure
    using the provided 1D flat array, and saves it to out_path.

    Args:
        reference_path: Path to the clean reference model.
        flat: 1D flat float32 array of modified weights.
        out_path: Destination path (.pt, .pth, or .safetensors).
    """
    ref_dict = load_state_dict_safely(reference_path)
    sorted_keys = sorted(ref_dict.keys())

    new_dict: Dict[str, torch.Tensor] = {}
    current_idx = 0
    flat_len = len(flat)

    for k in sorted_keys:
        orig_tensor = ref_dict[k]
        if not isinstance(orig_tensor, torch.Tensor):
            new_dict[k] = orig_tensor
            continue

        num_elements = orig_tensor.numel()
        if current_idx + num_elements > flat_len:
            raise ValueError(
                f"Flat weight array exhausted at key '{k}'. Expected at least "
                f"{current_idx + num_elements} elements, got {flat_len}."
            )

        chunk = flat[current_idx : current_idx + num_elements]
        new_tensor = torch.from_numpy(chunk.reshape(orig_tensor.shape)).to(
            dtype=orig_tensor.dtype, device="cpu"
        )
        new_dict[k] = new_tensor
        current_idx += num_elements

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_ext = os.path.splitext(out_path)[1].lower()

    if out_ext == ".safetensors":
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors package required to save .safetensors format")
        safetensors.torch.save_file(new_dict, out_path)
    else:
        torch.save(new_dict, out_path)
