"""
Weight I/O utilities for safe loading, deterministic flattening,
and structured state_dict reconstruction.
"""

import os
import zipfile
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

SUPPORTED_EXTENSIONS = (".pt", ".pth", ".bin", ".safetensors")
MAX_REASONABLE_BYTES_PER_PARAM = 32.0
PICKLE_SIGNATURES = (b"GLOBAL", b"REDUCE", b"BUILD", b"INST", b"OBJ")


def inspect_serialization_risk(path: str) -> Dict[str, Any]:
    """Returns static file-format risk signals before tensor deserialization."""
    ext = os.path.splitext(path)[1].lower()
    size = os.path.getsize(path)
    pickle_like = ext in {".pt", ".pth", ".bin"}
    details: Dict[str, Any] = {
        "extension": ext,
        "supported_format": ext in SUPPORTED_EXTENSIONS,
        "file_size_bytes": size,
        "pickle_based_format": pickle_like,
        "safe_loader": "torch.load(weights_only=True)" if pickle_like else "safetensors.torch.load_file",
        "risk_level": "low",
        "findings": [],
    }

    if ext not in SUPPORTED_EXTENSIONS:
        details["risk_level"] = "high"
        details["findings"].append("unsupported model serialization extension")
        return details

    if ext == ".safetensors":
        details["findings"].append("safetensors format avoids pickle execution semantics")
        return details

    details["risk_level"] = "medium"
    details["findings"].append("PyTorch checkpoint formats are pickle-based; loading is restricted to weights_only=True")

    sample = b""
    try:
        if zipfile.is_zipfile(path):
            details["container"] = "zip"
            with zipfile.ZipFile(path) as zf:
                data_members = [n for n in zf.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
                details["pickle_members"] = data_members
                if data_members:
                    sample = zf.read(data_members[0])[:8192]
        else:
            details["container"] = "pickle_stream"
            with open(path, "rb") as f:
                sample = f.read(8192)
    except Exception as exc:
        details["risk_level"] = "high"
        details["findings"].append(f"could not statically inspect pickle payload: {exc}")
        return details

    hits = sorted({sig.decode("ascii") for sig in PICKLE_SIGNATURES if sig in sample})
    details["pickle_opcode_hints"] = hits
    if hits:
        details["findings"].append("pickle opcodes capable of object reconstruction were observed")
    return details


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
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        if isinstance(checkpoint, torch.Tensor):
            return {"weights": checkpoint}
        elif isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                return checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                return checkpoint["model"]
            else:
                return {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
        elif hasattr(checkpoint, "state_dict"):
            return checkpoint.state_dict()
        else:
            raise ValueError(f"Unrecognized checkpoint structure in {path}")
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported: .pt, .pth, .bin, .safetensors")


def summarize_state_dict(path: str, state_dict: Dict[str, torch.Tensor] | None = None) -> Dict[str, Any]:
    """Builds model intake metadata: tensor names, shapes, dtypes, and file-size sanity checks."""
    if state_dict is None:
        state_dict = load_state_dict_safely(path)
    file_size_bytes = os.path.getsize(path)
    tensors = []
    total_params = 0
    total_tensor_bytes = 0
    dtype_counts: Dict[str, int] = {}

    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            continue
        numel = int(tensor.numel())
        elem_size = int(tensor.element_size())
        dtype = str(tensor.dtype)
        total_params += numel
        total_tensor_bytes += numel * elem_size
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        tensors.append({
            "name": name,
            "shape": list(tensor.shape),
            "dtype": dtype,
            "num_parameters": numel,
            "bytes": numel * elem_size,
        })

    bytes_per_param = float(file_size_bytes / total_params) if total_params else 0.0
    warnings = []
    if not tensors:
        warnings.append("no tensor entries found")
    if bytes_per_param > MAX_REASONABLE_BYTES_PER_PARAM:
        warnings.append("file is unusually large relative to tensor parameter count")
    if total_tensor_bytes and file_size_bytes < total_tensor_bytes * 0.75:
        warnings.append("file is smaller than expected for raw tensor payload; verify compression/container metadata")

    return {
        "format": os.path.splitext(path)[1].lower(),
        "file_size_bytes": file_size_bytes,
        "tensor_count": len(tensors),
        "num_parameters": total_params,
        "tensor_bytes": total_tensor_bytes,
        "bytes_per_parameter": round(bytes_per_param, 3),
        "dtype_counts": dtype_counts,
        "tensors": tensors,
        "file_size_sanity": {"passed": len(warnings) == 0, "warnings": warnings},
    }


def iter_float32_layers(path: str) -> List[Tuple[str, np.ndarray]]:
    """Returns sorted float32 tensors as named flattened arrays for layer-level analysis."""
    state_dict = load_state_dict_safely(path)
    layers: List[Tuple[str, np.ndarray]] = []
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32:
            layers.append((name, tensor.detach().cpu().numpy().ravel().astype(np.float32)))
    return layers


def load_flat_weights(path: str) -> np.ndarray:
    """Loads sorted float32 weights and returns a single concatenated 1D float32 array."""
    state_dict = load_state_dict_safely(path)
    if not state_dict:
        raise ValueError(f"No valid weight tensors found in {path}")

    flat_pieces: List[np.ndarray] = []
    for k in sorted(state_dict.keys()):
        tensor = state_dict[k]
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dtype != torch.float32:
            raise TypeError(
                f"ModelSentinel requires float32 weights for LSB steganography analysis. "
                f"Found tensor '{k}' with non-float32 dtype: {tensor.dtype}"
            )
        flat_pieces.append(tensor.detach().cpu().numpy().ravel())

    if not flat_pieces:
        raise ValueError(f"No float32 weight tensors found in {path}")

    return np.concatenate(flat_pieces).astype(np.float32)


def save_flat_weights_as_state_dict(reference_path: str, flat: np.ndarray, out_path: str) -> None:
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
