"""
Scan pipeline for ModelSentinel.
End-to-end execution:
Model file -> Safe Load -> GF Image -> Statistical Ensemble + CNN -> Ensemble Verdict.
"""

import base64
import io
import os
import time
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image

from src.detectors.ensemble import compute_ensemble_verdict
from src.detectors.layer_analysis import covert_channel_static_analysis, layer_correlation_analysis
from src.detectors.fewshot_cnn import classify, load_fewshot_cnn
from src.detectors.statistical import (
    autocorrelation_score,
    byte_entropy_score,
    kl_divergence_score,
    load_calibration,
    weight_distribution_score,
)
from src.representation.grayscale_fourpart import resize_for_cnn, to_gf_image
from src.utils.weight_io import inspect_serialization_risk, iter_float32_layers, load_flat_weights, summarize_state_dict

# Cached models for sub-second repeat scanning
_CACHED_CNN = None
_CACHED_REF = None
_CACHED_CALIB = None


def get_detectors(calib_path: str = "models/calibration.json", cnn_path: str = "models/fewshot_cnn.pt"):
    global _CACHED_CNN, _CACHED_REF, _CACHED_CALIB
    if _CACHED_CALIB is None:
        _CACHED_CALIB = load_calibration(calib_path)
    if _CACHED_CNN is None or _CACHED_REF is None:
        _CACHED_CNN, _CACHED_REF = load_fewshot_cnn(cnn_path)
    return _CACHED_CALIB, _CACHED_CNN, _CACHED_REF


def image_to_base64_png(image_arr: np.ndarray, max_side: int = 512) -> str:
    """Encodes a 2D uint8 numpy array to a base64 PNG string for web rendering."""
    if image_arr.dtype != np.uint8:
        image_arr = np.clip(image_arr, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(image_arr)
    # Downscale if excessively large to keep payload lightweight
    if max(pil_img.size) > max_side:
        pil_img.thumbnail((max_side, max_side), Image.Resampling.NEAREST)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def scan_model(
    path: str,
    calib_path: str = "models/calibration.json",
    cnn_path: str = "models/fewshot_cnn.pt",
    cnn_mode: str = "centroid",
) -> Dict[str, Any]:
    """
    Scans a PyTorch (.pt/.pth) or .safetensors model file for steganographic malware.

    Args:
        path: Path to model file.
        calib_path: Path to statistical calibration JSON.
        cnn_path: Path to trained Few-Shot CNN checkpoint.
        cnn_mode: "centroid" or "1-nn".

    Returns:
        JSON-serializable scan results dictionary.
    """
    start_time = time.time()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    file_size_bytes = os.path.getsize(path)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    filename = os.path.basename(path)

    # 1. Safe weight loading and model intake metadata
    t0 = time.time()
    serialization_risk = inspect_serialization_risk(path)
    intake_metadata = summarize_state_dict(path)
    layers = iter_float32_layers(path)
    flat_weights = load_flat_weights(path)
    load_time = round(time.time() - t0, 3)
    num_params = len(flat_weights)

    # 2. Grayscale-Fourpart (GF) representation
    t0 = time.time()
    gf_image = to_gf_image(flat_weights)
    cnn_input = resize_for_cnn(gf_image, size=64)
    gf_time = round(time.time() - t0, 3)

    # 3. Statistical Baseline Ensemble (4 tests)
    calib, cnn_model, cnn_ref = get_detectors(calib_path, cnn_path)

    t0 = time.time()
    ent_res = byte_entropy_score(flat_weights, calib)
    ac_res = autocorrelation_score(flat_weights, calib)
    kl_res = kl_divergence_score(flat_weights, calib)
    wd_res = weight_distribution_score(flat_weights, calib)
    stat_time = round(time.time() - t0, 3)

    # 4. Layer-aware and static backdoor/covert-channel heuristics
    t0 = time.time()
    layer_res = layer_correlation_analysis(layers)
    covert_res = covert_channel_static_analysis(layers)
    layer_time = round(time.time() - t0, 3)

    # 5. Few-Shot CNN Inference
    t0 = time.time()
    cnn_res = classify(cnn_model, cnn_input, cnn_ref, mode=cnn_mode)
    cnn_time = round(time.time() - t0, 3)

    # 6. Ensemble synthesis
    verdict = compute_ensemble_verdict(
        ent_res, ac_res, kl_res, wd_res, cnn_res, layer_res, covert_res, serialization_risk
    )

    total_time = round(time.time() - start_time, 3)

    # Convert GF image to base64 for UI rendering
    gf_b64 = image_to_base64_png(gf_image)

    return {
        "filename": filename,
        "filepath": path,
        "file_size_mb": file_size_mb,
        "num_parameters": num_params,
        "gf_dimensions": list(gf_image.shape),
        "metadata": intake_metadata,
        "serialization_risk": serialization_risk,
        "timing": {
            "load_time_sec": load_time,
            "gf_transform_sec": gf_time,
            "statistical_sec": stat_time,
            "layer_static_sec": layer_time,
            "cnn_sec": cnn_time,
            "total_scan_sec": total_time,
        },
        "verdict": verdict,
        "gf_image_base64": gf_b64,
    }
