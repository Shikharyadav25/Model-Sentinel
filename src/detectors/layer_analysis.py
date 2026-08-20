"""Layer-aware static analyses for steganography, covert-channel, and backdoor indicators."""

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

from src.detectors.statistical import compute_b3_histogram, compute_byte_entropy
from src.representation.grayscale_fourpart import extract_byte_planes


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b), 100_000)
    if n < 3:
        return 0.0
    x = a[:n].astype(np.float64)
    y = b[:n].astype(np.float64)
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def layer_correlation_analysis(layers: Sequence[Tuple[str, np.ndarray]]) -> Tuple[float, Dict[str, Any]]:
    """Finds suspicious per-layer entropy outliers and unusual adjacent-layer B3 correlations."""
    records: List[Dict[str, Any]] = []
    for name, weights in layers:
        if len(weights) == 0:
            continue
        ent = compute_byte_entropy(weights)
        _, _, _, b3 = extract_byte_planes(weights)
        records.append({
            "name": name,
            "num_parameters": int(len(weights)),
            "entropy_b3": float(ent[3]),
            "b3_unique_fraction": float(len(np.unique(b3)) / 256.0),
        })

    if not records:
        return 0.0, {"layers": [], "correlations": [], "score": 0.0}

    entropies = np.array([r["entropy_b3"] for r in records], dtype=np.float64)
    median = float(np.median(entropies))
    mad = float(np.median(np.abs(entropies - median)) + 1e-4)
    for rec in records:
        rec["entropy_outlier_score"] = float(abs(rec["entropy_b3"] - median) / (1.4826 * mad))
        rec["suspicious"] = rec["entropy_outlier_score"] >= 3.5 or rec["entropy_b3"] >= 7.98

    correlations: List[Dict[str, Any]] = []
    for idx in range(len(layers) - 1):
        left_name, left_weights = layers[idx]
        right_name, right_weights = layers[idx + 1]
        _, _, _, left_b3 = extract_byte_planes(left_weights)
        _, _, _, right_b3 = extract_byte_planes(right_weights)
        corr = _safe_corr(left_b3, right_b3)
        correlations.append({
            "layer_a": left_name,
            "layer_b": right_name,
            "b3_correlation": corr,
            "suspicious": abs(corr) >= 0.25,
        })

    max_outlier = max((r["entropy_outlier_score"] for r in records), default=0.0)
    max_corr = max((abs(c["b3_correlation"]) for c in correlations), default=0.0)
    score = max(
        min(1.0, max_outlier / 8.0),
        min(1.0, max(0.0, max_corr - 0.15) / 0.35),
    )
    suspicious_layers = sorted(records, key=lambda r: r["entropy_outlier_score"], reverse=True)[:10]

    return float(score), {
        "score": float(score),
        "suspicious_layers": suspicious_layers,
        "correlations": correlations[:25],
        "layer_count": len(records),
    }


def covert_channel_static_analysis(layers: Sequence[Tuple[str, np.ndarray]]) -> Tuple[float, Dict[str, Any]]:
    """Flags static hidden-output/covert-channel indicators without executing untrusted code."""
    findings: List[Dict[str, Any]] = []
    for name, weights in layers:
        lower = name.lower()
        name_signal = any(token in lower for token in ("trigger", "backdoor", "secret", "payload", "covert"))
        if len(weights) == 0:
            continue
        hist = compute_b3_histogram(weights)
        dominant_byte_fraction = float(np.max(hist))
        repeated_low_byte = dominant_byte_fraction >= 0.20 and len(weights) >= 16
        if name_signal or repeated_low_byte:
            findings.append({
                "layer": name,
                "name_signal": name_signal,
                "dominant_low_byte_fraction": dominant_byte_fraction,
                "repeated_low_byte_signal": repeated_low_byte,
            })

    score = min(1.0, 0.35 * len(findings))
    return score, {
        "score": score,
        "findings": findings,
        "note": "Static heuristic only; trigger-based behavior requires an optional sandboxed model-execution harness.",
    }
