"""
Statistical baseline ensemble for ModelSentinel.
Contains 4 lightweight, fast anomaly detectors based on:
1. Byte Entropy across all 4 byte-planes
2. Byte Autocorrelation (Lag-1 Pearson on B3)
3. Histogram KL-Divergence on B3 vs calibrated benign reference
4. Weight-Value Distribution statistics
"""

import json
import os
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.stats import kurtosis, skew

from src.representation.grayscale_fourpart import extract_byte_planes


def compute_byte_entropy(weights: np.ndarray) -> np.ndarray:
    """
    Computes Shannon entropy (base 2) for each of the 4 byte-planes (B0, B1, B2, B3).
    Returns a 4-dimensional vector of entropy values (0.0 to 8.0 bits).
    """
    planes = extract_byte_planes(weights)
    entropies = []
    for p in planes:
        counts = np.bincount(p, minlength=256)
        probs = counts / float(len(p))
        nonzero_probs = probs[probs > 0]
        ent = -np.sum(nonzero_probs * np.log2(nonzero_probs))
        entropies.append(float(ent))
    return np.array(entropies, dtype=np.float32)


def compute_byte_autocorrelation(weights: np.ndarray, max_samples: int = 500_000) -> float:
    """
    Computes lag-1 Pearson autocorrelation on B3 (lowest mantissa byte sequence).
    """
    _, _, _, b3 = extract_byte_planes(weights)
    if len(b3) > max_samples:
        # Subsample continuously for speed on huge models
        b3 = b3[:max_samples]

    x = b3.astype(np.float64)
    n = len(x)
    if n <= 1:
        return 0.0

    mean_x = np.mean(x)
    var_x = np.var(x)
    if var_x < 1e-10:
        return 0.0

    # Lag-1 correlation
    x0 = x[:-1] - mean_x
    x1 = x[1:] - mean_x
    autocorr = float(np.mean(x0 * x1) / var_x)
    return autocorr


def compute_b3_histogram(weights: np.ndarray, laplace_smooth: bool = True) -> np.ndarray:
    """
    Computes 256-bin normalized probability distribution of B3 byte values.
    """
    _, _, _, b3 = extract_byte_planes(weights)
    counts = np.bincount(b3, minlength=256).astype(np.float64)
    if laplace_smooth:
        counts += 1.0
    probs = counts / np.sum(counts)
    return probs


def compute_weight_distribution_stats(weights: np.ndarray, max_samples: int = 500_000) -> Dict[str, float]:
    """
    Computes raw float32 statistics: mean, std, skewness, kurtosis, min, max,
    and fraction of weights in [-0.01, 0.01].
    """
    flat = np.ascontiguousarray(weights, dtype=np.float32).ravel()
    if len(flat) > max_samples:
        flat = flat[:max_samples]

    w_mean = float(np.mean(flat))
    w_std = float(np.std(flat))
    w_min = float(np.min(flat))
    w_max = float(np.max(flat))
    w_skew = float(skew(flat)) if len(flat) > 2 else 0.0
    w_kurt = float(kurtosis(flat)) if len(flat) > 3 else 0.0
    near_zero = float(np.mean((flat >= -0.01) & (flat <= 0.01)))

    return {
        "mean": w_mean,
        "std": w_std,
        "min": w_min,
        "max": w_max,
        "skewness": w_skew,
        "kurtosis": w_kurt,
        "near_zero_fraction": near_zero,
    }


# ====================================================================
# Anomaly Scoring Functions (Given Calibration Reference)
# ====================================================================

def byte_entropy_score(weights: np.ndarray, calibration: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluates byte entropy against calibrated benign baseline.
    Returns (anomaly_score 0.0-1.0, details_dict).
    """
    ent = compute_byte_entropy(weights)
    ref_mean = np.array(calibration.get("entropy_mean", [6.0, 7.5, 7.8, 7.8]), dtype=np.float32)
    ref_std = np.array(calibration.get("entropy_std", [0.5, 0.2, 0.1, 0.1]), dtype=np.float32)

    # Z-scores across all 4 planes, emphasizing B3
    z_scores = np.abs(ent - ref_mean) / (ref_std + 1e-4)
    # B3 entropy under attack typically approaches theoretical maximum (~7.99 - 8.0 bits for EICAR/random)
    b3_dev = z_scores[3]
    overall_z = np.max(z_scores)

    # Sigmoid normalization
    score = float(1.0 / (1.0 + np.exp(-(overall_z - 3.0) / 1.5)))
    # Clip to [0, 1]
    score = float(np.clip(score, 0.0, 1.0))

    details = {
        "entropy_b0": float(ent[0]),
        "entropy_b1": float(ent[1]),
        "entropy_b2": float(ent[2]),
        "entropy_b3": float(ent[3]),
        "b3_z_score": float(b3_dev),
        "score": score,
    }
    return score, details


def autocorrelation_score(weights: np.ndarray, calibration: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluates B3 lag-1 autocorrelation against calibrated benign baseline.
    Returns (anomaly_score 0.0-1.0, details_dict).
    """
    ac = compute_byte_autocorrelation(weights)
    ref_mean = float(calibration.get("autocorr_mean", 0.05))
    ref_std = float(calibration.get("autocorr_std", 0.02))

    z_score = abs(ac - ref_mean) / (ref_std + 1e-4)
    # Sigmoid scaling
    score = float(1.0 / (1.0 + np.exp(-(z_score - 3.0) / 1.5)))
    score = float(np.clip(score, 0.0, 1.0))

    details = {
        "autocorrelation": ac,
        "ref_mean": ref_mean,
        "z_score": float(z_score),
        "score": score,
    }
    return score, details


def kl_divergence_score(weights: np.ndarray, calibration: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluates B3 histogram KL divergence D_KL(P || Q_ref).
    Returns (anomaly_score 0.0-1.0, details_dict).
    """
    p = compute_b3_histogram(weights, laplace_smooth=True)
    q_ref = np.array(
        calibration.get("b3_ref_histogram", [1.0 / 256.0] * 256),
        dtype=np.float64,
    )
    # Ensure q_ref is normalized
    q_ref = q_ref / np.sum(q_ref)

    # KL Divergence D_KL(P || Q)
    kl_div = float(np.sum(p * np.log(p / q_ref)))

    # Threshold calibration
    ref_kl_mean = float(calibration.get("kl_mean", 0.02))
    ref_kl_std = float(calibration.get("kl_std", 0.01))

    z_score = max(0.0, (kl_div - ref_kl_mean)) / (ref_kl_std + 1e-4)
    score = float(1.0 / (1.0 + np.exp(-(z_score - 3.0) / 1.5)))
    score = float(np.clip(score, 0.0, 1.0))

    details = {
        "kl_divergence": kl_div,
        "z_score": float(z_score),
        "score": score,
    }
    return score, details


def weight_distribution_score(weights: np.ndarray, calibration: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluates raw weight value distribution features against calibrated baseline.
    Returns (anomaly_score 0.0-1.0, details_dict).
    """
    stats = compute_weight_distribution_stats(weights)
    ref_stats = calibration.get("weight_stats", {})

    deviations = []
    for k, val in stats.items():
        if k in ref_stats:
            m = ref_stats[k]["mean"]
            s = ref_stats[k]["std"]
            z = abs(val - m) / (s + 1e-4)
            deviations.append(z)

    avg_z = float(np.mean(deviations)) if deviations else 0.0
    score = float(1.0 / (1.0 + np.exp(-(avg_z - 3.0) / 1.5)))
    score = float(np.clip(score, 0.0, 1.0))

    details = {
        **stats,
        "avg_z_score": avg_z,
        "score": score,
    }
    return score, details


# ====================================================================
# Calibration Engine
# ====================================================================

def calibrate(benign_flat_arrays: List[np.ndarray], save_path: str = "models/calibration.json") -> Dict[str, Any]:
    """
    Calibrates reference distributions and anomaly thresholds from a set
    of clean, known-benign model weights. Saves calibration artifact to JSON.

    Args:
        benign_flat_arrays: List of 1D float32 arrays from clean models.
        save_path: Destination path for calibration.json.

    Returns:
        Calibration dictionary.
    """
    if not benign_flat_arrays:
        raise ValueError("Cannot calibrate on empty benign weight list")

    entropies: List[np.ndarray] = []
    autocorrs: List[float] = []
    b3_histograms: List[np.ndarray] = []
    weight_stat_records: List[Dict[str, float]] = []

    for w in benign_flat_arrays:
        entropies.append(compute_byte_entropy(w))
        autocorrs.append(compute_byte_autocorrelation(w))
        b3_histograms.append(compute_b3_histogram(w, laplace_smooth=True))
        weight_stat_records.append(compute_weight_distribution_stats(w))

    # 1. Entropy stats (4-d)
    ent_arr = np.array(entropies)
    entropy_mean = np.mean(ent_arr, axis=0).tolist()
    entropy_std = np.maximum(np.std(ent_arr, axis=0), 0.01).tolist()

    # 2. Autocorrelation stats
    ac_arr = np.array(autocorrs)
    autocorr_mean = float(np.mean(ac_arr))
    autocorr_std = float(max(np.std(ac_arr), 0.005))

    # 3. Reference B3 Histogram & KL divergence stats
    ref_b3_hist = np.mean(b3_histograms, axis=0)
    ref_b3_hist = (ref_b3_hist / np.sum(ref_b3_hist)).tolist()

    # Measure benign KL variations
    benign_kls = []
    for h in b3_histograms:
        kl = float(np.sum(h * np.log(h / np.array(ref_b3_hist))))
        benign_kls.append(kl)
    kl_mean = float(np.mean(benign_kls))
    kl_std = float(max(np.std(benign_kls), 0.005))

    # 4. Weight stats
    weight_keys = ["mean", "std", "min", "max", "skewness", "kurtosis", "near_zero_fraction"]
    weight_stats_calib = {}
    for k in weight_keys:
        vals = [rec[k] for rec in weight_stat_records]
        weight_stats_calib[k] = {
            "mean": float(np.mean(vals)),
            "std": float(max(np.std(vals), 0.005)),
        }

    calibration = {
        "entropy_mean": entropy_mean,
        "entropy_std": entropy_std,
        "autocorr_mean": autocorr_mean,
        "autocorr_std": autocorr_std,
        "b3_ref_histogram": ref_b3_hist,
        "kl_mean": kl_mean,
        "kl_std": kl_std,
        "weight_stats": weight_stats_calib,
    }

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(calibration, f, indent=2)

    return calibration


def load_calibration(path: str = "models/calibration.json") -> Dict[str, Any]:
    """Loads calibration dictionary from JSON file."""
    if not os.path.exists(path):
        # Return sensible default if not yet trained
        return {
            "entropy_mean": [5.5, 7.4, 7.8, 7.8],
            "entropy_std": [0.3, 0.1, 0.05, 0.05],
            "autocorr_mean": 0.0,
            "autocorr_std": 0.02,
            "b3_ref_histogram": [1.0 / 256.0] * 256,
            "kl_mean": 0.02,
            "kl_std": 0.01,
            "weight_stats": {},
        }
    with open(path, "r") as f:
        return json.load(f)
