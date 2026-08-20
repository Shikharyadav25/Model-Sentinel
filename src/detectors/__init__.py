"""Detectors package for ModelSentinel."""

from .ensemble import compute_ensemble_verdict
from .fewshot_cnn import OSLCNN, classify, load_fewshot_cnn, train_fewshot_cnn
from .statistical import (
    autocorrelation_score,
    byte_entropy_score,
    calibrate,
    kl_divergence_score,
    load_calibration,
    weight_distribution_score,
)

__all__ = [
    "byte_entropy_score",
    "autocorrelation_score",
    "kl_divergence_score",
    "weight_distribution_score",
    "calibrate",
    "load_calibration",
    "OSLCNN",
    "train_fewshot_cnn",
    "load_fewshot_cnn",
    "classify",
    "compute_ensemble_verdict",
]
