"""
Ensemble decision engine for ModelSentinel.
Combines 4 statistical anomaly tests and the Few-Shot CNN detector
into a unified risk score (0-100), verdict tier, detector breakdown,
and estimated embedding severity.
"""

from typing import Any, Dict, Optional
from src.attack.xlsb_attack import embedding_rate


def compute_ensemble_verdict(
    entropy_res: tuple,
    autocorr_res: tuple,
    kl_res: tuple,
    weight_dist_res: tuple,
    cnn_res: Dict[str, Any],
    layer_res: tuple | None = None,
    covert_res: tuple | None = None,
    serialization_res: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Synthesizes detector outputs into an overall risk score and verdict.

    Args:
        entropy_res: (score, details) from byte_entropy_score
        autocorr_res: (score, details) from autocorrelation_score
        kl_res: (score, details) from kl_divergence_score
        weight_dist_res: (score, details) from weight_distribution_score
        cnn_res: dict from fewshot_cnn.classify

    Returns:
        Structured verdict dictionary with risk score, tier, and breakdowns.
    """
    s_ent, d_ent = entropy_res
    s_ac, d_ac = autocorr_res
    s_kl, d_kl = kl_res
    s_wd, d_wd = weight_dist_res

    s_cnn = float(cnn_res.get("prob_malicious", 0.0))
    cnn_label = cnn_res.get("label", "Clean")
    cnn_conf = float(cnn_res.get("confidence", 0.5))
    est_severity = cnn_res.get("estimated_severity")

    s_layer, d_layer = layer_res if layer_res is not None else (0.0, {})
    s_covert, d_covert = covert_res if covert_res is not None else (0.0, {})
    s_serial = 0.0
    if serialization_res is not None:
        risk_level = serialization_res.get("risk_level", "low")
        s_serial = {"low": 0.05, "medium": 0.25, "high": 0.75}.get(risk_level, 0.0)

    # Statistical composite score (weighted sum)
    stat_score = float(
        0.35 * s_ent +
        0.35 * s_kl +
        0.20 * s_ac +
        0.10 * s_wd
    )

    # Combined composite (50% statistical, 50% CNN)
    combined_score = 0.50 * stat_score + 0.50 * s_cnn

    # Peak signal sensitivity: if any strong detector signals alert, reflect in risk
    peak_signal = max(stat_score, s_cnn, s_ent, s_kl, s_layer, s_covert, s_serial)
    risk_score_raw = max(combined_score, 0.75 * peak_signal) * 100.0
    risk_score = float(min(100.0, max(0.0, round(risk_score_raw, 1))))

    # Verdict tier
    if risk_score < 25.0:
        tier = "Clean"
        tier_color = "#10B981"  # Emerald Green
        summary = "No anomalous LSB steganography patterns detected. Model weights appear pristine."
    elif risk_score < 50.0:
        tier = "Low-Risk"
        tier_color = "#FBBF24"  # Amber Yellow
        summary = "Minor statistical variance detected, but well within typical benign model tolerance."
    elif risk_score < 75.0:
        tier = "Suspicious"
        tier_color = "#F97316"  # Orange
        summary = "Elevated LSB entropy and structural anomalies detected. Potential low-bit payload present."
    else:
        tier = "Malicious"
        tier_color = "#EF4444"  # Red
        summary = "High-confidence steganographic payload detected in float32 mantissa LSBs."

    # Estimated embedding rate
    est_er_percent = None
    if est_severity is not None:
        try:
            est_er_percent = round(embedding_rate(est_severity) * 100.0, 2)
        except Exception:
            est_er_percent = None

    breakdown = {
        "byte_entropy": {
            "name": "Byte-Plane Entropy (B0-B3)",
            "score": round(s_ent, 4),
            "percentage": round(s_ent * 100.0, 1),
            "fired": s_ent >= 0.5,
            "details": d_ent,
        },
        "autocorrelation": {
            "name": "B3 Lag-1 Autocorrelation",
            "score": round(s_ac, 4),
            "percentage": round(s_ac * 100.0, 1),
            "fired": s_ac >= 0.5,
            "details": d_ac,
        },
        "kl_divergence": {
            "name": "B3 Histogram KL-Divergence",
            "score": round(s_kl, 4),
            "percentage": round(s_kl * 100.0, 1),
            "fired": s_kl >= 0.5,
            "details": d_kl,
        },
        "weight_distribution": {
            "name": "Float32 Value Distribution",
            "score": round(s_wd, 4),
            "percentage": round(s_wd * 100.0, 1),
            "fired": s_wd >= 0.5,
            "details": d_wd,
        },
        "layer_correlation": {
            "name": "Layer Entropy & Cross-Layer Correlation",
            "score": round(s_layer, 4),
            "percentage": round(s_layer * 100.0, 1),
            "fired": s_layer >= 0.5,
            "details": d_layer,
        },
        "covert_channel_static": {
            "name": "Static Backdoor/Covert-Channel Indicators",
            "score": round(s_covert, 4),
            "percentage": round(s_covert * 100.0, 1),
            "fired": s_covert >= 0.5,
            "details": d_covert,
        },
        "serialization_safety": {
            "name": "Serialization Malware Surface",
            "score": round(s_serial, 4),
            "percentage": round(s_serial * 100.0, 1),
            "fired": s_serial >= 0.5,
            "details": serialization_res or {},
        },
        "fewshot_cnn": {
            "name": "Few-Shot Siamese CNN",
            "score": round(s_cnn, 4),
            "percentage": round(s_cnn * 100.0, 1),
            "fired": s_cnn >= 0.5,
            "label": cnn_label,
            "confidence": round(cnn_conf, 4),
            "details": {
                "dist_benign": round(cnn_res.get("dist_benign", 0.0), 4),
                "dist_malicious": round(cnn_res.get("dist_malicious", 0.0), 4),
            },
        },
    }

    return {
        "risk_score": risk_score,
        "tier": tier,
        "tier_color": tier_color,
        "summary": summary,
        "statistical_composite_score": round(stat_score, 4),
        "cnn_score": round(s_cnn, 4),
        "estimated_severity_x": est_severity if (tier in ["Suspicious", "Malicious"]) else None,
        "estimated_embedding_rate_percent": est_er_percent if (tier in ["Suspicious", "Malicious"]) else None,
        "breakdown": breakdown,
    }
