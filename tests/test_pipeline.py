"""
Comprehensive test suite for ModelSentinel:
- X-LSB attack and payload roundtrip
- Grayscale-Fourpart (GF) representation
- Safe weight I/O and reconstruction
- Statistical baseline detectors
- Few-Shot CNN inference
- Ensemble decision engine
- FastAPI endpoints
"""

import os
import tempfile
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.main import app
from src.attack.xlsb_attack import EICAR_PAYLOAD, embed_payload, embedding_rate, extract_payload
from src.detectors.ensemble import compute_ensemble_verdict
from src.detectors.fewshot_cnn import OSLCNN, classify
from src.detectors.layer_analysis import covert_channel_static_analysis, layer_correlation_analysis
from src.detectors.statistical import (
    autocorrelation_score,
    byte_entropy_score,
    calibrate,
    kl_divergence_score,
    weight_distribution_score,
)
from src.representation.grayscale_fourpart import extract_byte_planes, resize_for_cnn, to_gf_image
from src.utils.weight_io import (
    inspect_serialization_risk,
    iter_float32_layers,
    load_flat_weights,
    load_state_dict_safely,
    save_flat_weights_as_state_dict,
    summarize_state_dict,
)


def test_embedding_rate():
    assert embedding_rate(1) == 1 / 32.0
    assert embedding_rate(4) == 4 / 32.0
    assert embedding_rate(8) == 8 / 32.0
    assert embedding_rate(23) == 23 / 32.0

    with pytest.raises(ValueError):
        embedding_rate(0)
    with pytest.raises(ValueError):
        embedding_rate(24)


def test_attack_and_extraction():
    rng = np.random.RandomState(42)
    weights = rng.randn(1000).astype(np.float32)
    custom_payload = b"TEST_PAYLOAD_12345"

    for x in [1, 2, 4, 8, 16, 23]:
        attacked = embed_payload(weights, X=x, payload=custom_payload)
        assert attacked.shape == weights.shape
        assert attacked.dtype == np.float32

        if x <= 4:
            np.testing.assert_allclose(attacked, weights, atol=1e-3)

        extracted = extract_payload(attacked, X=x, num_bytes=len(custom_payload))
        assert extracted == custom_payload


def test_eicar_default_payload():
    weights = np.random.randn(500).astype(np.float32)
    attacked = embed_payload(weights, X=8)
    extracted = extract_payload(attacked, X=8, num_bytes=len(EICAR_PAYLOAD))
    assert extracted == EICAR_PAYLOAD


def test_gf_image_representation():
    n = 10000
    weights = np.random.randn(n).astype(np.float32)

    b0, b1, b2, b3 = extract_byte_planes(weights)
    assert len(b0) == n
    assert len(b1) == n
    assert len(b2) == n
    assert len(b3) == n
    assert b0.dtype == np.uint8

    gf_img = to_gf_image(weights)
    s = int(np.ceil(np.sqrt(n)))
    assert gf_img.shape == (2 * s, 2 * s)
    assert gf_img.dtype == np.uint8

    cnn_img = resize_for_cnn(gf_img, size=64)
    assert cnn_img.shape == (64, 64)
    assert cnn_img.dtype == np.float32
    assert cnn_img.min() >= 0.0
    assert cnn_img.max() <= 1.0


def test_weight_io_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = os.path.join(tmpdir, "model.pt")
        out_path = os.path.join(tmpdir, "attacked.pt")

        orig_dict = {
            "layer2.weight": torch.randn(4, 4, dtype=torch.float32),
            "layer1.weight": torch.randn(2, 4, dtype=torch.float32),
            "layer1.bias": torch.randn(2, dtype=torch.float32),
        }
        torch.save(orig_dict, ref_path)

        flat = load_flat_weights(ref_path)
        expected_len = 2 + 8 + 16
        assert len(flat) == expected_len

        attacked_flat = embed_payload(flat, X=4)
        save_flat_weights_as_state_dict(ref_path, attacked_flat, out_path)

        loaded_attacked_dict = load_state_dict_safely(out_path)
        assert set(loaded_attacked_dict.keys()) == set(orig_dict.keys())
        assert loaded_attacked_dict["layer1.weight"].shape == (2, 4)

        loaded_flat = load_flat_weights(out_path)
        np.testing.assert_array_equal(loaded_flat, attacked_flat)



def test_model_intake_metadata_and_serialization_risk():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        torch.save({
            "layer.weight": torch.randn(3, 4, dtype=torch.float32),
            "layer.bias": torch.randn(3, dtype=torch.float32),
        }, path)

        state = load_state_dict_safely(path)
        metadata = summarize_state_dict(path, state)
        assert metadata["tensor_count"] == 2
        assert metadata["num_parameters"] == 15
        assert metadata["dtype_counts"]["torch.float32"] == 2
        assert metadata["tensors"][0]["shape"] in ([3], [3, 4])
        assert "file_size_sanity" in metadata

        risk = inspect_serialization_risk(path)
        assert risk["pickle_based_format"] is True
        assert risk["safe_loader"] == "torch.load(weights_only=True)"


def test_layer_and_covert_channel_analysis():
    layers = [
        ("encoder.weight", np.random.randn(2048).astype(np.float32)),
        ("secret_trigger.weight", np.zeros(2048, dtype=np.float32)),
    ]
    layer_score, layer_details = layer_correlation_analysis(layers)
    covert_score, covert_details = covert_channel_static_analysis(layers)

    assert layer_details["layer_count"] == 2
    assert layer_score >= 0.0
    assert covert_score > 0.0
    assert covert_details["findings"][0]["name_signal"] is True

def test_statistical_calibration_and_scoring():
    benign_samples = [np.random.randn(20000).astype(np.float32) for _ in range(3)]
    with tempfile.TemporaryDirectory() as tmpdir:
        calib_path = os.path.join(tmpdir, "calib.json")
        calib = calibrate(benign_samples, save_path=calib_path)
        assert "entropy_mean" in calib
        assert "b3_ref_histogram" in calib

        # Benign sample should yield low anomaly score
        score_ent, _ = byte_entropy_score(benign_samples[0], calib)
        assert score_ent < 0.6

        # Heavily attacked sample should yield elevated anomaly score
        att = embed_payload(benign_samples[0], X=8)
        score_ent_att, _ = byte_entropy_score(att, calib)
        score_kl_att, _ = kl_divergence_score(att, calib)
        assert score_ent_att > score_ent
        assert score_kl_att > 0.5


def test_fewshot_cnn_forward():
    model = OSLCNN(embedding_dim=64)
    dummy_input = torch.randn(2, 1, 64, 64)
    emb = model(dummy_input)
    assert emb.shape == (2, 64)
    # L2 normalized embeddings should have norm 1.0
    norms = torch.norm(emb, p=2, dim=1)
    np.testing.assert_allclose(norms.detach().numpy(), [1.0, 1.0], atol=1e-5)


def test_ensemble_verdict():
    ent_clean = (0.1, {})
    ac_clean = (0.1, {})
    kl_clean = (0.1, {})
    wd_clean = (0.1, {})
    cnn_clean = {"label": "Clean", "confidence": 0.95, "prob_malicious": 0.05, "dist_benign": 0.1, "dist_malicious": 1.2}

    verdict_clean = compute_ensemble_verdict(ent_clean, ac_clean, kl_clean, wd_clean, cnn_clean)
    assert verdict_clean["tier"] == "Clean"
    assert verdict_clean["risk_score"] < 25.0

    ent_mal = (0.95, {})
    ac_mal = (0.9, {})
    kl_mal = (0.98, {})
    wd_mal = (0.2, {})
    cnn_mal = {"label": "Malicious", "confidence": 0.98, "prob_malicious": 0.98, "dist_benign": 1.4, "dist_malicious": 0.1, "estimated_severity": 8}

    verdict_mal = compute_ensemble_verdict(ent_mal, ac_mal, kl_mal, wd_mal, cnn_mal)
    assert verdict_mal["tier"] == "Malicious"
    assert verdict_mal["risk_score"] > 75.0
    assert verdict_mal["estimated_severity_x"] == 8


def test_fastapi_endpoints():
    client = TestClient(app)
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["service"] == "ModelSentinel"

    res_models = client.get("/models")
    assert res_models.status_code == 200
