"""
Evaluation script for ModelSentinel.
Benchmarks detection accuracy across embedding rates (X = 1, 2, 4, 8, 16, 23)
on out-of-distribution held-out model architectures.
Produces publication-quality figure: results/accuracy_vs_er.png.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure root directory in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.attack.xlsb_attack import EICAR_PAYLOAD, embed_payload, embedding_rate
from src.data.build_dataset import ATTACK_SEVERITIES, HELDOUT_ARCHITECTURES, build_dataset
from src.detectors.ensemble import compute_ensemble_verdict
from src.detectors.fewshot_cnn import classify, load_fewshot_cnn
from src.detectors.statistical import (
    autocorrelation_score,
    byte_entropy_score,
    kl_divergence_score,
    load_calibration,
    weight_distribution_score,
)
from src.representation.grayscale_fourpart import resize_for_cnn, to_gf_image
from src.utils.weight_io import load_flat_weights


def evaluate_pipeline(data_dir: str = "data", output_plot: str = "results/accuracy_vs_er.png"):
    os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)

    # Ensure dataset exists
    build_dataset(data_dir=data_dir, architectures=HELDOUT_ARCHITECTURES, severities=ATTACK_SEVERITIES)

    calib = load_calibration("models/calibration.json")
    cnn_model, cnn_ref = load_fewshot_cnn("models/fewshot_cnn.pt")

    severities = [1, 2, 4, 8, 16, 23]
    er_percentages = [embedding_rate(x) * 100.0 for x in severities]

    stat_accuracies = []
    cnn_accuracies = []
    ensemble_accuracies = []

    print("\n" + "=" * 60)
    print("  ModelSentinel Benchmark: Out-of-Distribution Evaluation")
    print(f"  Held-out Architectures: {HELDOUT_ARCHITECTURES}")
    print("=" * 60)

    for x in severities:
        er_pct = embedding_rate(x) * 100.0
        stat_correct = 0
        cnn_correct = 0
        ens_correct = 0
        total_samples = 0

        # We evaluate on each held-out model both benign and attacked
        for arch in HELDOUT_ARCHITECTURES:
            benign_path = os.path.join(data_dir, "benign", f"{arch}.pt")
            if not os.path.exists(benign_path):
                continue

            flat_benign = load_flat_weights(benign_path)

            # Test 1: Benign evaluation (ground truth = Clean)
            total_samples += 1
            gf_b = to_gf_image(flat_benign)
            cnn_in_b = resize_for_cnn(gf_b, size=64)

            ent_b = byte_entropy_score(flat_benign, calib)
            ac_b = autocorrelation_score(flat_benign, calib)
            kl_b = kl_divergence_score(flat_benign, calib)
            wd_b = weight_distribution_score(flat_benign, calib)
            cnn_res_b = classify(cnn_model, cnn_in_b, cnn_ref)
            verdict_b = compute_ensemble_verdict(ent_b, ac_b, kl_b, wd_b, cnn_res_b)

            stat_score_b = 0.35 * ent_b[0] + 0.35 * kl_b[0] + 0.20 * ac_b[0] + 0.10 * wd_b[0]
            if stat_score_b < 0.5:
                stat_correct += 1
            if cnn_res_b["label"] == "Clean":
                cnn_correct += 1
            if verdict_b["tier"] in ["Clean", "Low-Risk"]:
                ens_correct += 1

            # Test 2: Attacked evaluation (ground truth = Malicious)
            total_samples += 1
            # Also test random bytes payload variation for robustness
            payload = EICAR_PAYLOAD if np.random.rand() > 0.3 else np.random.bytes(512)
            flat_att = embed_payload(flat_benign, X=x, payload=payload)

            gf_a = to_gf_image(flat_att)
            cnn_in_a = resize_for_cnn(gf_a, size=64)

            ent_a = byte_entropy_score(flat_att, calib)
            ac_a = autocorrelation_score(flat_att, calib)
            kl_a = kl_divergence_score(flat_att, calib)
            wd_a = weight_distribution_score(flat_att, calib)
            cnn_res_a = classify(cnn_model, cnn_in_a, cnn_ref)
            verdict_a = compute_ensemble_verdict(ent_a, ac_a, kl_a, wd_a, cnn_res_a)

            stat_score_a = 0.35 * ent_a[0] + 0.35 * kl_a[0] + 0.20 * ac_a[0] + 0.10 * wd_a[0]
            if stat_score_a >= 0.5:
                stat_correct += 1
            if cnn_res_a["label"] == "Malicious":
                cnn_correct += 1
            if verdict_a["tier"] in ["Suspicious", "Malicious"]:
                ens_correct += 1

        acc_stat = (stat_correct / total_samples) * 100.0
        acc_cnn = (cnn_correct / total_samples) * 100.0
        acc_ens = (ens_correct / total_samples) * 100.0

        stat_accuracies.append(acc_stat)
        cnn_accuracies.append(acc_cnn)
        ensemble_accuracies.append(acc_ens)

        print(f"  [X = {x:2d} | ER = {er_pct:5.2f}%] -> Stat: {acc_stat:5.1f}% | CNN: {acc_cnn:5.1f}% | Combined Ensemble: {acc_ens:5.1f}%")

    print("=" * 60)

    # Generate high quality evaluation plot
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Plot curves
    ax.plot(er_percentages, ensemble_accuracies, marker="o", linewidth=2.8, markersize=8, color="#2563EB", label="ModelSentinel Combined Ensemble", zorder=4)
    ax.plot(er_percentages, cnn_accuracies, marker="s", linewidth=2.0, markersize=7, linestyle="--", color="#7C3AED", label="Few-Shot Siamese CNN (OSL-CNN)", zorder=3)
    ax.plot(er_percentages, stat_accuracies, marker="^", linewidth=2.0, markersize=7, linestyle=":", color="#059669", label="Statistical Baseline (4 Tests)", zorder=2)

    # Annotate X values on top of x-axis
    for i, x in enumerate(severities):
        ax.annotate(
            f"X={x}",
            (er_percentages[i], ensemble_accuracies[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#1E3A8A",
        )

    # Reference threshold line
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random Guess Baseline")

    ax.set_title("Steganography Detection Accuracy vs. Embedding Rate (%)\n(Evaluated on Out-of-Distribution Held-Out Architectures)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Embedding Rate (% = X / 32 bits per float32 weight)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Detection Accuracy (%)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylim(40, 105)
    ax.set_xlim(0, 75)
    ax.set_xticks(er_percentages)
    ax.set_xticklabels([f"{er:.1f}%\n(X={x})" for er, x in zip(er_percentages, severities)])

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=10, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[✓] High-resolution evaluation plot successfully saved to {output_plot}\n")


if __name__ == "__main__":
    evaluate_pipeline()
