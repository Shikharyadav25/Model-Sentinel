#!/usr/bin/env python3
"""
ModelSentinel CLI: Command-line interface for model scanning,
attack simulation, few-shot training, and benchmark evaluation.
"""

import argparse
import os
import sys
import json
import numpy as np

from src.attack.xlsb_attack import EICAR_PAYLOAD, embed_payload, embedding_rate
from src.data.build_dataset import (
    ALL_ARCHITECTURES,
    ATTACK_SEVERITIES,
    TRAIN_ARCHITECTURES,
    build_dataset,
    load_cnn_training_data,
)
from src.detectors.fewshot_cnn import train_fewshot_cnn
from src.detectors.statistical import calibrate
from src.pipeline.scan import scan_model
from src.utils.weight_io import load_flat_weights, save_flat_weights_as_state_dict


# ANSI Color Codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
ORANGE = "\033[38;5;208m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    banner = f"""
{CYAN}{BOLD}======================================================================
  __  __           _      _  _____            _   _            _ 
 |  \\/  |         | |    | |/ ____|          | | (_)          | |
 | \\  / | ___   __| | ___| | (___   ___ _ __ | |_ _ _ __   ___| |
 | |\\/| |/ _ \\ / _` |/ _ \\ |\\___ \\ / _ \\ '_ \\| __| | '_ \\ / _ \\ |
 | |  | | (_) | (_| |  __/ |____) |  __/ | | | |_| | | | |  __/ |
 |_|  |_|\\___/ \\__,_|\\___|_|_____/ \\___|_| |_|\\__|_|_| |_|\\___|_|
 
 Steganographic Malware Detector & X-LSB Attack Simulator for AI Models
======================================================================{RESET}
"""
    print(banner)


def cmd_scan(args):
    """Scans a model file and prints a formatted terminal report."""
    path = args.path
    if not os.path.exists(path):
        print(f"{RED}[!] Error: File '{path}' does not exist.{RESET}")
        sys.exit(1)

    print(f"\n{CYAN}[*] Scanning model file:{RESET} {path}")
    print(f"{CYAN}[*] Running Grayscale-Fourpart extraction, Statistical tests & Few-Shot CNN...{RESET}\n")

    try:
        res = scan_model(path, cnn_mode=args.mode)
    except Exception as e:
        print(f"{RED}[!] Scan failed: {e}{RESET}")
        sys.exit(1)

    verdict = res["verdict"]
    tier = verdict["tier"]
    risk = verdict["risk_score"]

    color_map = {
        "Clean": GREEN,
        "Low-Risk": YELLOW,
        "Suspicious": ORANGE,
        "Malicious": RED,
    }
    tier_color = color_map.get(tier, RESET)

    print("+" + "-" * 68 + "+")
    print(f"| {BOLD}SCAN VERDICT:{RESET} {tier_color}{BOLD}{tier.upper()}{RESET}" + " " * (53 - len(tier)) + "|")
    print(f"| {BOLD}Risk Score:{RESET}   {tier_color}{risk:.1f} / 100.0{RESET}" + " " * (49 - len(f"{risk:.1f}")) + "|")
    if verdict.get("estimated_severity_x") is not None:
        est_x = verdict['estimated_severity_x']
        est_er = verdict['estimated_embedding_rate_percent']
        sev_str = f"Estimated Attack Severity: X = {est_x} (ER = {est_er:.1f}%)"
        print(f"| {ORANGE}{sev_str}{RESET}" + " " * (68 - len(sev_str)) + "|")
    print("+" + "-" * 68 + "+")
    print(f"| {verdict['summary'][:66]:<66} |")
    print("+" + "-" * 68 + "+")

    print(f"\n{BOLD}DETECTOR BREAKDOWN:{RESET}")
    for k, d in verdict["breakdown"].items():
        score_pct = d["percentage"]
        fired = d["fired"]
        status = f"{RED}ALERT{RESET}" if fired else f"{GREEN}PASS {RESET}"
        bar_len = int(score_pct / 5)
        bar = ("#" * bar_len).ljust(20)
        print(f"  [{status}] {d['name']:<30} [{bar}] {score_pct:5.1f}%")

    timing = res["timing"]
    print(f"\n{CYAN}Model Info:{RESET} {res['num_parameters']:,} parameters ({res['file_size_mb']} MB)")
    print(f"{CYAN}Scan Time:{RESET}  {timing['total_scan_sec']:.3f}s (Load: {timing['load_time_sec']}s, GF: {timing['gf_transform_sec']}s, Stat: {timing['statistical_sec']}s, CNN: {timing['cnn_sec']}s)\n")


def cmd_simulate_attack(args):
    """Simulates X-LSB steganography attack on a target model."""
    in_path = args.path
    out_path = args.out
    x = args.x

    if not os.path.exists(in_path):
        print(f"{RED}[!] Error: File '{in_path}' does not exist.{RESET}")
        sys.exit(1)

    if not (1 <= x <= 23):
        print(f"{RED}[!] Error: X must be between 1 and 23.{RESET}")
        sys.exit(1)

    er = embedding_rate(x) * 100.0
    print(f"\n{CYAN}[*] Starting X-LSB-Attack-Fill Simulation{RESET}")
    print(f"    Source Model:       {in_path}")
    print(f"    Attack Severity:    X = {x} bits ({er:.2f}% Embedding Rate)")
    print(f"    Payload Type:       {args.payload}")
    print(f"    Output Destination: {out_path}\n")

    if args.payload == "random":
        payload = np.random.bytes(1024)
    else:
        payload = EICAR_PAYLOAD

    flat = load_flat_weights(in_path)
    print(f"[*] Loaded {len(flat):,} float32 weights. Injecting synthetic payload...")
    attacked_flat = embed_payload(flat, X=x, payload=payload)

    save_flat_weights_as_state_dict(in_path, attacked_flat, out_path)
    print(f"{GREEN}[✓] Successfully generated attacked model: {out_path}{RESET}")
    print(f"[*] Try scanning it with: {BOLD}python cli.py scan {out_path}{RESET}\n")


def cmd_train(args):
    """Downloads model zoo, generates attacked dataset, calibrates statistical baselines, and trains CNN."""
    print(f"\n{CYAN}[*] Phase 1: Building Model Zoo & Stego Dataset...{RESET}")
    build_dataset(data_dir="data", architectures=ALL_ARCHITECTURES, severities=ATTACK_SEVERITIES)

    print(f"\n{CYAN}[*] Phase 2: Calibrating Statistical Baseline Ensemble...{RESET}")
    benign_flats = []
    for arch in TRAIN_ARCHITECTURES:
        b_path = f"data/benign/{arch}.pt"
        if os.path.exists(b_path):
            benign_flats.append(load_flat_weights(b_path))

    calib = calibrate(benign_flats, save_path="models/calibration.json")
    print(f"{GREEN}[✓] Statistical calibration saved to models/calibration.json{RESET}")

    print(f"\n{CYAN}[*] Phase 3: Loading Few-Shot CNN Dataset & Training OSLCNN...{RESET}")
    b_imgs, m_imgs, x_arr, arch_names = load_cnn_training_data(
        data_dir="data",
        train_archs=TRAIN_ARCHITECTURES,
        severities=ATTACK_SEVERITIES,
    )
    print(f"    Loaded {len(b_imgs)} benign and {len(m_imgs)} malicious GF representations.")

    train_fewshot_cnn(
        benign_images=b_imgs,
        malicious_images=m_imgs,
        malicious_severities=x_arr,
        epochs=args.epochs,
        lr=1e-3,
        save_path="models/fewshot_cnn.pt",
    )
    print(f"\n{GREEN}[✓] Training pipeline complete! Model checkpoints ready in models/{RESET}\n")


def cmd_evaluate(args):
    """Runs evaluation script to generate benchmark figures."""
    import subprocess
    cmd = [sys.executable, "scripts/evaluate.py"]
    subprocess.run(cmd, check=True)


def main():
    print_banner()
    parser = argparse.ArgumentParser(
        description="ModelSentinel: AI Model Steganography Malware Detector & Attack Simulator",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Scan
    p_scan = subparsers.add_parser("scan", help="Scan an AI model file (.pt, .pth, .safetensors)")
    p_scan.add_argument("path", type=str, help="Path to the model file to scan")
    p_scan.add_argument("--mode", type=str, default="centroid", choices=["centroid", "1-nn"], help="CNN classification mode")

    # Simulate Attack
    p_attack = subparsers.add_parser("simulate-attack", help="Simulate an X-LSB payload injection")
    p_attack.add_argument("path", type=str, help="Path to clean benign model file")
    p_attack.add_argument("--x", type=int, default=4, help="LSB severity X (1-23, default: 4)")
    p_attack.add_argument("--payload", type=str, default="eicar", choices=["eicar", "random"], help="Payload type")
    p_attack.add_argument("--out", type=str, default="attacked.pt", help="Output file path")

    # Train
    p_train = subparsers.add_parser("train", help="Build dataset, calibrate stats, and train Few-Shot CNN")
    p_train.add_argument("--epochs", type=int, default=40, help="Training epochs (default: 40)")

    # Evaluate
    subparsers.add_parser("evaluate", help="Run benchmark evaluation and generate results plot")

    args = parser.parse_args()
    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "simulate-attack":
        cmd_simulate_attack(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
