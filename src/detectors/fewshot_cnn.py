"""
Few-Shot Siamese / Embedding CNN Detector (OSL-CNN style).
4 Conv blocks -> 64-d L2-normalized embedding space.
Trained with Triplet Loss on benign vs attacked Grayscale-Fourpart representations.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OSLCNN(nn.Module):
    """
    4-block Convolutional Embedding Network for Grayscale-Fourpart stego representations.
    Produces a 64-dimensional L2-normalized feature embedding.
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8 -> 4
        )
        self.fc = nn.Linear(64 * 4 * 4, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, 1, 64, 64)
        Returns: (B, 64) L2-normalized embeddings
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        emb = self.fc(x)
        return F.normalize(emb, p=2, dim=1)


class TripletLoss(nn.Module):
    """
    Triplet loss with Euclidean distance on L2-normalized embeddings.
    L = max(0, ||a - p||_2 - ||a - n||_2 + margin)
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        dist_pos = torch.norm(anchor - positive, p=2, dim=1)
        dist_neg = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        return torch.mean(loss)


def generate_triplets(
    benign_tensors: torch.Tensor,
    malicious_tensors: torch.Tensor,
    num_triplets: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples (anchor, positive, negative) triplet batches from benign and malicious pools.
    """
    num_b = len(benign_tensors)
    num_m = len(malicious_tensors)

    anchors: List[torch.Tensor] = []
    positives: List[torch.Tensor] = []
    negatives: List[torch.Tensor] = []

    for _ in range(num_triplets):
        # 50% benign anchors, 50% malicious anchors
        if np.random.rand() > 0.5 and num_b >= 2:
            idx_a, idx_p = np.random.choice(num_b, size=2, replace=False)
            idx_n = np.random.choice(num_m)
            anchors.append(benign_tensors[idx_a])
            positives.append(benign_tensors[idx_p])
            negatives.append(malicious_tensors[idx_n])
        elif num_m >= 2:
            idx_a, idx_p = np.random.choice(num_m, size=2, replace=False)
            idx_n = np.random.choice(num_b)
            anchors.append(malicious_tensors[idx_a])
            positives.append(malicious_tensors[idx_p])
            negatives.append(benign_tensors[idx_n])

    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives),
    )


def train_fewshot_cnn(
    benign_images: np.ndarray,
    malicious_images: np.ndarray,
    malicious_severities: Optional[np.ndarray] = None,
    epochs: int = 40,
    lr: float = 1e-3,
    save_path: str = "models/fewshot_cnn.pt",
) -> Tuple[OSLCNN, Dict[str, Any]]:
    """
    Trains OSLCNN via Triplet Loss on few-shot benign and malicious GF images.
    Computes benign centroid, malicious centroid, and per-severity reference embeddings.

    Args:
        benign_images: (N_b, 1, 64, 64) float32 in [0, 1]
        malicious_images: (N_m, 1, 64, 64) float32 in [0, 1]
        malicious_severities: (N_m,) int array of X values
        epochs: Number of training epochs (default: 40)
        lr: Learning rate (default: 1e-3)
        save_path: Path to save trained model and centroid references

    Returns:
        (trained_model, reference_dict)
    """
    device = torch.device("cpu")
    model = OSLCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = TripletLoss(margin=0.5)

    b_tens = torch.from_numpy(benign_images).float().to(device)
    m_tens = torch.from_numpy(malicious_images).float().to(device)

    model.train()
    print(f"[*] Training Few-Shot CNN ({len(b_tens)} benign, {len(m_tens)} malicious images) for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        anchors, positives, negatives = generate_triplets(b_tens, m_tens, num_triplets=96)

        emb_a = model(anchors)
        emb_p = model(positives)
        emb_n = model(negatives)

        loss = criterion(emb_a, emb_p, emb_n)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:2d}/{epochs:2d} - Triplet Loss: {loss.item():.4f}")

    # Compute reference embeddings & centroids in eval mode
    model.eval()
    with torch.no_grad():
        b_embs = model(b_tens).cpu().numpy()
        m_embs = model(m_tens).cpu().numpy()

        b_centroid = np.mean(b_embs, axis=0)
        b_centroid = b_centroid / (np.linalg.norm(b_centroid) + 1e-8)

        m_centroid = np.mean(m_embs, axis=0)
        m_centroid = m_centroid / (np.linalg.norm(m_centroid) + 1e-8)

        # Severity centroids for X in {1, 2, 4, 8, 16, 23}
        severity_centroids: Dict[int, List[float]] = {}
        if malicious_severities is not None:
            for x_val in np.unique(malicious_severities):
                mask = malicious_severities == x_val
                if np.any(mask):
                    x_embs = m_embs[mask]
                    c_x = np.mean(x_embs, axis=0)
                    c_x = c_x / (np.linalg.norm(c_x) + 1e-8)
                    severity_centroids[int(x_val)] = c_x.tolist()

    ref_dict = {
        "benign_centroid": b_centroid.tolist(),
        "malicious_centroid": m_centroid.tolist(),
        "benign_embeddings": b_embs.tolist(),
        "malicious_embeddings": m_embs.tolist(),
        "severity_centroids": severity_centroids,
    }

    # Save state dict and reference metadata
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "reference_metadata": ref_dict,
    }
    torch.save(checkpoint, save_path)
    print(f"[*] Saved Few-Shot CNN checkpoint to {save_path}")

    return model, ref_dict


def load_fewshot_cnn(path: str = "models/fewshot_cnn.pt") -> Tuple[OSLCNN, Dict[str, Any]]:
    """Loads trained OSLCNN model and reference embeddings."""
    model = OSLCNN()
    if not os.path.exists(path):
        # Return initialized model if not trained yet
        dummy_ref = {
            "benign_centroid": [0.0] * 64,
            "malicious_centroid": [1.0] * 64,
            "benign_embeddings": [],
            "malicious_embeddings": [],
            "severity_centroids": {},
        }
        return model, dummy_ref

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["reference_metadata"]


def classify(
    model: OSLCNN,
    image: np.ndarray,
    reference_dict: Dict[str, Any],
    mode: str = "centroid",
) -> Dict[str, Any]:
    """
    Classifies a 64x64 Grayscale-Fourpart image.

    Args:
        model: Trained OSLCNN model.
        image: (64, 64) or (1, 1, 64, 64) float32 in [0, 1].
        reference_dict: Stored reference embeddings and centroids.
        mode: "centroid" or "1-nn".

    Returns:
        Dictionary with:
        - label: "Clean" or "Malicious"
        - confidence: float 0.0 - 1.0
        - dist_benign: float
        - dist_malicious: float
        - estimated_severity: int (or None)
        - embedding: list of 64 floats
    """
    model.eval()
    if image.ndim == 2:
        inp = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        inp = torch.from_numpy(image).float().unsqueeze(0)
    else:
        inp = torch.from_numpy(image).float()

    with torch.no_grad():
        emb = model(inp).squeeze(0).cpu().numpy()

    b_centroid = np.array(reference_dict.get("benign_centroid", [0.0] * 64), dtype=np.float32)
    m_centroid = np.array(reference_dict.get("malicious_centroid", [0.0] * 64), dtype=np.float32)

    dist_b = float(np.linalg.norm(emb - b_centroid))
    dist_m = float(np.linalg.norm(emb - m_centroid))

    if mode == "1-nn" and reference_dict.get("benign_embeddings"):
        b_all = np.array(reference_dict["benign_embeddings"])
        m_all = np.array(reference_dict["malicious_embeddings"])
        min_b = float(np.min(np.linalg.norm(b_all - emb, axis=1)))
        min_m = float(np.min(np.linalg.norm(m_all - emb, axis=1)))
        dist_b, dist_m = min_b, min_m

    # Softmax over negative distances to calculate pseudo-probability
    scale = 3.0
    exp_b = np.exp(-scale * dist_b)
    exp_m = np.exp(-scale * dist_m)
    prob_malicious = float(exp_m / (exp_b + exp_m + 1e-8))

    label = "Malicious" if dist_m < dist_b else "Clean"
    confidence = prob_malicious if label == "Malicious" else (1.0 - prob_malicious)

    # Estimate embedding severity X from severity centroids
    severity_centroids = reference_dict.get("severity_centroids", {})
    estimated_x = None
    if severity_centroids:
        best_x = None
        min_x_dist = float("inf")
        for x_str, c_x_list in severity_centroids.items():
            c_x = np.array(c_x_list, dtype=np.float32)
            d = float(np.linalg.norm(emb - c_x))
            if d < min_x_dist:
                min_x_dist = d
                best_x = int(x_str)
        estimated_x = best_x

    return {
        "label": label,
        "confidence": confidence,
        "prob_malicious": prob_malicious,
        "dist_benign": dist_b,
        "dist_malicious": dist_m,
        "estimated_severity": estimated_x,
        "embedding": emb.tolist(),
    }
