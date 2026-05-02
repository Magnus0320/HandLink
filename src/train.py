"""
Train a small MLP on the collected .npy gesture files.

Usage:
    python src/train.py --data-dir data --model-dir models

Saves:
    models/gesture_model.pkl   — trained sklearn MLPClassifier
    models/label_map.pkl       — {index: gesture_name} dict
    models/scaler.pkl          — fitted StandardScaler

Tip: collect a 'none' gesture class (random, non-intentional hand poses)
     so the recognizer can stay silent instead of always forcing a guess:
         python src/collect_data.py --gesture none --samples 300
"""

import argparse
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(__file__))
from config import GESTURES
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def load_dataset(data_dir: str):
    X, y = [], []
    label_map = {}
    for idx, fname in enumerate(sorted(os.listdir(data_dir))):
        if not fname.endswith(".npy"):
            continue
        label = fname[:-4]
        label_map[idx] = label
        data = np.load(os.path.join(data_dir, fname))
        X.append(data)
        y.extend([idx] * len(data))
        print(f"  [{idx}] {label}: {len(data)} samples")
    return np.vstack(X), np.array(y), label_map


# ── augmentation ───────────────────────────────────────────────────────────────

def _rotate_landmarks_2d(row: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate x, y components of all 21 landmarks around the wrist (origin)."""
    c, s  = np.cos(angle_rad), np.sin(angle_rad)
    out   = row.copy()
    x_idx = np.arange(0, len(row), 3)   # x values at positions 0, 3, 6 …
    y_idx = np.arange(1, len(row), 3)   # y values at positions 1, 4, 7 …
    x, y  = out[x_idx].copy(), out[y_idx].copy()
    out[x_idx] = x * c - y * s
    out[y_idx] = x * s + y * c
    return out


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    copies: int = 4,
    noise_std: float = 0.02,
    max_rot_deg: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return augmented arrays with `copies` synthetic copies of every sample.

    Each copy applies:
      • Random 2-D in-plane rotation  (±max_rot_deg) around the wrist.
        This simulates different hand tilts and orientations.
      • Gaussian landmark noise  (σ = noise_std, in scale-normalised units).
        This simulates finger-proportion variation between people and
        small detector jitter.

    Together they greatly reduce overfitting to a single person's hand.
    The test split is kept clean (augmentation is applied only to X_train).
    """
    rng = np.random.default_rng(42)
    X_parts, y_parts = [X], [y]
    n = len(X)
    for _ in range(copies):
        angles = rng.uniform(-max_rot_deg, max_rot_deg, n) * (np.pi / 180)
        noise  = rng.normal(0, noise_std, X.shape)
        aug    = np.array([_rotate_landmarks_2d(X[i], angles[i]) for i in range(n)])
        aug   += noise
        X_parts.append(aug)
        y_parts.append(y)
    return np.vstack(X_parts), np.concatenate(y_parts)


# ── training ───────────────────────────────────────────────────────────────────

def train(data_dir: str, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)

    collected = {f[:-4] for f in os.listdir(data_dir) if f.endswith(".npy")}
    missing   = sorted(GESTURES.keys() - collected)
    if missing:
        print("Warning: the following registered gestures have no data yet:")
        for name in missing:
            print(f"  {name:12s}  —  {GESTURES[name]}")
        print("  Collect them with: python src/collect_data.py --gesture <name>\n")

    print("Loading dataset…")
    X, y, label_map = load_dataset(data_dir)
    print(f"Total samples: {len(X)}, classes: {list(label_map.values())}\n")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Augment only the training split so the test set stays clean
    print(f"Augmenting training set ({len(X_train)}", end="", flush=True)
    X_train, y_train = augment_dataset(X_train, y_train)
    print(f" → {len(X_train)} samples)…")

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True,
        alpha=0.01,     # L2 regularisation — 100× stronger than sklearn default
                        # (0.0001); key lever against overfitting to one person
    )
    print("Training MLP…")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n--- Test set results ---")
    print(classification_report(y_test, y_pred, target_names=list(label_map.values())))

    model_path  = os.path.join(model_dir, "gesture_model.pkl")
    label_path  = os.path.join(model_dir, "label_map.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    with open(model_path,  "wb") as f: pickle.dump(clf,       f)
    with open(label_path,  "wb") as f: pickle.dump(label_map, f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler,    f)

    print(f"\nModel saved → {model_path}")
    print(f"Label map  → {label_path}")
    print(f"Scaler     → {scaler_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default="data")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    train(args.data_dir, args.model_dir)
