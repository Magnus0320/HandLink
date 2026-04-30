"""
Train a small MLP on the collected .npy gesture files.

Usage:
    python src/train.py --data-dir data --model-dir models

Saves:
    models/gesture_model.pkl   — trained sklearn MLPClassifier
    models/label_map.pkl       — {index: gesture_name} dict
"""

import argparse
import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


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


def train(data_dir: str, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    print("Loading dataset...")
    X, y, label_map = load_dataset(data_dir)
    print(f"Total samples: {len(X)}, classes: {list(label_map.values())}\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True,
    )
    print("Training MLP...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n--- Test set results ---")
    print(classification_report(y_test, y_pred, target_names=list(label_map.values())))

    model_path = os.path.join(model_dir, "gesture_model.pkl")
    label_path = os.path.join(model_dir, "label_map.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    with open(label_path, "wb") as f:
        pickle.dump(label_map, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nModel saved → {model_path}")
    print(f"Label map  → {label_path}")
    print(f"Scaler     → {scaler_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    train(args.data_dir, args.model_dir)
