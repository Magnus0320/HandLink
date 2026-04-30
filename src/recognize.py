"""
Live gesture recognition from webcam.

Usage:
    python src/recognize.py --model-dir models

Requires a trained model (run train.py first).
Press Q to quit.
"""

import argparse
import os
import pickle
import sys
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mp_utils import draw_hand, ensure_model, extract_landmarks, make_landmarker, to_mp_image


def load_clf(model_dir: str):
    paths = {
        "model":  os.path.join(model_dir, "gesture_model.pkl"),
        "labels": os.path.join(model_dir, "label_map.pkl"),
        "scaler": os.path.join(model_dir, "scaler.pkl"),
    }
    for name, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {name}: {p}. Run train.py first.")
    with open(paths["model"],  "rb") as f: clf       = pickle.load(f)
    with open(paths["labels"], "rb") as f: label_map = pickle.load(f)
    with open(paths["scaler"], "rb") as f: scaler    = pickle.load(f)
    return clf, label_map, scaler


def run(model_dir: str, smoothing: int = 7):
    clf, label_map, scaler = load_clf(model_dir)
    recent: deque[int] = deque(maxlen=smoothing)

    model_path = ensure_model(model_dir)
    landmarker = make_landmarker(model_path)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(to_mp_image(rgb))

        label_text = ""
        confidence = 0.0

        if result.hand_landmarks:
            # use only the first detected hand
            hand_lms = result.hand_landmarks[0]
            draw_hand(frame, hand_lms)

            features = np.array(extract_landmarks(hand_lms), dtype=np.float32).reshape(1, -1)
            features_scaled = scaler.transform(features)
            pred_idx = clf.predict(features_scaled)[0]
            proba = clf.predict_proba(features_scaled)[0]
            confidence = proba[pred_idx]
            recent.append(pred_idx)

        if recent:
            smoothed = max(set(recent), key=recent.count)
            label_text = label_map[smoothed]

        h, w = frame.shape[:2]
        if label_text:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 70), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, label_text.upper(), (20, h - 18),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 255, 120), 2)
            cv2.putText(frame, f"{confidence * 100:.1f}%", (w - 110, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 100), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        cv2.putText(frame, "Q = quit", (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow("Gesture Recognizer", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--smoothing", type=int, default=7,
                        help="Frames to majority-vote over (reduces flicker)")
    args = parser.parse_args()
    run(args.model_dir, args.smoothing)
