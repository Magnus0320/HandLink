"""
Record gesture samples from webcam.

Usage:
    python src/collect_data.py --gesture thumbs_up --samples 200

Hold your gesture in front of the webcam. Press SPACE to start a 3-second
countdown then hold still. Press Q to quit early.
Samples are saved (or appended) to data/<gesture>.npy.
"""

import argparse
import os
import sys

import cv2
import numpy as np

# Allow running as `python src/collect_data.py` from the project root
sys.path.insert(0, os.path.dirname(__file__))
from mp_utils import draw_hand, ensure_model, extract_landmarks, make_landmarker, to_mp_image


def collect(gesture: str, n_samples: int, output_dir: str, model_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{gesture}.npy")
    existing = np.load(out_path) if os.path.exists(out_path) else np.empty((0, 63))
    samples: list[list[float]] = []

    model_path = ensure_model(model_dir)
    landmarker = make_landmarker(model_path)

    cap = cv2.VideoCapture(0)
    recording = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(to_mp_image(rgb))

        for hand_lms in result.hand_landmarks:
            draw_hand(frame, hand_lms)
            if recording:
                samples.append(extract_landmarks(hand_lms))

        total = len(existing) + len(samples)
        target = n_samples + len(existing)
        status = f"Gesture: {gesture}  |  Collected: {total}/{target}"
        color = (0, 200, 0) if recording else (0, 100, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "SPACE = start recording   Q = quit", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Collect Gesture Data", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" ") and not recording:
            # 3-second countdown before recording starts
            for i in range(3, 0, -1):
                ret2, f2 = cap.read()
                if ret2:
                    f2 = cv2.flip(f2, 1)
                    cv2.putText(f2, f"Starting in {i}...", (180, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.imshow("Collect Gesture Data", f2)
                cv2.waitKey(1000)
            recording = True

        if recording and len(samples) >= n_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    if samples:
        new_data = np.array(samples)
        combined = np.vstack([existing, new_data]) if len(existing) else new_data
        np.save(out_path, combined)
        print(f"Saved {len(samples)} samples → {out_path}  (total: {len(combined)})")
    else:
        print("No samples collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture", required=True, help="Gesture label name")
    parser.add_argument("--samples", type=int, default=200, help="Frames to capture")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    collect(args.gesture, args.samples, args.data_dir, args.model_dir)
