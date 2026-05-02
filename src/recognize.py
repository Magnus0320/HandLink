"""
Live gesture recognition from webcam.

Usage:
    python src/recognize.py --model-dir models
    python src/recognize.py --model-dir models --conf-threshold 0.75

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
from actions import ACTIONS, cooldown_fraction, reset_hold, try_fire
from mp_utils import draw_hand, ensure_model, extract_landmarks, make_landmarker, to_mp_image

_DEFAULT_CONF      = 0.70
_DEFAULT_SMOOTHING = 7


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


# ── HUD drawing ────────────────────────────────────────────────────────────────

def _draw_action_hud(frame, last_label: str) -> None:
    """
    Top-right corner HUD showing:
      • Last fired action label  (e.g. 'Vol +', 'Pause')
      • Cooldown progress bar    (fills left→right; green = ready, orange = cooling)
    """
    h, w    = frame.shape[:2]
    pad     = 10
    box_w   = 155
    box_h   = 54
    x1      = w - box_w - pad
    y1      = pad
    x2      = x1 + box_w
    y2      = y1 + box_h

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # Action label
    display = last_label if last_label else "—"
    cv2.putText(frame, display,
                (x1 + 10, y1 + 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1)

    # Cooldown bar track (dark grey)
    bx1, by1 = x1 + 8,  y1 + 36
    bx2, by2 = x2 - 8,  y1 + 46
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (55, 55, 55), -1)

    # Cooldown bar fill
    frac    = cooldown_fraction()
    fill_x  = bx1 + int((bx2 - bx1) * frac)
    color   = (0, 210, 80) if frac >= 1.0 else (0, 150, 255)
    if fill_x > bx1:
        cv2.rectangle(frame, (bx1, by1), (fill_x, by2), color, -1)

    # "READY" tick when cooldown is complete
    if frac >= 1.0:
        cv2.putText(frame, "ready", (bx1, by1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 210, 80), 1)


# ── main loop ─────────────────────────────────────────────────────────────────

def run(
    model_dir: str,
    smoothing: int        = _DEFAULT_SMOOTHING,
    conf_threshold: float = _DEFAULT_CONF,
) -> None:
    clf, label_map, scaler = load_clf(model_dir)
    recent: deque[int] = deque(maxlen=smoothing)

    model_path = ensure_model(model_dir)
    landmarker = make_landmarker(model_path, num_hands=2)

    last_action_label: str = ""   # persists in HUD until next action fires

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(to_mp_image(rgb))

        confidence = 0.0

        if result.hand_landmarks:
            # Draw skeleton for all detected hands
            for hand_lms in result.hand_landmarks:
                draw_hand(frame, hand_lms)

            # Classify from the primary (first) hand
            hand_lms        = result.hand_landmarks[0]
            features        = np.array(extract_landmarks(hand_lms), dtype=np.float32).reshape(1, -1)
            features_scaled = scaler.transform(features)
            pred_idx        = clf.predict(features_scaled)[0]
            proba           = clf.predict_proba(features_scaled)[0]
            confidence      = proba[pred_idx]

            if confidence >= conf_threshold:
                recent.append(pred_idx)
            else:
                recent.clear()
                reset_hold()   # partial hold must not carry over to next attempt

        h, w = frame.shape[:2]

        if recent and result.hand_landmarks and confidence >= conf_threshold:
            smoothed   = max(set(recent), key=recent.count)
            label_text = label_map[smoothed]

            # ── fire action if gesture is mapped and cooldown has elapsed ──
            fired = try_fire(label_text)
            if fired:
                last_action_label = fired

            # ── bottom banner: gesture name + confidence ───────────────────
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 70), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Tint the gesture name green if it has a mapped action
            name_color = (0, 255, 120) if label_text in ACTIONS else (200, 200, 200)
            cv2.putText(frame, label_text.upper(), (20, h - 18),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, name_color, 2)
            cv2.putText(frame, f"{confidence * 100:.1f}%", (w - 110, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 100), 2)

        elif result.hand_landmarks:
            cv2.putText(frame, f"Uncertain  ({confidence * 100:.0f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            reset_hold()   # hand left frame; require a fresh hold next time
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        # ── action HUD (always visible) ────────────────────────────────────
        _draw_action_hud(frame, last_action_label)

        cv2.putText(frame,
                    f"Q = quit  |  threshold: {conf_threshold:.0%}",
                    (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow("HandLink", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",      default="models")
    parser.add_argument("--smoothing",      type=int,   default=_DEFAULT_SMOOTHING,
                        help="Frames to majority-vote over (reduces flicker)")
    parser.add_argument("--conf-threshold", type=float, default=_DEFAULT_CONF,
                        help=f"Min confidence to show a label (default {_DEFAULT_CONF})")
    args = parser.parse_args()
    run(args.model_dir, args.smoothing, args.conf_threshold)
