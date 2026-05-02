"""
Shared MediaPipe Tasks API helpers (MediaPipe 0.10.30+).

The old mp.solutions.hands context-manager API is gone in 0.10.30+.
We now use mp.tasks.python.vision.HandLandmarker directly.
"""

import math
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── model ──────────────────────────────────────────────────────────────────────

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_model(model_dir: str) -> str:
    """Download hand_landmarker.task into model_dir if not already present."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "hand_landmarker.task")
    if not os.path.exists(path):
        print(f"Downloading hand_landmarker.task → {path} …")
        urllib.request.urlretrieve(_MODEL_URL, path)
        print("Download complete.")
    return path


def make_landmarker(
    model_path: str,
    num_hands: int = 1,
    det_confidence: float = 0.7,
    presence_confidence: float = 0.7,
    tracking_confidence: float = 0.6,
) -> mp_vision.HandLandmarker:
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_hands=num_hands,
        min_hand_detection_confidence=det_confidence,
        min_hand_presence_confidence=presence_confidence,
        min_tracking_confidence=tracking_confidence,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


# ── landmarks ──────────────────────────────────────────────────────────────────

# (start_idx, end_idx) pairs for all 21 hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17), (0, 5), (5, 9), (9, 13),         # palm
]


def extract_landmarks(hand_landmarks: list) -> list[float]:
    """
    Return 63 floats: (x, y, z) × 21 landmarks.

    • Translation-invariant: every coordinate is offset by the wrist
      (landmark 0), so the hand position on screen doesn't matter.

    • Scale-invariant: divided by the 2-D wrist → middle-finger MCP
      (landmark 9) distance, so hand size and camera distance don't
      affect the features.  This is the key fix for cross-person
      generalisation — different people have proportionally similar
      hands even if the absolute pixel size varies.
    """
    wrist = hand_landmarks[0]
    ref   = hand_landmarks[9]          # middle-finger MCP — stable reference
    scale = math.hypot(ref.x - wrist.x, ref.y - wrist.y)
    if scale < 1e-6:
        scale = 1e-6                   # guard against degenerate / edge frames

    coords: list[float] = []
    for lm in hand_landmarks:
        coords.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale,
        ])
    return coords


def draw_hand(frame, hand_landmarks: list) -> None:
    """Draw skeleton overlay directly with OpenCV (no mp.solutions needed)."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, pts[start], pts[end], (0, 220, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)
        cv2.circle(frame, pt, 4, (0, 160, 0), 1)


def to_mp_image(rgb_frame) -> mp.Image:
    """Wrap a numpy RGB frame in the MediaPipe Image container."""
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
