"""
Central registry of all supported gesture classes.

This is the single source of truth for gesture names and their
expected hand poses.  collect_data.py reads descriptions from here
and displays them on-screen during recording so poses stay consistent.
train.py warns if any registered gesture is missing from data/.

To add a new gesture:
  1. Add it to GESTURES below.
  2. Collect samples:  python src/collect_data.py --gesture <name>
  3. Re-train:         python src/train.py
"""

# gesture_name → description shown on screen during data collection
GESTURES: dict[str, str] = {
    # ── original ────────────────────────────────────────────────────────────
    "fist":      "All fingers curled tightly into palm, thumb over fingers",
    "open_hand": "All five fingers fully extended and spread wide",
    "peace":     "Index + middle fingers raised in a V, others curled",
    "thumbs_up": "Fist with thumb pointing straight up, other fingers curled",
    # ── new ─────────────────────────────────────────────────────────────────
    "pointing":   "Index finger extended straight forward, all others curled into palm",
    "stop":       "All fingers extended together, palm flat and facing the camera",
    "thumbs_down": "Fist with thumb pointing straight down, other fingers curled",
    # ── safety net ──────────────────────────────────────────────────────────
    "none":      "Random non-gesture poses: relaxed hand, partial poses, wrist only visible",
}
