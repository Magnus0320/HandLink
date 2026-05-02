# HandLink: A Vision-Based Hand Gesture Recognition System for macOS Media Control

HandLink recognises hand gestures in real time from a webcam and maps them to macOS media actions — volume up/down and play/pause — using native CoreGraphics HID events, fully compatible with macOS Tahoe.

---

## How It Works

```
Webcam → MediaPipe HandLandmarker → Scale-normalised 63-D features
    → MLPClassifier + confidence gate → Temporal smoother (7-frame vote)
    → Gesture label → Action dispatcher → CoreGraphics HID media-key event
```

The classifier runs every frame. Only predictions above a confidence threshold enter the voting buffer; uncertain frames flush the buffer immediately. When a mapped gesture has been held continuously for its required hold duration (and the post-fire cooldown has elapsed), a Quartz HID event is posted directly into the macOS input stream — no Automation permission required.

---

## Supported Gestures

| Gesture | Hand pose | Action | Hold | Cooldown |
|---|---|---|---|---|
| `fist` | All fingers curled tightly into palm | Play / Pause | 0.35 s | 3 s |
| `thumbs_up` | Fist with thumb pointing straight up | Volume + | instant | 1 s |
| `thumbs_down` | Fist with thumb pointing straight down | Volume − | instant | 1 s |
| `open_hand` | All five fingers fully extended and spread | *(recognised, no action)* | — | — |
| `peace` | Index + middle fingers raised in a V | *(recognised, no action)* | — | — |
| `pointing` | Index finger extended forward, others curled | *(recognised, no action)* | — | — |
| `stop` | All fingers together, palm facing camera | *(recognised, no action)* | — | — |
| `none` | Random non-gesture hand poses | *(silence class)* | — | — |

Gestures with no action are still classified and shown in the UI; they simply do not trigger any system command. New actions can be wired up in `src/actions.py` without touching any other file.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Hand detection | MediaPipe `HandLandmarker` (Tasks API, float16) |
| Computer vision | OpenCV |
| ML classifier | scikit-learn `MLPClassifier` |
| Numerical ops | NumPy |
| macOS media control | `pyobjc-framework-Quartz` — CoreGraphics HID events |

**Requirements:** Python 3.10+, macOS, webcam.

---

## Installation

```bash
# 1. Clone and enter the project
git clone <repo-url> HandLink
cd HandLink

# 2. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` installs:

| Package | Purpose |
|---|---|
| `mediapipe>=0.10.30` | Hand landmark detection |
| `opencv-python>=4.10.0` | Video capture and rendering |
| `numpy>=1.26.0` | Array operations |
| `scikit-learn>=1.4.0` | MLP classifier and scaler |
| `pyobjc-framework-Quartz>=10.0` | macOS media-key HID events |

---

## Usage

### 1 — Collect gesture data

```bash
# Core gestures
python src/collect_data.py --gesture fist        --samples 300
python src/collect_data.py --gesture open_hand   --samples 300
python src/collect_data.py --gesture peace       --samples 300
python src/collect_data.py --gesture thumbs_up   --samples 300
python src/collect_data.py --gesture thumbs_down --samples 300
python src/collect_data.py --gesture pointing    --samples 300
python src/collect_data.py --gesture stop        --samples 300

# Silence class — collect varied non-gesture hand poses
python src/collect_data.py --gesture none        --samples 500
```

**During collection:**
- The camera feed stays live throughout — a real-time countdown replaces the old blocking freeze.
- The on-screen description tells you exactly what hand pose to hold.
- Press `SPACE` to start a 3-second countdown, then hold the gesture.
- Press `Q` to quit early; any samples collected so far are saved.
- Running the command again for the same gesture **appends** to the existing file.

**Tips for cross-person generalisation:**
- Collect in 2–3 short sessions rather than one long one.
- Vary your distance from the camera (40 cm, 60 cm, 80 cm) and wrist tilt (±20°).
- Use the front of one hand consistently (right hand, palm facing camera).

### 2 — Train the model

```bash
python src/train.py --data-dir data --model-dir models
```

Training prints a per-class classification report and saves three files:

| File | Contents |
|---|---|
| `models/gesture_model.pkl` | Trained `MLPClassifier` |
| `models/label_map.pkl` | `{index: gesture_name}` dict |
| `models/scaler.pkl` | Fitted `StandardScaler` |

### 3 — Run live recognition

```bash
python src/recognize.py

# Optional flags
python src/recognize.py --conf-threshold 0.75   # raise confidence gate (default 0.70)
python src/recognize.py --smoothing 10           # wider vote window (default 7)
```

**What you see:**

| UI element | Location | Meaning |
|---|---|---|
| Green skeleton | Over detected hand(s) | MediaPipe landmarks |
| Bottom banner — green label | Gesture name | Confident, mapped gesture |
| Bottom banner — grey label | Gesture name | Confident, unmapped gesture |
| Bottom banner — confidence % | Bottom right | Classifier probability |
| Top-right HUD — label | e.g. `Play/Pause` | Last fired action |
| Top-right HUD — progress bar | Fills left → right | Cooldown remaining (orange → green) |
| `Uncertain (XX%)` | Top left | Hand visible but below confidence threshold |
| `No hand detected` | Top left | No hand in frame |

Press `Q` to quit.

---

## Architecture

### Feature engineering

MediaPipe returns 21 (x, y, z) normalised-image-coordinate landmarks per hand. HandLink transforms them into a 63-dimensional feature vector with two invariance properties:

**Translation invariance** — every landmark is offset by the wrist (landmark 0), so the hand's position on screen does not affect the features.

**Scale invariance** — every coordinate is divided by the 2-D Euclidean distance from the wrist to the middle-finger MCP joint (landmark 9). This makes features the same regardless of hand size or camera distance, which is the primary fix for cross-person generalisation.

```python
scale = hypot(ref.x - wrist.x, ref.y - wrist.y)   # landmark 9 vs landmark 0
feature[i] = (lm[i] - wrist) / scale
```

### Augmentation

To further reduce overfitting to a single person's hand, the training split (not the test split) receives 4× synthetic copies:

- **Random 2-D in-plane rotation** ±20° around the wrist — simulates different hand tilts.
- **Gaussian landmark noise** σ = 0.02 (in normalised units) — simulates finger-proportion variation and detector jitter.

Combined with the 300-sample base and 5× augmentation, the MLP trains on ~9,800 samples for 7 gesture classes.

### Classifier

| Hyperparameter | Value | Reason |
|---|---|---|
| Architecture | 128 → 64 → N classes | Sufficient capacity for 63-D input |
| Activation | ReLU | Standard for MLPs |
| L2 regularisation (`alpha`) | `0.01` | 100× default — key lever against single-person overfitting |
| Early stopping | `validation_fraction=0.1` | Prevents overtraining |
| `max_iter` | 500 | Upper bound; early stopping typically kicks in first |

### Temporal smoothing and confidence gating

Every frame, the raw classifier output goes through two filters before a label is displayed or an action is fired:

1. **Confidence gate** — if `max(proba) < threshold` (default 0.70), the prediction is discarded and the vote buffer is flushed. This is what silences phantom predictions when no deliberate gesture is shown.
2. **Majority-vote smoother** — a `deque(maxlen=7)` of recent class indices; the displayed label is the mode. Frames that fail the confidence gate clear the buffer, so a dropped hand never leaves a stale label on screen.

### Action system and macOS Tahoe compatibility

Media controls use `Quartz.NSEvent.otherEventWithType…` + `CGEventPost` to post `NSSystemDefined` HID events with `NX_KEYTYPE_*` constants from `hidsystem/ev_keymap.h`. This is identical to what a physical media key produces and requires **no Automation or Accessibility permissions**.

This approach was adopted because:
- `osascript "set volume output volume"` silently fails on Tahoe without explicit Automation permission for the Python interpreter.
- `tell application "System Events" to key code 179` opens the emoji picker on Tahoe — `179` is outside the standard 0–127 key code range.

| NX constant | Value | Used for |
|---|---|---|
| `NX_KEYTYPE_SOUND_UP` | 0 | Volume + |
| `NX_KEYTYPE_SOUND_DOWN` | 1 | Volume − |
| `NX_KEYTYPE_PLAY` | 16 | Play / Pause toggle |

Each action has an independent **cooldown** (locks out re-firing) and an optional **hold duration** (gesture must be continuously detected before the action fires):

- Volume gestures fire instantly with a 1-second cooldown.
- Play/Pause requires a **0.35-second continuous hold** (prevents accidental triggers from a briefly-clenched fist) and a 3-second cooldown (prevents the toggle from flipping back immediately).

The hold timer resets whenever the gesture disappears — confidence drop, hand leaving frame, or gesture change — so a partial hold never carries over.

---

## File Structure

```
HandLink/
├── README.md
├── requirements.txt
├── data/                        # Collected training samples (.npy per gesture)
│   ├── fist.npy
│   ├── open_hand.npy
│   ├── peace.npy
│   ├── thumbs_up.npy
│   ├── thumbs_down.npy
│   ├── pointing.npy
│   ├── stop.npy
│   └── none.npy
├── models/
│   ├── gesture_model.pkl        # Trained MLPClassifier
│   ├── label_map.pkl            # {index → gesture name}
│   ├── scaler.pkl               # Fitted StandardScaler
│   └── hand_landmarker.task     # MediaPipe model (auto-downloaded)
└── src/
    ├── config.py                # Gesture registry (names + descriptions)
    ├── actions.py               # Gesture → macOS action mapping + cooldown/hold logic
    ├── mp_utils.py              # MediaPipe helpers, feature extraction
    ├── collect_data.py          # Interactive data collection
    ├── train.py                 # Model training + augmentation
    └── recognize.py             # Live recognition loop + HUD rendering
```

---

## Adding a New Gesture

1. **Register it** in `src/config.py` — add a name and pose description to `GESTURES`.
2. **Collect data** — `python src/collect_data.py --gesture <name> --samples 300`
3. **Retrain** — `python src/train.py`
4. *(Optional)* **Wire an action** — add an entry to `ACTIONS` in `src/actions.py`.

## Adding a New Action

Edit `ACTIONS` in `src/actions.py`:

```python
"pointing": Action("Action label", NX_KEYTYPE_CONSTANT, cooldown=1.0, hold_seconds=0.0),
```

No other file needs to change.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Phantom gestures when hand is idle | Lower `--conf-threshold` slightly or collect more `none` samples |
| Works for you but not others | Re-collect data at varied distances/tilts; scale invariance handles size differences |
| Volume / play-pause has no effect | Install `pyobjc-framework-Quartz`: `pip install pyobjc-framework-Quartz` |
| "Missing model" error | Run `python src/train.py` first |
| Webcam not opening | Check macOS camera permissions for Terminal / your IDE |
| Gesture flickers between labels | Increase `--smoothing` (e.g. `--smoothing 12`) |
