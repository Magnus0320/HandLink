# Hand Gesture Recognizer

Real-time hand gesture recognition from webcam using MediaPipe hand landmarks and a scikit-learn neural network classifier.

## Overview

This project captures hand gestures via webcam, extracts hand landmark features, trains a machine learning model, and performs live gesture recognition with confidence scores displayed on the video feed.

**Key Features:**
- Real-time hand detection and landmark extraction using MediaPipe
- Custom gesture data collection pipeline
- Neural network (MLP) classification trained on collected samples
- Live inference with temporal smoothing for stable predictions
- Visual feedback with skeleton overlay and confidence percentage

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Hand Detection** | MediaPipe (hand_landmarker.task) |
| **Computer Vision** | OpenCV |
| **ML Model** | scikit-learn MLPClassifier |
| **Data Processing** | NumPy |

**Requirements:**
- Python 3.8+
- Webcam

## Installation

1. **Clone/navigate to the project:**
   ```bash
   cd hand-gesture-recognizer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This installs:
   - `mediapipe>=0.10.30` — hand landmark detection
   - `opencv-python>=4.10.0` — video capture & rendering
   - `numpy>=1.26.0` — numerical operations
   - `scikit-learn>=1.4.0` — machine learning

## Usage

### Step 1: Collect Gesture Data

Record samples for each gesture (you need at least 50-100 samples per gesture):

```bash
python src/collect_data.py --gesture fist --samples 200
python src/collect_data.py --gesture open_hand --samples 200
python src/collect_data.py --gesture peace --samples 200
python src/collect_data.py --gesture thumbs_up --samples 200
```

**During collection:**
- Press `SPACE` to start a 3-second countdown
- Hold your gesture still during recording
- Press `Q` to quit early
- Data is saved to `data/<gesture>.npy`

### Step 2: Train the Model

```bash
python src/train.py --data-dir data --model-dir models
```

This:
- Loads all `.npy` files from `data/`
- Extracts hand landmark features (63 floats per sample: x,y,z × 21 landmarks)
- Trains an MLPClassifier on the dataset
- Saves model artifacts:
  - `models/gesture_model.pkl` — trained classifier
  - `models/label_map.pkl` — gesture name mapping
  - `models/scaler.pkl` — feature scaler

### Step 3: Run Live Recognition

```bash
python src/recognize.py --model-dir models
```

This starts live gesture recognition:
- Webcam feed with hand skeleton overlay (green lines)
- Real-time gesture classification with confidence %
- Temporal smoothing (last 7 predictions averaged)
- Press `Q` to quit

## Supported Gestures

Four gestures are recognized:
- **fist** — closed fist
- **open_hand** — open palm
- **peace** — peace sign (two fingers up)
- **thumbs_up** — thumbs up

*Pre-collected training data is included for all four gestures.*

## Architecture

### Data Pipeline

```
Webcam → MediaPipe HandLandmarker (21 landmarks per hand)
    ↓
Extract Features (63D: x,y,z offset from wrist)
    ↓
Normalize & Scale
    ↓
MLPClassifier Prediction
    ↓
Temporal Smoothing (voting over last N frames)
    ↓
Display Label + Confidence
```

### Feature Engineering

- **21 hand landmarks** detected by MediaPipe (wrist, fingers, knuckles)
- **63 features** per frame: (x, y, z) coordinates relative to wrist
- **Sample shape**: (N, 63) NumPy arrays stored in `data/<gesture>.npy`; N = number of captured frames
- **Translation invariance**: Each landmark (x, y, z) is offset by wrist coordinates so features are position-invariant
- **Scaling**: StandardScaler fitted on training data and applied to both train and test splits

**Data Collection Mechanics:**
**Architecture:**
- **Algorithm**: Multi-Layer Perceptron (MLPClassifier from scikit-learn) — supervised learning on labeled landmark features
- **Input**: 63-dimensional feature vectors (21 landmarks × 3 coordinates)
- **Hidden layers**: 128 → 64 neurons
- **Activation function**: ReLU (Rectified Linear Unit: f(x) = max(0, x))
- **Output**: 4 class probabilities (one per gesture)
- **Total parameters**: ~16.7k (63×128 + 128 + 128×64 + 64 + 64×4 + 4)
- **Training hyperparams**: max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=42
- **Test accuracy**: 100% (on 80/20 train/test split with 200 samples per gesture)

**Inference & Smoothing:**
- **Temporal smoothing**: Majority voting over a rolling deque of the last 7 predictions
- **Data structure**: `collections.deque(maxlen=7)` stores recent predicted class indices; smoothed label = most frequent class
- **Error handling**: If no hand is detected, skips feature extraction and displays "No hand detected" (prior predictions fade as deque ages)

**Skeleton Overlay:**
- Green hand skeleton drawn via `cv2.line()` connecting 21 landmarks
- Connections follow anatomical hand structure (thumb chain, finger chains, palm)
- Landmarks rendered as white-filled circles with green outlines
### Model Details

- **Algorithm**: Multi-Layer Perceptron (MLPClassifier from scikit-learn)
- **Input**: 63-dimensional feature vectors
- **Output**: Class probabilities for each gesture
- **Smoothing**: Majority voting over the last 7 predictions for temporal stability

## File Structure

```
hand-gesture-recognizer/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── data/                  # Pre-collected training data
│   ├── fist.npy
│   ├── open_hand.npy
│   ├── peace.npy
│   └── thumbs_up.npy
├── models/                # Trained models & MediaPipe asset
│   └── hand_landmarker.task
└── src/
    ├── collect_data.py    # Data collection script
    ├── train.py           # Model training script
    ├── recognize.py       # Live recognition script
    └── mp_utils.py        # MediaPipe utilities
```

## Example Workflow

```bash
# 1. Collect 200 samples for each gesture
for gesture in fist open_hand peace thumbs_up; do
  python src/collect_data.py --gesture $gesture --samples 200
done

# 2. Train the model
python src/train.py --data-dir data --model-dir models

# 3. Run live recognition
python src/recognize.py --model-dir models
```

## Troubleshooting

- **"No hand detected"** → Ensure adequate lighting and hand is clearly visible
- **"Missing model"** → Run `train.py` first to generate model files
- **Slow/laggy recognition** → Reduce gesture complexity or increase training samples
- **Webcam not working** → Check camera permissions and test with `cv2.VideoCapture(0)`

## Tips for Best Results

1. **Collect diverse samples** — vary hand position, angle, and scale
2. **Use consistent lighting** — train and test under similar conditions
3. **More data = better accuracy** — aim for 200+ samples per gesture
4. **Clear gestures** — ensure gestures are visually distinct
5. **Re-train regularly** — add new samples and retrain to improve performance