import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_META = {
    "angry": {"color": "#FF4D4D", "emoji": "😡"},
    "disgust": {"color": "#2ECC71", "emoji": "🤢"},
    "fear": {"color": "#9B59B6", "emoji": "😨"},
    "happy": {"color": "#F1C40F", "emoji": "😃"},
    "neutral": {"color": "#95A5A6", "emoji": "😐"},
    "sad": {"color": "#3498DB", "emoji": "😢"},
    "surprise": {"color": "#E67E22", "emoji": "😮"}
}

CUSTOM_CNN_SHAPE = (48, 48, 1)
TRANSFER_LEARNING_SHAPE = (224, 224, 3)

BATCH_SIZE = 64
EPOCHS = 50
INITIAL_LR = 1e-3
LABEL_SMOOTHING = 0.1

AUGMENTATION_CONFIG = {
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.15,
    "horizontal_flip": True
}
