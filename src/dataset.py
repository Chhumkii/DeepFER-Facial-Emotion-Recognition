import os
import numpy as np
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import DATA_DIR, EMOTION_LABELS, BATCH_SIZE, AUGMENTATION_CONFIG


def verify_and_clean_dataset(data_dir=DATA_DIR, splits=["train", "test"]):
    """
    Validates image integrity across train/test splits and flags corrupted files.
    """
    corrupted_files = []
    verified_counts = {split: 0 for split in splits}

    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            continue
        for emotion in EMOTION_LABELS:
            emotion_path = os.path.join(split_path, emotion)
            if not os.path.exists(emotion_path):
                continue
            for fname in os.listdir(emotion_path):
                fpath = os.path.join(emotion_path, fname)
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                    verified_counts[split] += 1
                except Exception as e:
                    print(f"Corrupted image flagged: {fpath} - Error: {e}")
                    corrupted_files.append(fpath)

    return verified_counts, corrupted_files


def compute_class_weights(data_dir=DATA_DIR, split="train"):
    """
    Computes inverse frequency class weights for balanced training.
    """
    split_path = os.path.join(data_dir, split)
    labels = []

    for idx, emotion in enumerate(EMOTION_LABELS):
        emotion_path = os.path.join(split_path, emotion)
        if os.path.exists(emotion_path):
            count = len(os.listdir(emotion_path))
            labels.extend([idx] * count)

    if not labels:
        return {i: 1.0 for i in range(len(EMOTION_LABELS))}

    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    class_weight_dict = dict(zip(classes, weights))
    return class_weight_dict


def build_tf_data_generators(data_dir=DATA_DIR, target_size=(48, 48), color_mode="grayscale", batch_size=BATCH_SIZE):
    """
    Creates Keras ImageDataGenerator pipelines with real-time augmentation and rescaling.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=AUGMENTATION_CONFIG.get("rotation_range", 15),
        width_shift_range=AUGMENTATION_CONFIG.get("width_shift_range", 0.1),
        height_shift_range=AUGMENTATION_CONFIG.get("height_shift_range", 0.1),
        zoom_range=AUGMENTATION_CONFIG.get("zoom_range", 0.15),
        horizontal_flip=AUGMENTATION_CONFIG.get("horizontal_flip", True),
        fill_mode="nearest"
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=target_size,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    val_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=target_size,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, val_generator
