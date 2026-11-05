import os
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from src.config import (
    DATA_DIR, MODELS_DIR, ASSETS_DIR, BATCH_SIZE, EPOCHS, INITIAL_LR,
    LABEL_SMOOTHING, CUSTOM_CNN_SHAPE, TRANSFER_LEARNING_SHAPE
)
from src.dataset import build_tf_data_generators, compute_class_weights
from src.models import build_custom_deepfer_cnn, build_transfer_learning_model


def plot_training_curves(history, output_dir=ASSETS_DIR):
    """
    Plots and saves training and validation accuracy/loss curves.
    """
    os.makedirs(output_dir, exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, acc, label="Training Accuracy")
    ax1.plot(epochs_range, val_acc, label="Validation Accuracy")
    ax1.set_title("Training and Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True)

    ax2.plot(epochs_range, loss, label="Training Loss")
    ax2.plot(epochs_range, val_loss, label="Validation Loss")
    ax2.set_title("Training and Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training curves saved to {plot_path}")


def train_model(model_type="custom", epochs=EPOCHS, batch_size=BATCH_SIZE, lr=INITIAL_LR):
    """
    Main training function compiling and fitting the specified model architecture.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    if model_type.lower() == "custom":
        target_size = (CUSTOM_CNN_SHAPE[0], CUSTOM_CNN_SHAPE[1])
        color_mode = "grayscale"
        model = build_custom_deepfer_cnn(input_shape=CUSTOM_CNN_SHAPE)
    else:
        target_size = (TRANSFER_LEARNING_SHAPE[0], TRANSFER_LEARNING_SHAPE[1])
        color_mode = "rgb"
        model = build_transfer_learning_model(model_type=model_type, input_shape=TRANSFER_LEARNING_SHAPE)

    train_gen, val_gen = build_tf_data_generators(
        data_dir=DATA_DIR,
        target_size=target_size,
        color_mode=color_mode,
        batch_size=batch_size
    )

    class_weights = compute_class_weights(data_dir=DATA_DIR)

    optimizer = Adam(learning_rate=lr)
    loss_fn = CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    checkpoint_filepath = os.path.join(MODELS_DIR, f"best_{model_type}_model.h5")
    csv_log_path = os.path.join(MODELS_DIR, f"{model_type}_training_log.csv")

    callbacks = [
        ModelCheckpoint(filepath=checkpoint_filepath, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        CSVLogger(filename=csv_log_path, append=True)
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks
    )

    plot_training_curves(history, output_dir=ASSETS_DIR)
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepFER Facial Emotion Recognition Model")
    parser.add_argument("--model_type", type=str, default="custom", choices=["custom", "MobileNetV2", "ResNet50"], help="Model architecture type")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=INITIAL_LR, help="Initial learning rate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
