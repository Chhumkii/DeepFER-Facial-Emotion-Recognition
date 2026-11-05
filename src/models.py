import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D,
    Dropout, Flatten, Dense, GlobalAveragePooling2D, Input
)
from tensorflow.keras.applications import MobileNetV2, ResNet50
from src.config import CUSTOM_CNN_SHAPE, TRANSFER_LEARNING_SHAPE, EMOTION_LABELS


def build_custom_deepfer_cnn(input_shape=CUSTOM_CNN_SHAPE, num_classes=len(EMOTION_LABELS)):
    """
    4-Block Convolutional Neural Network with 64, 128, 256, and 512 filters,
    paired 3x3 Conv2D layers, BatchNormalization, LeakyReLU, MaxPooling2D,
    progressive spatial Dropout, Dense 512 & 256 head, and 7-class Softmax output.
    """
    model = Sequential([
        # Block 1 - 64 filters
        Conv2D(64, (3, 3), padding="same", input_shape=input_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(64, (3, 3), padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2 - 128 filters
        Conv2D(128, (3, 3), padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(128, (3, 3), padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        # Block 3 - 256 filters
        Conv2D(256, (3, 3), padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(256, (3, 3), padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        # Block 4 - 512 filters
        Conv2D(512, (3, 3), padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(512, (3, 3), padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        # Classifier Head
        Flatten(),
        Dense(512),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(256),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    return model


def build_transfer_learning_model(model_type="MobileNetV2", input_shape=TRANSFER_LEARNING_SHAPE, num_classes=len(EMOTION_LABELS)):
    """
    Supports MobileNetV2 and ResNet50 pre-trained on ImageNet with GlobalAveragePooling2D,
    Dense 256, BatchNormalization, Dropout, and Softmax head.
    """
    inputs = Input(shape=input_shape)

    if model_type.lower() == "mobilenetv2":
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=inputs)
    elif model_type.lower() == "resnet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'MobileNetV2' or 'ResNet50'.")

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
