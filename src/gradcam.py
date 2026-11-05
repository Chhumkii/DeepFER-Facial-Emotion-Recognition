"""
Grad-CAM Explainability Module for Facial Emotion Recognition (FER)
Author: Chumki Nanda (chumkinanda066@gmail.com)
"""

import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes Grad-CAM heatmap for a given input image and model layer using tf.GradientTape.
    """
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Blends jet colormaps onto the face crop image.
    """
    if len(img.shape) == 2 or img.shape[-1] == 1:
        img_rgb = cv2.cvtColor(np.squeeze(img).astype('uint8'), cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img.astype('uint8')

    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    superimposed_img = color_heatmap * alpha + img_rgb * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    return superimposed_img

def generate_gradcam_comparison_plot(original_img, heatmap, overlay_img, pred_label, confidence, save_path=None):
    """
    Saves a 3-panel interpretability figure: [Original Image, Grad-CAM Heatmap, Overlay].
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    if len(original_img.shape) == 2 or original_img.shape[-1] == 1:
        axes[0].imshow(np.squeeze(original_img), cmap='gray')
    else:
        axes[0].imshow(original_img)
    axes[0].set_title("Original Face Crop")
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')

    axes[2].imshow(overlay_img)
    axes[2].set_title(f"Overlay ({pred_label}: {confidence*100:.1f}%)")
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Grad-CAM 3-panel comparison plot saved to {save_path}")
    plt.close()
    return fig
