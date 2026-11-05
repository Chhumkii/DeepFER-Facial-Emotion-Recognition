"""
Evaluation Module for Facial Emotion Recognition (FER)
Author: Chumki Nanda (chumkinanda066@gmail.com)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score
)
from src.config import CLASS_NAMES, NUM_CLASSES, PLOTS_DIR

def evaluate_model(model, val_dataset, class_names=CLASS_NAMES):
    """
    Evaluates the trained model on validation/test dataset.
    Computes predictions, ground truth labels, accuracy, macro/weighted F1, and classification report.
    """
    y_true = []
    y_pred = []

    for images, labels in val_dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    macro_f1 = np.mean(f1)
    weighted_f1 = precision_recall_fscore_support(y_true, y_pred, average='weighted')[2]

    metrics = {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'classification_report': classification_report(y_true, y_pred, target_names=class_names)
    }

    print("=== Evaluation Summary ===")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Macro F1-Score:   {macro_f1:.4f}")
    print(f"Weighted F1-Score:{weighted_f1:.4f}")
    print("\nClassification Report:\n", metrics['classification_report'])

    return y_true, y_pred, metrics

def plot_confusion_matrix(y_true, y_pred, class_names=CLASS_NAMES, save_path=None, normalize=True):
    """
    Computes and plots normalized (or raw) confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        data_to_plot = cm_norm
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        data_to_plot = cm
        title = 'Confusion Matrix'

    plt.figure(figsize=(8, 6))
    sns.heatmap(data_to_plot, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    plt.close()

def analyze_misclassifications(y_true, y_pred, class_names=CLASS_NAMES, top_k=5):
    """
    Identifies top misclassification error pairs (True vs Predicted).
    """
    cm = confusion_matrix(y_true, y_pred)
    misclass_pairs = []

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                count = cm[i, j]
                misclass_pairs.append((class_names[i], class_names[j], int(count)))

    misclass_pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop {top_k} Misclassification Error Pairs:")
    for rank, (true_cls, pred_cls, count) in enumerate(misclass_pairs[:top_k], 1):
        print(f"{rank}. True: '{true_cls}' --> Predicted as: '{pred_cls}' (Count: {count})")

    return misclass_pairs[:top_k]
