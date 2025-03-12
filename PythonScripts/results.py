import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def generate_confusion_matrix(
    eval_pred, 
    labels: list[str], 
    save_path, 
    class_names=None, 
    labels_dropped:list[str]=[]
):
    """
    Generates and saves the confusion matrix as a .jpg file.

    Args:
    - eval_pred: Tuple containing (logits, labels).
    - labels: List of unique labels.
    - class_names: List of class names (optional).
    - save_path: File path to save the confusion matrix image.
    """
    class_names = [label for label in class_names if label not in labels_dropped]
    logits, true_labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    cm = confusion_matrix(true_labels, predictions, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig(save_path, format="jpg", dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
