import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import Trainer

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

def get_training_history(trainer):
    """
    Extracts training and validation loss history from a Hugging Face Trainer object.

    Args:
        trainer (Trainer): A trained Hugging Face Trainer instance.

    Returns:
        dict: A dictionary containing:
            - 'train_loss': List of training loss values per epoch.
            - 'val_loss': List of validation loss values per epoch.
    """
    train_losses = []
    val_losses = []
    
    for log in trainer.state.log_history:
        if "loss" in log:  # Training loss
            train_losses.append(log["loss"])
        if "eval_loss" in log:  # Validation loss
            val_losses.append(log["eval_loss"])
    
    return {"train_loss": train_losses, "val_loss": val_losses}

def plot_training_history(trainer, save_path):
    """
    Plots training and validation loss history.

    Args:
        trainer (Trainer): Hugging Face Trainer instance.
        save_path (str): Path to save the plot.
    """
    history = get_training_history(trainer)  # Ensure correct history extraction

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    # Ensure lengths match
    min_length = min(len(train_loss), len(val_loss))

    if min_length == 0:
        print("Error: No valid training or validation loss data found.")
        return

    train_loss = train_loss[:min_length]
    val_loss = val_loss[:min_length]
    epochs = list(range(1, min_length + 1))  # Ensure x-axis matches loss values

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Training Loss", marker='o')
    plt.plot(epochs, val_loss, label="Validation Loss", marker='o')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.grid()

    plt.savefig(save_path)
    plt.close()