import evaluate
import os
import json
import numpy as np
from collections import Counter
from custom_dataset import CustomDataset
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from dataclasses import dataclass
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

@dataclass
class TrainingInformation:
    pretrained_model: str

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def print_label_counts(dataset: Dataset, label_col='label'):
    '''
    Prints the count of each label in the dataset.

    Args:
    - dataset: Dataset, the dataset to analyze.
    - label_col: str, column name containing labels.
    '''
    label_counts = Counter(dataset[label_col])
    print("Label counts:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average="macro")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"]
    }

def save_training_result(results: dict[str, float], training_information: "TrainingInformation") -> bool:
    '''
    Save results from evaluation after training using pretrained model.
    
    Args:
    - results: dictionary of the evaluation results (acc, precision, and other metrics)
    - training_information: training configuration that is going to be used for the file name.
    
    Returns:
    - bool: If the saving is successful or not.
    '''
    try:
        folder_name = f"results_{training_information.pretrained_model}_epoch{training_information.epoch}"
        folder_name = folder_name.replace("/", "_").replace(" ", "_")  # Ensure safe folder name
        os.makedirs(folder_name, exist_ok=True)
        
        filename = os.path.join(folder_name, "results.json")
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {filename}")
        return True
    except Exception as e:
        print(f'Error saving result: {e}')
        return False

def train_model(dataset: CustomDataset, training_information: TrainingInformation):
    model = AutoModelForSequenceClassification.from_pretrained(training_information.pretrained_model, num_labels=dataset.count_unique_labels())

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",        # ✅ Saves the model after each epoch
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train,
        eval_dataset=dataset.test,
        compute_metrics=compute_metrics
    )
    # trainer.add_callback(LoggingCallback())

    trainer.train()

    evaluation_result = trainer.evaluate(dataset.test)
    print(evaluation_result)
    
    # Get predictions and labels
    predictions = trainer.predict(dataset.test)
    eval_pred = (predictions.predictions, predictions.label_ids)
    
    class_names = dataset.get_class_names()
    generate_confusion_matrix(eval_pred, list(range(dataset.count_unique_labels())), "./results/confusion_matrix.jpeg") # TODO: Correct result path

def generate_confusion_matrix(eval_pred, labels, save_path, class_names=None):
    """
    Generates and saves the confusion matrix as a .jpg file.

    Args:
    - eval_pred: Tuple containing (logits, labels).
    - labels: List of unique labels.
    - class_names: List of class names (optional).
    - save_path: File path to save the confusion matrix image.
    """
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