import evaluate
import os
import gc
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import Counter
from custom_dataset import CustomDataset
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from dataclasses import dataclass
from datasets import Dataset
from sklearn.metrics import confusion_matrix

@dataclass
class TrainingInformation:
    pretrained_model: str
    epoch: int
    dataset_name: str

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

def save_training_result(results: dict[str, float], training_information: "TrainingInformation"\
    , dataset_augmented: bool, dataset_name: str, save_path='../Experiments/February14th/', balanced=True) -> str:
    '''
    Save results from evaluation after training using pretrained model.
    
    Args:
    - results: dictionary of the evaluation results (acc, precision, and other metrics)
    - training_information: training configuration that is going to be used for the file name.
    
    Returns:
    - bool: If the saving is successful or not.
    '''
    balance_string = "_balance" if balanced else ""
    augmented_string = "_Augmented" if dataset_augmented else "" 
    try:
        folder_name = f"results_{training_information.pretrained_model}_epoch{training_information.epoch}{augmented_string}_{dataset_name}{balance_string}"
        folder_name = folder_name.replace("/", "_").replace(" ", "_")
        folder_name = os.path.join(save_path, folder_name)
        os.makedirs(folder_name, exist_ok=True)
     
        filename = os.path.join(folder_name, "results.json")
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {filename}")
        return folder_name 
    except Exception as e:
        print(f'Error saving result: {e}')
        return ""

def train_model(dataset: CustomDataset, training_information: TrainingInformation):
    torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(training_information.pretrained_model, num_labels=dataset.count_unique_labels())
    
    if torch.cuda.is_available():
        print(f"Training model is using GPU {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
        model.to(device)
    else:
        print(f'Training model is using CPU')

    print_label_counts(dataset=dataset.train, label_col=dataset.settings.label_col)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=training_information.epoch,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train,
        eval_dataset=dataset.test,
        compute_metrics=compute_metrics
    )
    
    print(gc.collect())

    trainer.train()

    evaluation_result = trainer.evaluate(dataset.test)
    folder_path = save_training_result(evaluation_result, training_information,\
        dataset_augmented=dataset.data_loader.settings.with_augmentation\
            , dataset_name=dataset.data_loader.settings.dataset_link.replace("/", "").replace(".", ""))

    predictions = trainer.predict(dataset.test)
    eval_pred = (predictions.predictions, predictions.label_ids)
    
    class_names = dataset.get_class_names()
    generate_confusion_matrix(eval_pred, list(range(dataset.count_unique_labels())), os.path.join(folder_path, "confusion_matrix.jpeg"), class_names)

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