import evaluate
import os
import gc
import numpy as np
import torch
import pandas as pd
from custom_dataset import CustomDataset
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, EarlyStoppingCallback
from dataclasses import dataclass
from results import generate_confusion_matrix, plot_training_history
import json

@dataclass
class TrainingInformation:
    pretrained_model: str
    epoch: int
    dataset_name: str
    dropout_probability: float = 0.1
    learning_rate: float=2e-5
    weight_decay: float=0.01
    early_stopping_patience: float=0
    batch_size: float=8
    folder_name_from_info:bool=False

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

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

def train_model(
    dataset: CustomDataset, 
    training_information: TrainingInformation, 
    labels_dropped: list[str],
) -> None :
    torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(
        training_information.pretrained_model, num_labels=dataset.count_unique_labels()
    )
    
    model.config.hidden_dropout_prob = training_information.dropout_probability
    model.config.attention_probs_dropout_prob = training_information.dropout_probability
        
    if torch.cuda.is_available():
        print(f"Training model is using GPU {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
        model.to(device)
    else:
        print(f'Training model is using CPU')
    
    callbacks = []
    if training_information.early_stopping_patience > 0:
        early_stopping = EarlyStoppingCallback(early_stopping_patience=training_information.early_stopping_patience)
        callbacks.append(early_stopping)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=training_information.epoch,
        per_device_train_batch_size=training_information.batch_size,
        per_device_eval_batch_size=training_information.batch_size,
        warmup_steps=500,
        weight_decay=training_information.weight_decay,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        learning_rate=training_information.learning_rate,
        gradient_accumulation_steps=2,
        max_grad_norm=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train,
        eval_dataset=dataset.val,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    
    trainer.train()

    evaluation_result = trainer.evaluate(dataset.test)
    suggested_path = ''    
    if training_information.folder_name_from_info:
        suggested_path = f"{training_information.dataset_name}\
            _model-{training_information.pretrained_model.replace('/', '-')}\
                _epochs-{training_information.epoch}_dropout-{training_information.dropout_probability}\
                    _lr-{training_information.learning_rate}_wd-{training_information.weight_decay}\
                        _es-{training_information.early_stopping_patience}_bs-{training_information.batch_size}"
                        

    folder_path = save_training_result(evaluation_result, training_information,\
        dataset_augmented=dataset.data_loader.settings.with_augmentation\
            , dataset_name=dataset.data_loader.settings.dataset_link.replace("/", "").replace(".", ""), labels_dropped=labels_dropped\
                ,folder_path=suggested_path)

    predictions = trainer.predict(dataset.test)
    eval_pred = (predictions.predictions, predictions.label_ids)
    
    class_names = dataset.get_class_names()
    generate_confusion_matrix\
        (eval_pred, list(range(dataset.count_unique_labels())), os.path.join(folder_path, "confusion_matrix.jpeg"), class_names, labels_dropped=labels_dropped)
    
    plot_training_history(trainer, os.path.join(folder_path, "overfitting_analysis"))
        
        
def save_training_result(results: dict[str, float], training_information: "TrainingInformation"\
    , dataset_augmented: bool, dataset_name: str, save_path='../Experiments/March11th/', balanced=True,
    labels_dropped=[], folder_path='') -> str:
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
    labels_dropped_string = "dropped"
    for label in labels_dropped:
        labels_dropped_string += f'_{label}'
    try:
        folder_name = f"{training_information.pretrained_model}_epoch{training_information.epoch}{augmented_string}_{dataset_name}{balance_string}_{labels_dropped_string}"
        folder_name = folder_name.replace("/", "_").replace(" ", "_")
        folder_name = os.path.join(save_path, folder_name)
        
        folder_name = folder_name if folder_path == '' else f'{save_path}{folder_path}'
        folder_name = folder_name.replace(' ', '')
        print(f'folder name: {folder_name}')
        os.makedirs(folder_name, exist_ok=True)
     
        filename = os.path.join(folder_name, "results.json")
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {filename}")
        
        save_to_csv(results=results, save_path=save_path, training_information=training_information)
        
        return folder_name 
    except Exception as e:
        print(f'Error saving result: {e}')
        return ""

FILE_NAME = 'log.csv'

def save_to_csv(
    results: dict[str, float],
    save_path: str,
    training_information: TrainingInformation
):
    try:
        file_path = f'{save_path}{FILE_NAME}'

        existing_df = None

        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)

        data = []
        columns = []

        data.append(training_information.pretrained_model)
        columns.append('Model')

        data.append(training_information.dataset_name)
        columns.append('Dataset')

        data.append(training_information.epoch)
        columns.append('Epoch')

        data.append(training_information.batch_size)
        columns.append('Batch Size')

        data.append(training_information.dropout_probability)
        columns.append('Dropout')
        
        data.append(training_information.early_stopping_patience)
        columns.append('Early Stopping')

        data.append(training_information.learning_rate)
        columns.append('Learning Rate')

        data.append(training_information.weight_decay)
        columns.append('Weight Decay')

        # Append result
        for key in results:
            columns.append(key)
            data.append(results[key])

        df = pd.DataFrame([data], columns=columns)

        # Jika ada data lama, gabungkan
        if existing_df is not None:
            existing_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            existing_df = df
            
        existing_df.to_csv(file_path)

    except Exception as e:
        print(f"ERROR SAVING TO CSV {e}")