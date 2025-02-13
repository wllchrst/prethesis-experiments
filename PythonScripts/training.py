from custom_dataset import CustomDataset
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from dataclasses import dataclass
import evaluate
import numpy as np

@dataclass
class TrainingInformation:
    pretrained_model: str

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
    print("EVALUATION RESULT")
    print(evaluation_result)