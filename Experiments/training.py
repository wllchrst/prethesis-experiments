from custom_dataset import CustomDataset
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from dataclasses import dataclass

@dataclass
class TrainingInformation:
    pretrained_model: str

def train_model(dataset: CustomDataset, training_information: TrainingInformation):
    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=dataset.count_unique_labels())

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train,
        eval_dataset=dataset.test  # Use a separate validation dataset in real training
    )

    trainer.train()

    trainer.evaluate()