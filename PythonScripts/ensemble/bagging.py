from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from PythonScripts.training.training import TrainingInformation, compute_metrics

class Bagging:
    def __init__(
        self,
        base_model: str,
        num_model: int,
        dataset,
        training_information: TrainingInformation
    ):
        self.models = []
        self.base_model = base_model
        self.num_model = num_model
        self.training_information = training_information
        pass
    
    def train(self):
        for _ in range(self.num_model):
            model = AutoModelForSequenceClassification.from_pretrained(self.base_model)
            callbacks = []
            
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=self.training_information.epoch,
                per_device_train_batch_size=self.training_information.batch_size,
                per_device_eval_batch_size=self.training_information.batch_size,
                warmup_steps=500,
                weight_decay=self.training_information.weight_decay,
                logging_dir="./logs",
                logging_steps=10,
                eval_strategy="epoch",
                logging_strategy="epoch",
                save_strategy="epoch",
                report_to="none",
                learning_rate=self.training_information.learning_rate,
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