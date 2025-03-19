import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from PythonScripts.training.training import TrainingInformation, compute_metrics
from sklearn.utils import resample
from PythonScripts.dataset.custom_dataset import CustomDataset
from datasets import Dataset


class Bagging:
    def __init__(
        self,
        model_paths: list[str],
        num_model: int,
        dataset: CustomDataset,
        training_information: TrainingInformation
    ):
        self.model_paths = model_paths
        self.models = []
        self.num_model = num_model
        self.training_information = training_information
        self.dataset = dataset
    
    def train(self):
        train_dataset = self.dataset.train
        dataset_size = len(train_dataset)

        bootstrap_indices = [
            np.random.choice(dataset_size, size=dataset_size, replace=True).tolist()
            for _ in range(self.num_model)
        ]

        bootstrap_samples = [train_dataset.select(indices) for indices in bootstrap_indices]

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

        callbacks = []
        if self.training_information.early_stopping_patience != 0:
            callbacks = EarlyStoppingCallback(self.training_information.early_stopping_patience)

        for i in range(self.num_model):
            sample = bootstrap_samples[i]
            model = AutoModelForSequenceClassification.from_pretrained(self.model_paths[i])

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=sample,
                eval_dataset=self.dataset.val,
                compute_metrics=compute_metrics,
                callbacks=callbacks
            )
            
            trainer.train()
            
            self.models.append(model)


    def predict(self, aggregation="majority"):
        """
        Perform inference using ensemble models.

        Args:
        - test_dataset: Dataset to predict.
        - aggregation: "majority" for majority voting, "average" for probability averaging.

        Returns:
        - List of final predictions.
        """
        test_dataset = self.dataset.test

        if not self.models:
            raise ValueError("Models are not trained yet!")

        all_predictions = []
        
        for model in self.models:
            trainer = Trainer(model=model)
            predictions = trainer.predict(test_dataset).predictions
            all_predictions.append(predictions)

        all_predictions = np.array(all_predictions)  # Shape: (num_models, num_samples, num_classes)

        if aggregation == "majority":
            final_predictions = np.argmax(np.sum(all_predictions, axis=0), axis=1)
        elif aggregation == "average":
            final_predictions = np.argmax(np.mean(all_predictions, axis=0), axis=1)
        else:
            raise ValueError("Invalid aggregation method. Use 'majority' or 'average'.")

        return final_predictions
