import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from PythonScripts.training.training import TrainingInformation, compute_metrics
from sklearn.utils import resample
from PythonScripts.dataset.custom_dataset import CustomDataset

class Bagging:
    def __init__(
        self,
        model_paths: list[str],
        num_model: int,
        dataset: CustomDataset,
        training_information: TrainingInformation
    ):
        self.model_paths = model_paths
        self.models 
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
            callbacks = 
        for i in range(self.num_model):
            sample = bootstrap_samples[i]
            model = AutoModelForSequenceClassification.from_pretrained(self.model_paths[i])
            
            trainer = Trainer()
            
            